from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.config import AppConfig


class FaissIndexBuilder:
    """
    Class hóa toàn bộ logic của script build_faiss_index.py cũ.

    Interface công khai được giữ nguyên:
        builder = FaissIndexBuilder(config)
        index_path = builder.build()

    FAISS ID luôn bằng embedding_row, vì vậy HybridRetriever hiện tại vẫn có
    thể ánh xạ kết quả bằng embedding_metadata.iloc[faiss_id].
    """

    _METRIC_ALIASES = {
        "ip": "cosine",
        "inner_product": "cosine",
        "cos": "cosine",
        "cosine": "cosine",
        "l2": "l2",
        "euclidean": "l2",
    }

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.settings = config.section("faiss")
        self.logger = logging.getLogger(self.__class__.__name__)

        # Giữ base index sống trong suốt quá trình build. Một số phiên bản
        # Python binding của FAISS chỉ giữ con trỏ C++ trong IndexIDMap2; nếu
        # base index bị garbage-collect sớm có thể gây segmentation fault.
        self._base_index_reference: Any | None = None

    @staticmethod
    def _require_faiss() -> Any:
        try:
            import faiss
        except ImportError as error:
            raise RuntimeError(
                "Thiếu faiss-cpu. Cài bằng: pip install faiss-cpu"
            ) from error

        return faiss

    @staticmethod
    def _save_json_atomic(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")

        try:
            with temporary.open("w", encoding="utf-8") as file:
                json.dump(data, file, ensure_ascii=False, indent=2)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink(missing_ok=True)

    @staticmethod
    def _load_embeddings(path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy embedding: {path.resolve()}"
            )

        embeddings = np.load(path, mmap_mode="r")

        if embeddings.ndim != 2:
            raise ValueError(
                "embeddings.npy phải có 2 chiều, "
                f"nhận được shape={embeddings.shape}."
            )

        if embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
            raise ValueError("embeddings.npy rỗng.")

        if not np.issubdtype(embeddings.dtype, np.floating):
            raise TypeError(
                "Embedding phải là số thực, "
                f"nhận được dtype={embeddings.dtype}."
            )

        return embeddings

    @staticmethod
    def _check_metadata(
        metadata_path: Path,
        embedding_rows: int,
    ) -> int:
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy metadata: {metadata_path.resolve()}"
            )

        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise RuntimeError(
                "Thiếu pyarrow để kiểm tra metadata. "
                "Cài bằng: pip install pyarrow"
            ) from error

        parquet_file = pq.ParquetFile(metadata_path)
        metadata_rows = int(parquet_file.metadata.num_rows)

        if metadata_rows != embedding_rows:
            raise ValueError(
                "Số dòng metadata không bằng số embedding: "
                f"metadata={metadata_rows:,}, "
                f"embeddings={embedding_rows:,}."
            )

        return metadata_rows

    @classmethod
    def _normalize_metric_name(cls, value: Any) -> str:
        raw = str(value or "cosine").strip().casefold()

        if raw not in cls._METRIC_ALIASES:
            raise ValueError(
                "faiss.metric chỉ nhận một trong: "
                "ip, inner_product, cosine, l2, euclidean. "
                f"Nhận được: {value!r}."
            )

        return cls._METRIC_ALIASES[raw]

    @staticmethod
    def _normalize_index_type(value: Any) -> str:
        index_type = str(value or "flat").strip().casefold()

        if index_type not in {"flat", "hnsw"}:
            raise ValueError(
                "faiss.index_type chỉ nhận 'flat' hoặc 'hnsw', "
                f"nhận được {value!r}."
            )

        return index_type

    def _create_index(
        self,
        faiss: Any,
        dimension: int,
        index_type: str,
        metric: str,
        hnsw_m: int,
        ef_construction: int,
        ef_search: int,
    ) -> Any:
        metric_type = (
            faiss.METRIC_INNER_PRODUCT
            if metric == "cosine"
            else faiss.METRIC_L2
        )

        if index_type == "flat":
            base_index = (
                faiss.IndexFlatIP(dimension)
                if metric == "cosine"
                else faiss.IndexFlatL2(dimension)
            )
        else:
            if hnsw_m <= 0:
                raise ValueError("faiss.hnsw_m phải lớn hơn 0.")
            if ef_construction <= 0:
                raise ValueError("faiss.ef_construction phải lớn hơn 0.")
            if ef_search <= 0:
                raise ValueError("faiss.ef_search phải lớn hơn 0.")

            base_index = faiss.IndexHNSWFlat(
                dimension,
                hnsw_m,
                metric_type,
            )
            base_index.hnsw.efConstruction = ef_construction
            base_index.hnsw.efSearch = ef_search

        # FAISS ID được gán tường minh bằng embedding_row.
        # Giữ reference Python tới base index để tránh con trỏ treo.
        self._base_index_reference = base_index
        return faiss.IndexIDMap2(base_index)

    @staticmethod
    def _validate_sample(embeddings: np.ndarray) -> dict[str, float | int]:
        row_count = int(embeddings.shape[0])
        sample_count = min(row_count, 1000)
        indices = np.linspace(
            0,
            row_count - 1,
            num=sample_count,
            dtype=np.int64,
        )

        sample = np.asarray(embeddings[indices], dtype=np.float32)

        if not np.isfinite(sample).all():
            raise ValueError("Embedding chứa NaN hoặc Inf.")

        norms = np.linalg.norm(sample, axis=1)

        return {
            "sample_count": int(sample_count),
            "minimum_norm_before_index": float(norms.min()),
            "mean_norm_before_index": float(norms.mean()),
            "maximum_norm_before_index": float(norms.max()),
        }

    def _add_embeddings(
        self,
        faiss: Any,
        index: Any,
        embeddings: np.ndarray,
        batch_size: int,
        normalize: bool,
    ) -> None:
        row_count = int(embeddings.shape[0])

        if batch_size <= 0:
            raise ValueError("faiss.add_batch_size phải lớn hơn 0.")

        started = time.time()

        for start in range(0, row_count, batch_size):
            end = min(start + batch_size, row_count)

            # np.load(..., mmap_mode="r") trả về vùng nhớ read-only.
            # FAISS normalize_L2 sửa trực tiếp mảng, nên phải copy sang mảng
            # C-contiguous có quyền ghi; ascontiguousarray có thể giữ read-only.
            vectors = np.array(
                embeddings[start:end],
                dtype=np.float32,
                order="C",
                copy=True,
            )

            if not np.isfinite(vectors).all():
                raise ValueError(
                    "Phát hiện NaN hoặc Inf trong khoảng dòng "
                    f"[{start}, {end})."
                )

            if normalize:
                faiss.normalize_L2(vectors)

            ids = np.arange(start, end, dtype=np.int64)
            index.add_with_ids(vectors, ids)

            elapsed = max(time.time() - started, 1e-9)
            speed = end / elapsed

            self.logger.info(
                "Đã thêm %s/%s vector (%.2f%%) | %.2f vector/giây",
                f"{end:,}",
                f"{row_count:,}",
                100.0 * end / row_count,
                speed,
            )

    @staticmethod
    def _verify_index(
        index: Any,
        embeddings: np.ndarray,
        normalize: bool,
        faiss: Any,
    ) -> dict[str, Any]:
        if int(index.ntotal) != int(embeddings.shape[0]):
            raise RuntimeError(
                f"Index chứa {int(index.ntotal):,} vector, "
                f"nhưng embedding có {int(embeddings.shape[0]):,} dòng."
            )

        probe_count = min(5, int(embeddings.shape[0]))
        queries = np.array(
            embeddings[:probe_count],
            dtype=np.float32,
            order="C",
            copy=True,
        )

        if normalize:
            faiss.normalize_L2(queries)

        scores, ids = index.search(queries, 1)
        expected_ids = np.arange(probe_count, dtype=np.int64)
        returned_ids = ids[:, 0]

        return {
            "ntotal": int(index.ntotal),
            "probe_count": int(probe_count),
            "self_retrieval_ids": returned_ids.astype(int).tolist(),
            "expected_ids": expected_ids.astype(int).tolist(),
            "self_retrieval_passed": bool(
                np.array_equal(returned_ids, expected_ids)
            ),
            "self_retrieval_scores": scores[:, 0].astype(float).tolist(),
        }

    @staticmethod
    def _write_index_atomic(
        faiss: Any,
        index: Any,
        output: Path,
    ) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")

        try:
            faiss.write_index(index, str(temporary))
            os.replace(temporary, output)
        finally:
            if temporary.exists():
                temporary.unlink(missing_ok=True)

    def _manifest_path(self, index_path: Path) -> Path:
        configured = self.settings.get("manifest_path")

        if configured:
            path = Path(str(configured))
            return path if path.is_absolute() else self.config.root / path

        name = str(
            self.settings.get("manifest_name", "faiss_manifest.json")
        ).strip()

        if not name:
            raise ValueError("faiss.manifest_name không được để trống.")

        return index_path.parent / name

    def build(self) -> Path:
        """Xây FAISS index bằng logic cũ và trả về đường dẫn index."""

        faiss = self._require_faiss()

        embeddings_path = self.config.path("embeddings")
        metadata_path = self.config.path("embedding_metadata")
        index_path = self.config.path("faiss_index")
        manifest_path = self._manifest_path(index_path)

        index_type = self._normalize_index_type(
            self.settings.get("index_type", "flat")
        )
        metric = self._normalize_metric_name(
            self.settings.get("metric", "cosine")
        )
        add_batch_size = int(
            self.settings.get(
                "add_batch_size",
                self.settings.get("batch_size", 10_000),
            )
        )
        hnsw_m = int(self.settings.get("hnsw_m", 32))
        ef_construction = int(
            self.settings.get("ef_construction", 200)
        )
        ef_search = int(self.settings.get("ef_search", 128))

        # Logic cũ: chỉ normalize khi dùng cosine và không skip normalize.
        normalize_enabled = bool(self.settings.get("normalize", True))
        skip_normalize = bool(
            self.settings.get("skip_normalize", False)
        )
        normalize = (
            metric == "cosine"
            and normalize_enabled
            and not skip_normalize
        )

        # Hỗ trợ cả tên dương và tên skip của script cũ.
        check_metadata = bool(
            self.settings.get("check_metadata", True)
        ) and not bool(
            self.settings.get("skip_metadata_check", False)
        )
        overwrite = bool(self.settings.get("overwrite", False))

        if bool(self.settings.get("use_gpu", False)):
            self.logger.warning(
                "faiss.use_gpu=true bị bỏ qua vì logic builder cũ xây và "
                "lưu CPU index. Retrieval vẫn có thể đọc index bình thường."
            )

        if index_path.exists() and not overwrite:
            raise FileExistsError(
                f"Index đã tồn tại: {index_path.resolve()}. "
                "Đặt faiss.overwrite=true để ghi đè."
            )

        embeddings = self._load_embeddings(embeddings_path)
        row_count, dimension = map(int, embeddings.shape)

        self.logger.info("Embeddings: %s", embeddings_path.resolve())
        self.logger.info("Shape: (%s, %s)", f"{row_count:,}", dimension)
        self.logger.info("Index type: %s", index_type)
        self.logger.info("Metric: %s", metric)
        self.logger.info("Normalize trước khi add: %s", normalize)

        metadata_rows: int | None = None

        if check_metadata:
            metadata_rows = self._check_metadata(
                metadata_path=metadata_path,
                embedding_rows=row_count,
            )
            self.logger.info(
                "Metadata hợp lệ: %s dòng",
                f"{metadata_rows:,}",
            )

        sample_validation = self._validate_sample(embeddings)

        index = self._create_index(
            faiss=faiss,
            dimension=dimension,
            index_type=index_type,
            metric=metric,
            hnsw_m=hnsw_m,
            ef_construction=ef_construction,
            ef_search=ef_search,
        )

        self._add_embeddings(
            faiss=faiss,
            index=index,
            embeddings=embeddings,
            batch_size=add_batch_size,
            normalize=normalize,
        )

        verification = self._verify_index(
            index=index,
            embeddings=embeddings,
            normalize=normalize,
            faiss=faiss,
        )

        self._write_index_atomic(
            faiss=faiss,
            index=index,
            output=index_path,
        )

        manifest = {
            "embeddings_path": str(embeddings_path.resolve()),
            "metadata_path": (
                str(metadata_path.resolve()) if check_metadata else None
            ),
            "index_path": str(index_path.resolve()),
            "index_type": index_type,
            "metric": metric,
            "configured_metric": self.settings.get("metric", "cosine"),
            "normalized_before_add": normalize,
            "row_count": row_count,
            "dimension": dimension,
            "dtype_source": str(embeddings.dtype),
            "metadata_rows": metadata_rows,
            "add_batch_size": add_batch_size,
            "hnsw_m": hnsw_m if index_type == "hnsw" else None,
            "ef_construction": (
                ef_construction if index_type == "hnsw" else None
            ),
            "ef_search": ef_search if index_type == "hnsw" else None,
            "sample_validation": sample_validation,
            "verification": verification,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._save_json_atomic(manifest_path, manifest)

        self.logger.info(
            "Đã lưu FAISS index: %s | vectors=%s | dimension=%s",
            index_path.resolve(),
            f"{row_count:,}",
            dimension,
        )
        self.logger.info(
            "Đã lưu FAISS manifest: %s | self-test=%s",
            manifest_path.resolve(),
            "PASSED"
            if verification["self_retrieval_passed"]
            else "WARNING",
        )

        return index_path


class BM25IndexBuilder:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def build(self) -> Path:
        import bm25s

        settings = self.config.section("bm25")
        frame = pd.read_parquet(self.config.path("legal_chunks"))
        texts = (
            frame[settings["text_column"]]
            .fillna("")
            .astype(str)
            .tolist()
        )
        corpus_tokens = bm25s.tokenize(texts, stopwords=None)
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)

        output_dir = self.config.path("bm25_dir")
        output_dir.mkdir(parents=True, exist_ok=True)
        retriever.save(str(output_dir))
        frame.to_parquet(
            self.config.path("bm25_lookup"),
            index=False,
        )

        self.logger.info(
            "Saved BM25 index with %d documents",
            len(frame),
        )
        return output_dir
