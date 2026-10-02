from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from core.config import AppConfig


TEXT_COLUMNS = [
    "document_number",
    "title",
    "part",
    "chapter",
    "section",
    "articles",
    "content",
]


class QwenEmbeddingModel:
    """
    Class wrapper cho logic embedding Qwen3 cũ.

    Interface bên ngoài được giữ nguyên:
        model = QwenEmbeddingModel(config).load()
        vectors = model.encode([query])

    Quy ước nội bộ:
    - encode(...) được dùng cho query và tự áp dụng prompt_name="query".
    - EmbeddingBuilder gọi _encode_documents(...) để embedding document
      mà không thêm query instruction.
    - SentenceTransformer tự dùng pooling đúng được đóng gói cùng model.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.settings = config.section("embedding")
        self.logger = logging.getLogger(self.__class__.__name__)

        # Giữ các thuộc tính cũ để không làm hỏng code đang truy cập chúng.
        self.tokenizer: Any | None = None
        self.model: Any | None = None
        self.device = "cpu"
        self.full_dimension: int | None = None

    @staticmethod
    def _safe_text(value: Any) -> str:
        if value is None:
            return ""

        if isinstance(value, (float, np.floating)) and np.isnan(value):
            return ""

        text = str(value).strip()
        if text.casefold() in {"", "none", "null", "nan"}:
            return ""

        return text

    @classmethod
    def _prepare_texts(cls, texts: list[str]) -> list[str]:
        return [cls._safe_text(text) or "[EMPTY LEGAL CHUNK]" for text in texts]

    def _choose_device(self, torch_module: Any) -> str:
        requested = str(self.settings.get("device", "auto")).casefold()

        if requested not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError(
                "embedding.device chỉ nhận auto, cpu, cuda hoặc mps."
            )

        if requested == "cuda":
            if not torch_module.cuda.is_available():
                raise RuntimeError("PyTorch không phát hiện GPU CUDA.")
            return "cuda"

        if requested == "mps":
            available = (
                hasattr(torch_module.backends, "mps")
                and torch_module.backends.mps.is_available()
            )
            if not available:
                raise RuntimeError("Thiết bị hiện tại không hỗ trợ MPS.")
            return "mps"

        if requested == "cpu":
            return "cpu"

        if torch_module.cuda.is_available():
            return "cuda"

        mps_available = (
            hasattr(torch_module.backends, "mps")
            and torch_module.backends.mps.is_available()
        )
        return "mps" if mps_available else "cpu"

    def _resolve_model_source(self) -> str:
        local_dir = self.settings.get("local_model_dir")
        if local_dir:
            local_path = self.config.root / str(local_dir)
            if local_path.exists() and any(local_path.iterdir()):
                return str(local_path)

        model_name = self.settings.get(
            "model_name",
            "Qwen/Qwen3-Embedding-0.6B",
        )
        return str(model_name)

    def load(self) -> "QwenEmbeddingModel":
        try:
            import torch
        except ImportError as error:
            raise RuntimeError(
                "Thiếu torch. Cài bằng: pip install torch"
            ) from error

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Thiếu sentence-transformers. Cài bằng: "
                'pip install "sentence-transformers>=2.7.0" '
                '"transformers>=4.51.0"'
            ) from error

        self.device = self._choose_device(torch)
        source = self._resolve_model_source()

        kwargs: dict[str, Any] = {
            "device": self.device,
        }

        cache_dir = self.settings.get("cache_dir")
        if cache_dir:
            cache_path = Path(str(cache_dir))
            if not cache_path.is_absolute():
                cache_path = self.config.root / cache_path
            kwargs["cache_folder"] = str(cache_path)

        if "local_files_only" in self.settings:
            kwargs["local_files_only"] = bool(
                self.settings["local_files_only"]
            )

        self.logger.info("Đang load embedding model: %s", source)
        self.model = SentenceTransformer(source, **kwargs)
        self.model.max_seq_length = int(
            self.settings.get("max_length", 512)
        )

        # Giữ tương thích với thuộc tính tokenizer của implementation cũ.
        self.tokenizer = getattr(self.model, "tokenizer", None)

        dimension = self.model.get_sentence_embedding_dimension()
        if dimension is None:
            probe = self.model.encode(
                ["Kiểm tra kích thước embedding"],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                device=self.device,
            )
            dimension = int(probe.shape[1])

        self.full_dimension = int(dimension)
        self.logger.info(
            "Embedding model đã load | device=%s | max_length=%d | dim=%d",
            self.device,
            self.model.max_seq_length,
            self.full_dimension,
        )
        return self

    def _target_dimension(self) -> int:
        if self.full_dimension is None:
            self.load()

        assert self.full_dimension is not None
        target = int(
            self.settings.get(
                "embedding_dim",
                self.settings.get("dimension", self.full_dimension),
            )
        )

        if not 1 <= target <= self.full_dimension:
            raise ValueError(
                f"embedding_dim={target} không hợp lệ; "
                f"dimension của model là {self.full_dimension}."
            )

        return target

    @staticmethod
    def _truncate_and_normalize(
        vectors: np.ndarray,
        target_dimension: int,
    ) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float32)

        if vectors.ndim != 2:
            raise RuntimeError(
                f"Embedding phải có 2 chiều, nhận được {vectors.shape}."
            )

        if target_dimension > vectors.shape[1]:
            raise ValueError(
                f"embedding_dim={target_dimension} lớn hơn dimension "
                f"model={vectors.shape[1]}."
            )

        if target_dimension < vectors.shape[1]:
            vectors = vectors[:, :target_dimension]

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return np.asarray(vectors / norms, dtype=np.float32)

    def _encode_internal(
        self,
        texts: list[str],
        *,
        prompt_name: str | None,
    ) -> np.ndarray:
        if self.model is None:
            self.load()

        assert self.model is not None

        clean_texts = self._prepare_texts(texts)
        if not clean_texts:
            return np.empty(
                (0, self._target_dimension()),
                dtype=np.float32,
            )

        encode_kwargs: dict[str, Any] = {
            "batch_size": int(self.settings.get("batch_size", 32)),
            "convert_to_numpy": True,
            "normalize_embeddings": True,
            "show_progress_bar": False,
            "device": self.device,
        }

        if prompt_name:
            encode_kwargs["prompt_name"] = prompt_name

        vectors = self.model.encode(clean_texts, **encode_kwargs)
        return self._truncate_and_normalize(
            vectors,
            target_dimension=self._target_dimension(),
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Giữ nguyên interface cũ của retriever.

        Hàm này được coi là query embedding, vì nơi gọi công khai hiện tại là:
            self.embedder.encode([query])
        """
        prompt_name = str(
            self.settings.get("query_prompt_name", "query")
        ).strip()
        return self._encode_internal(
            texts,
            prompt_name=prompt_name or None,
        )

    def _encode_documents(self, texts: list[str]) -> np.ndarray:
        """Document embedding không dùng query instruction."""
        return self._encode_internal(texts, prompt_name=None)


class EmbeddingBuilder:
    """
    Class hóa logic của embed_legal_chunks_qwen3_manual.py.

    Interface bên ngoài vẫn là:
        EmbeddingBuilder(config).build() -> (embeddings_path, metadata_path)
    """

    def __init__(
        self,
        config: AppConfig,
        model: QwenEmbeddingModel | None = None,
    ) -> None:
        self.config = config
        self.model = model or QwenEmbeddingModel(config)
        self.settings = config.section("embedding")
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def _save_json(path: Path, data: dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        os.replace(temporary, path)

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _safe_text(value: Any) -> str:
        return QwenEmbeddingModel._safe_text(value)

    def _build_contextual_text(self, record: dict[str, Any]) -> str:
        hierarchy = [
            ("Số hiệu văn bản", record.get("document_number")),
            ("Tên văn bản", record.get("title")),
            ("Phần", record.get("part")),
            ("Chương", record.get("chapter")),
            ("Mục", record.get("section")),
            ("Điều", record.get("articles")),
        ]

        parts: list[str] = []
        for label, value in hierarchy:
            normalized = self._safe_text(value)
            if normalized:
                parts.append(f"{label}: {normalized}")

        text_column = str(self.settings.get("text_column", "content"))
        content = self._safe_text(record.get(text_column))
        if content:
            parts.append(f"Nội dung:\n{content}")

        text = "\n".join(parts).strip()
        return text or "[EMPTY LEGAL CHUNK]"

    def _batch_to_texts(
        self,
        batch: pa.RecordBatch,
        text_mode: str,
    ) -> list[str]:
        columns = batch.to_pydict()
        text_column = str(self.settings.get("text_column", "content"))
        texts: list[str] = []

        for row_index in range(batch.num_rows):
            if text_mode == "content":
                text = self._safe_text(columns[text_column][row_index])
                texts.append(text or "[EMPTY LEGAL CHUNK]")
                continue

            record = {
                column: values[row_index]
                for column, values in columns.items()
            }
            texts.append(self._build_contextual_text(record))

        return texts

    def _validate_input(self, path: Path) -> tuple[list[str], int]:
        if not path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file: {path.resolve()}\n"
                f"Thư mục chạy hiện tại: {Path.cwd()}"
            )

        parquet_file = pq.ParquetFile(path)
        columns = parquet_file.schema_arrow.names

        id_column = str(self.settings.get("id_column", "id"))
        text_column = str(self.settings.get("text_column", "content"))
        missing = [
            column
            for column in (id_column, text_column)
            if column not in columns
        ]

        if missing:
            raise ValueError(
                f"File thiếu cột bắt buộc: {missing}. "
                f"Các cột hiện có: {columns}"
            )

        row_count = int(parquet_file.metadata.num_rows)
        if row_count <= 0:
            raise ValueError("File legal_chunks.parquet không có dòng dữ liệu.")

        return list(columns), row_count

    def _export_metadata(
        self,
        input_path: Path,
        output_path: Path,
        available_columns: list[str],
        overwrite: bool,
    ) -> None:
        if output_path.exists() and not overwrite:
            existing_file = pq.ParquetFile(output_path)
            existing_rows = existing_file.metadata.num_rows
            existing_columns = existing_file.schema_arrow.names
            input_rows = pq.ParquetFile(input_path).metadata.num_rows
            if (
                existing_rows == input_rows
                and "embedding_row" in existing_columns
            ):
                self.logger.info("Giữ metadata đã có: %s", output_path)
                return

        # Giữ schema metadata của class mới để downstream không bị thay đổi,
        # đồng thời thêm embedding_row giống script cũ.
        selected_columns = [
            column
            for column in available_columns
            if column != "embedding"
        ]
        table = pq.read_table(input_path, columns=selected_columns)

        if "embedding_row" in table.column_names:
            index = table.column_names.index("embedding_row")
            table = table.remove_column(index)

        embedding_rows = pa.array(
            np.arange(table.num_rows, dtype=np.int64)
        )
        table = table.add_column(0, "embedding_row", embedding_rows)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, output_path, compression="zstd")
        self.logger.info(
            "Đã lưu metadata: %s (%s dòng)",
            output_path,
            f"{table.num_rows:,}",
        )

    def _prepare_storage(
        self,
        embeddings_path: Path,
        state_path: Path,
        row_count: int,
        embedding_dim: int,
        model_source: str,
        input_path: Path,
        overwrite: bool,
        state_signature: dict[str, Any],
    ) -> tuple[np.memmap, dict[str, Any]]:
        if overwrite:
            embeddings_path.unlink(missing_ok=True)
            state_path.unlink(missing_ok=True)

        previous_state = self._load_json(state_path)

        if embeddings_path.exists() and previous_state is not None:
            expected = {
                "row_count": row_count,
                "embedding_dim": embedding_dim,
                "model": model_source,
                **state_signature,
            }

            for key, expected_value in expected.items():
                actual_value = previous_state.get(key)
                if actual_value != expected_value:
                    raise RuntimeError(
                        "Checkpoint không khớp cấu hình hiện tại. "
                        "Đặt embedding.overwrite=true hoặc xóa output cũ.\n"
                        f"{key}: cũ={actual_value!r}, mới={expected_value!r}"
                    )

            embeddings = np.lib.format.open_memmap(
                embeddings_path,
                mode="r+",
                dtype=np.float32,
                shape=(row_count, embedding_dim),
            )
            return embeddings, previous_state

        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        embeddings = np.lib.format.open_memmap(
            embeddings_path,
            mode="w+",
            dtype=np.float32,
            shape=(row_count, embedding_dim),
        )

        state = {
            "input_path": str(input_path.resolve()),
            "model": model_source,
            "row_count": row_count,
            "embedding_dim": embedding_dim,
            **state_signature,
            "next_row": 0,
            "completed": False,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._save_json(state_path, state)
        return embeddings, state

    def _embed_parquet(
        self,
        input_path: Path,
        embeddings: np.memmap,
        state: dict[str, Any],
        state_path: Path,
        text_columns: list[str],
        text_mode: str,
        read_batch_size: int,
    ) -> None:
        parquet_file = pq.ParquetFile(input_path)
        total_rows = int(parquet_file.metadata.num_rows)
        next_row = int(state.get("next_row", 0))
        resume_from = next_row

        if next_row >= total_rows:
            self.logger.info("Embedding đã hoàn tất từ lần chạy trước.")
            return

        self.logger.info(
            "Bắt đầu embedding từ dòng %s/%s",
            f"{next_row:,}",
            f"{total_rows:,}",
        )

        global_start = 0
        started = time.time()

        for record_batch in parquet_file.iter_batches(
            batch_size=read_batch_size,
            columns=text_columns,
            use_threads=True,
        ):
            original_batch_size = record_batch.num_rows
            global_end = global_start + original_batch_size

            if global_end <= next_row:
                global_start = global_end
                continue

            local_start = max(0, next_row - global_start)
            if local_start:
                record_batch = record_batch.slice(
                    local_start,
                    original_batch_size - local_start,
                )

            texts = self._batch_to_texts(record_batch, text_mode)
            output_start = global_start + local_start
            output_end = output_start + len(texts)

            vectors = self.model._encode_documents(texts)
            embeddings[output_start:output_end] = vectors
            embeddings.flush()

            next_row = output_end
            state["next_row"] = next_row
            state["completed"] = next_row >= total_rows
            state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self._save_json(state_path, state)

            elapsed = max(time.time() - started, 1e-9)
            speed = (next_row - resume_from) / elapsed
            self.logger.info(
                "Đã xử lý %s/%s (%.2f%%) | %.2f chunk/giây",
                f"{next_row:,}",
                f"{total_rows:,}",
                100.0 * next_row / total_rows,
                speed,
            )

            global_start = global_end

        state["completed"] = True
        state["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self._save_json(state_path, state)

    @staticmethod
    def _validate_result(embeddings: np.memmap) -> dict[str, Any]:
        row_count = int(embeddings.shape[0])
        sample_count = min(1000, row_count)
        indices = np.linspace(
            0,
            row_count - 1,
            num=sample_count,
            dtype=np.int64,
        )
        sample = np.asarray(embeddings[indices], dtype=np.float32)

        if not np.isfinite(sample).all():
            raise RuntimeError("Embedding chứa NaN hoặc Inf.")

        norms = np.linalg.norm(sample, axis=1)
        return {
            "sample_count": int(sample_count),
            "minimum_norm": float(norms.min()),
            "mean_norm": float(norms.mean()),
            "maximum_norm": float(norms.max()),
        }

    def build(self) -> tuple[Path, Path]:
        source = self.config.path("legal_chunks")
        output = self.config.path("embeddings")
        metadata_output = self.config.path("embedding_metadata")
        output_dir = output.parent
        config_output = output_dir / "embedding_config.json"
        state_output = output_dir / "embedding_state.json"

        available_columns, row_count = self._validate_input(source)

        text_mode = str(
            self.settings.get("text_mode", "content")
        ).casefold()
        if text_mode not in {"content", "contextual"}:
            raise ValueError(
                "embedding.text_mode chỉ nhận 'content' hoặc 'contextual'."
            )

        text_column = str(self.settings.get("text_column", "content"))
        text_columns = (
            [text_column]
            if text_mode == "content"
            else [
                column
                for column in TEXT_COLUMNS
                if column in available_columns
            ]
        )
        if text_column not in text_columns:
            text_columns.append(text_column)

        read_batch_size = int(
            self.settings.get("read_batch_size", 128)
        )
        if read_batch_size <= 0:
            raise ValueError("embedding.read_batch_size phải lớn hơn 0.")

        overwrite = bool(self.settings.get("overwrite", False))

        self.model.load()
        embedding_dim = self.model._target_dimension()
        model_source = self.model._resolve_model_source()

        self.logger.info("Input: %s", source.resolve())
        self.logger.info("Số chunk: %s", f"{row_count:,}")
        self.logger.info("Device: %s", self.model.device)
        self.logger.info(
            "Batch size model/read: %d/%d",
            int(self.settings.get("batch_size", 32)),
            read_batch_size,
        )
        self.logger.info("Các cột tạo text: %s", text_columns)

        output_dir.mkdir(parents=True, exist_ok=True)
        self._export_metadata(
            input_path=source,
            output_path=metadata_output,
            available_columns=available_columns,
            overwrite=overwrite,
        )

        state_signature = {
            "text_mode": text_mode,
            "text_columns": text_columns,
            "max_length": int(self.settings.get("max_length", 512)),
            "query_prompt_name": str(
                self.settings.get("query_prompt_name", "query")
            ),
        }

        embeddings, state = self._prepare_storage(
            embeddings_path=output,
            state_path=state_output,
            row_count=row_count,
            embedding_dim=embedding_dim,
            model_source=model_source,
            input_path=source,
            overwrite=overwrite,
            state_signature=state_signature,
        )

        embedding_config = {
            "input_path": str(source.resolve()),
            "model": model_source,
            "device": self.model.device,
            "batch_size": int(self.settings.get("batch_size", 32)),
            "read_batch_size": read_batch_size,
            "max_seq_length": int(self.settings.get("max_length", 512)),
            "full_model_dimension": self.model.full_dimension,
            "saved_embedding_dimension": embedding_dim,
            "text_mode": text_mode,
            "text_columns": text_columns,
            "normalize_embeddings": True,
            "row_count": row_count,
            "dtype": "float32",
            "document_prompt": None,
            "recommended_query_prompt_name": str(
                self.settings.get("query_prompt_name", "query")
            ),
        }
        self._save_json(config_output, embedding_config)

        self._embed_parquet(
            input_path=source,
            embeddings=embeddings,
            state=state,
            state_path=state_output,
            text_columns=text_columns,
            text_mode=text_mode,
            read_batch_size=read_batch_size,
        )

        validation = self._validate_result(embeddings)
        embedding_config["validation"] = validation
        self._save_json(config_output, embedding_config)

        self.logger.info(
            "Hoàn thành embedding | shape=(%d, %d) | output=%s",
            row_count,
            embedding_dim,
            output.resolve(),
        )
        return output, metadata_output
