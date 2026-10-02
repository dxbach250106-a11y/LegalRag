from __future__ import annotations

import shutil
import argparse
import logging
from pathlib import Path
from typing import Any

from core.config import AppConfig
from evaluation_metrics.runner import EvaluationRunner
from graph_rerank.graph_builder import LegalHierarchyGraphBuilder
from graph_rerank.graph_search_pipeline import GraphSearchPipeline
from models.answer_generator import LLMAnswerGenerator
from models.embedder import EmbeddingBuilder
from models.index_builders import BM25IndexBuilder, FaissIndexBuilder
from models.model_downloader import ModelDownloader
from preprocessing.chunker import chunk_dataset
from utils.logger import configure_logging



class LegalRAGApplication:
    """Build indexes, tune hyperparameters and compare retrieval methods."""
    SOURCE_MODES = ("existing", "rebuild")

    def __init__(self, config_path: str | Path = "config.json") -> None:
        self.config = AppConfig.load(config_path)
        configure_logging(self.config.get("project.log_level", "INFO"))
        self.logger = logging.getLogger(self.__class__.__name__)

    def download_models(self) -> None:
        downloader = ModelDownloader(self.config)
        downloader.download_embedding_model()
        if bool(self.config.get("generation.enabled", False)):
            downloader.download_generation_model()

    def load_raw_datasets(self) -> tuple[Any, Any]:
        """Đọc dataset content và metadata từ Hugging Face."""
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "Thiếu thư viện datasets. "
                "Cài bằng lệnh: pip install datasets"
            ) from exc

        dataset_name = str(
            self.config.get(
                "dataset.name",
                "minhdoan17/vietnamese-legal-documents",
            )
        )

        content_config = str(
            self.config.get(
                "dataset.content_config",
                "content",
            )
        )

        metadata_config = str(
            self.config.get(
                "dataset.metadata_config",
                "metadata",
            )
        )

        cache_dir = self.config.get("dataset.cache_dir")

        load_kwargs: dict[str, Any] = {}

        if cache_dir:
            load_kwargs["cache_dir"] = str(cache_dir)

        self.logger.info(
            "Loading raw dataset: "
            "name=%s, content=%s, metadata=%s",
            dataset_name,
            content_config,
            metadata_config,
        )

        ds_content = load_dataset(
            dataset_name,
            content_config,
            **load_kwargs,
        )

        ds_metadata = load_dataset(
            dataset_name,
            metadata_config,
            **load_kwargs,
        )

        return ds_content, ds_metadata

    def chunk_raw_datasets(
        self,
        ds_content: Any,
        ds_metadata: Any,
        legal_sector: str | None = None,
    ) -> Path:
        """Chunk raw dataset và lưu thành legal_chunks.parquet."""
        target_sector = (
            legal_sector
            or self.config.get(
                "chunking.target_legal_sector"
            )
        )

        if not target_sector:
            raise ValueError(
                "Chưa có lĩnh vực pháp luật. "
                "Hãy truyền --legal-sector hoặc đặt "
                "chunking.target_legal_sector trong config.json."
            )

        output_path = self.config.path("legal_chunks")
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = output_path.with_name(
            f"{output_path.stem}.tmp{output_path.suffix}"
        )

        temporary_path.unlink(missing_ok=True)

        self.logger.info(
            "Chunking dataset: sector=%s, output=%s",
            target_sector,
            output_path.resolve(),
        )

        try:
            chunk_dataset(
                ds_content=ds_content,
                ds_metadata=ds_metadata,
                target_legal_sector=str(target_sector),
                output_path=str(temporary_path),
                min_chunk_chars=int(
                    self.config.get(
                        "chunking.min_chunk_chars",
                        50,
                    )
                ),
                print_all_chunks=bool(
                    self.config.get(
                        "chunking.print_all_chunks",
                        False,
                    )
                ),
            )

            if not temporary_path.exists():
                raise RuntimeError(
                    "Chunker không tạo file output: "
                    f"{temporary_path}"
                )

            if temporary_path.stat().st_size == 0:
                raise RuntimeError(
                    "Chunker tạo file output rỗng: "
                    f"{temporary_path}"
                )

            temporary_path.replace(output_path)

        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

        self.logger.info(
            "Chunk file created: %s",
            output_path.resolve(),
        )

        return output_path

    def rebuild_chunks_from_raw(
        self,
        legal_sector: str | None = None,
    ) -> Path:
        """Đọc raw dataset rồi chunk lại từ đầu."""
        ds_content, ds_metadata = (
            self.load_raw_datasets()
        )

        return self.chunk_raw_datasets(
            ds_content=ds_content,
            ds_metadata=ds_metadata,
            legal_sector=legal_sector,
        )

    def ensure_existing_chunks(self) -> Path:
        """Kiểm tra file legal_chunks.parquet có sẵn."""
        chunks_path = self.config.path(
            "legal_chunks"
        )

        if not chunks_path.exists():
            raise FileNotFoundError(
                "Không tìm thấy file chunks có sẵn:\n"
                f"{chunks_path.resolve()}\n"
                "Hãy chạy với --source-mode rebuild."
            )

        if chunks_path.stat().st_size == 0:
            raise ValueError(
                "File chunks đang rỗng: "
                f"{chunks_path.resolve()}"
            )

        self.logger.info(
            "Using existing chunks: %s",
            chunks_path.resolve(),
        )

        return chunks_path

    @staticmethod
    def remove_artifact(path: Path) -> None:
        """Xóa file hoặc thư mục artifact."""
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()

    def clear_derived_artifacts(self) -> None:
        """
        Xóa embedding, FAISS, BM25 và graph cũ.

        Không xóa legal_chunks.parquet.
        """
        artifact_paths = [
            self.config.path("embedding_dir"),
            self.config.path("faiss_index").parent,
            self.config.path("bm25_dir"),
            self.config.path("graph_dir"),
        ]

        processed: set[Path] = set()

        for path in artifact_paths:
            resolved_path = path.resolve()

            if resolved_path in processed:
                continue

            processed.add(resolved_path)

            if path.exists():
                self.logger.info(
                    "Removing old artifact: %s",
                    resolved_path,
                )

                self.remove_artifact(path)

    def prepare_source(
        self,
        source_mode: str,
        legal_sector: str | None = None,
        force: bool = False,
    ) -> Path:
        """
        existing:
            Dùng legal_chunks.parquet có sẵn.

        rebuild:
            Đọc raw dataset và chunk lại từ đầu.
        """
        if source_mode not in self.SOURCE_MODES:
            raise ValueError(
                f"Unknown source mode: {source_mode}. "
                f"Supported: {list(self.SOURCE_MODES)}"
            )

        if source_mode == "rebuild":
            chunks_path = (
                self.rebuild_chunks_from_raw(
                    legal_sector=legal_sector
                )
            )

            self.clear_derived_artifacts()

            return chunks_path

        chunks_path = self.ensure_existing_chunks()

        if force:
            self.clear_derived_artifacts()

        return chunks_path

    def build(
        self,
        source_mode: str = "existing",
        legal_sector: str | None = None,
        force: bool = False,
    ) -> None:
        """
        Build từ chunks có sẵn hoặc chạy lại từ raw dataset.

        existing:
            legal_chunks.parquet
            -> embedding
            -> FAISS
            -> BM25
            -> graph

        rebuild:
            raw dataset
            -> chunk
            -> embedding
            -> FAISS
            -> BM25
            -> graph
        """
        self.prepare_source(
            source_mode=source_mode,
            legal_sector=legal_sector,
            force=force,
        )

        actions = {
            "download_models": self.download_models,
            "embed": lambda: EmbeddingBuilder(
                self.config
            ).build(),
            "faiss": lambda: FaissIndexBuilder(
                self.config
            ).build(),
            "bm25": lambda: BM25IndexBuilder(
                self.config
            ).build(),
            "graph": lambda: LegalHierarchyGraphBuilder(
                self.config
            ).build(),
        }

        steps = list(
            self.config.get(
                "pipeline.build_steps",
                actions.keys(),
            )
        )

        for index, step in enumerate(
            steps,
            start=1,
        ):
            if step not in actions:
                raise ValueError(
                    f"Unknown build step: {step}. "
                    f"Supported: {list(actions)}"
                )

            self.logger.info(
                "[BUILD %d/%d] %s",
                index,
                len(steps),
                step,
            )

            actions[step]()

    def train_tune_evaluate(
        self,
        source_mode: str = "existing",
        legal_sector: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """
        Mặc định dùng artifacts có sẵn.

        Chỉ build lại khi:
        - source_mode là rebuild;
        - sử dụng --force;
        - rebuild_before_evaluation=true.
        """
        should_rebuild = (
            source_mode == "rebuild"
            or force
            or bool(
                self.config.get(
                    "pipeline.rebuild_before_evaluation",
                    False,
                )
            )
        )

        if should_rebuild:
            self.build(
                source_mode=source_mode,
                legal_sector=legal_sector,
                force=force,
            )

        return EvaluationRunner(
            self.config
        ).run_experiment()

    def experiment(self) -> dict[str, Any]:
        return self.train_tune_evaluate()

    def evaluate(self) -> dict[str, Any]:
        return self.train_tune_evaluate()

    def search(self, query: str | None = None, method: str = "hybrid+graph") -> list[dict[str, Any]]:
        query = query or self.config.get("pipeline.query")
        if not query:
            raise ValueError("A query is required. Pass --query or set pipeline.query in config.json.")
        top_k = int(self.config.get("retrieval.final_top_k", 200))
        results = GraphSearchPipeline(self.config).load().search_method(query, method=method, top_k=top_k)
        output = [
            {"rank": rank, "chunk_id": item.chunk_id, "score": item.score, "source": item.source, "text": item.text}
            for rank, item in enumerate(results, start=1)
        ]
        for item in output[:20]:
            print(item)
        return output

    def answer(self, query: str | None = None) -> str:
        query = query or self.config.get("pipeline.query")
        if not query:
            raise ValueError("A query is required. Pass --query or set pipeline.query in config.json.")
        results = GraphSearchPipeline(self.config).load().search(query)
        answer = LLMAnswerGenerator(self.config).generate(query, results)
        print(answer)
        return answer

    def run(
        self,
        mode: str | None = None,
        query: str | None = None,
        method: str = "hybrid+graph",
        source_mode: str | None = None,
        legal_sector: str | None = None,
        force: bool = False,
    ) -> Any:
        selected_mode = (
            mode
            or self.config.get(
                "pipeline.mode",
                "experiment",
            )
        )

        selected_source_mode = (
            source_mode
            or self.config.get(
                "pipeline.source_mode",
                "existing",
            )
        )

        if (
            selected_source_mode
            not in self.SOURCE_MODES
        ):
            raise ValueError(
                "Unknown source mode: "
                f"{selected_source_mode}. "
                f"Supported: {list(self.SOURCE_MODES)}"
            )

        if (
            selected_mode in {"search", "answer"}
            and (
                selected_source_mode == "rebuild"
                or force
            )
        ):
            self.build(
                source_mode=selected_source_mode,
                legal_sector=legal_sector,
                force=force,
            )

        actions = {
            "experiment": lambda: (
                self.train_tune_evaluate(
                    source_mode=selected_source_mode,
                    legal_sector=legal_sector,
                    force=force,
                )
            ),
            "evaluate": lambda: (
                self.train_tune_evaluate(
                    source_mode=selected_source_mode,
                    legal_sector=legal_sector,
                    force=force,
                )
            ),
            "build": lambda: self.build(
                source_mode=selected_source_mode,
                legal_sector=legal_sector,
                force=force,
            ),
            "search": lambda: self.search(
                query,
                method,
            ),
            "answer": lambda: self.answer(
                query
            ),
        }

        if selected_mode not in actions:
            raise ValueError(
                f"Unknown mode: {selected_mode}. "
                f"Supported: {list(actions)}"
            )

        return actions[selected_mode]()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "LegalRAG training, hyperparameter "
            "tuning and evaluation"
        )
    )

    parser.add_argument(
        "--config",
        default="config.json",
    )

    parser.add_argument(
        "--mode",
        choices=[
            "experiment",
            "evaluate",
            "build",
            "search",
            "answer",
        ],
        default=None,
    )

    parser.add_argument(
        "--source-mode",
        choices=list(
            LegalRAGApplication.SOURCE_MODES
        ),
        default=None,
        help=(
            "existing: dùng file có sẵn; "
            "rebuild: chạy lại từ raw dataset."
        ),
    )

    parser.add_argument(
        "--legal-sector",
        default=None,
        help=(
            "Lĩnh vực dùng khi chunk dataset, "
            "ví dụ: Giáo dục."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Xóa embedding, indexes và graph "
            "cũ trước khi build lại."
        ),
    )

    parser.add_argument(
        "--query",
        default=None,
    )

    parser.add_argument(
        "--method",
        choices=list(
            GraphSearchPipeline.SUPPORTED_METHODS
        ),
        default="hybrid+graph",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    LegalRAGApplication(
        args.config
    ).run(
        mode=args.mode,
        query=args.query,
        method=args.method,
        source_mode=args.source_mode,
        legal_sector=args.legal_sector,
        force=args.force,
    )


if __name__ == "__main__":
    main()
