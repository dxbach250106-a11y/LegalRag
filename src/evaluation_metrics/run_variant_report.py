from __future__ import annotations

import json
from pathlib import Path
from typing import Any


from evaluation_metrics.dataset import load_ground_truth
from evaluation_metrics.variant_report import VariantReporter
from graph_rerank import GraphExpander
from retrieval.hybrid_retriever import HybridRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.json"


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _first(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, list):
        return value[0] if value else default
    return value


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    paths = config["paths"]
    evaluation = config["evaluation"]
    graph_config = config.get("graph", {})

    retriever = HybridRetriever(
        project_root=PROJECT_ROOT,
        model_name_or_path=config["embedding_model"]["cache"],
        bm25_dir=paths["bm25_dir"],
        bm25_lookup_path=paths["bm25_lookup_path"],
        faiss_index_path=paths["faiss_index_path"],
        embedding_metadata_path=paths["embedding_metadata_path"],
        device=evaluation.get("device", "cpu"),
    ).load()

    initial_graph_weight = float(
        _first(graph_config.get("graph_weight"), 0.1)
    )

    graph_expander = GraphExpander(
        graph_path=_path(paths.get("graph_dir", "graph")),
        retrieval_weight=1.0 - initial_graph_weight,
        graph_weight=initial_graph_weight,
        hop_decay=float(_first(graph_config.get("hop_decay"), 0.5)),
        direction=graph_config.get("direction", "both"),
        allowed_edge_types=None,
        edge_type_key=graph_config.get("edge_type_key", "edge_type"),
        edge_weight_key=graph_config.get("edge_weight_key", "reference_count"),
    ).load()

    output_dir = _path(
        evaluation.get("output_dir", "evaluation_output")
    )

    best_path = output_dir / "best_hyperparameters.json"
    best_payload = json.loads(best_path.read_text(encoding="utf-8"))
    best_params = best_payload["best_params"]

    test_df = load_ground_truth(
        _path(evaluation["test_path"]),
        split="test",
    )

    reporter = VariantReporter(
        retriever,
        graph_expander,
        output_dir=output_dir,
        log_every=int(evaluation.get("log_every", 10)),
    )

    reporter.run(
        test_df,
        best_params=best_params,
    )


if __name__ == "__main__":
    main()
