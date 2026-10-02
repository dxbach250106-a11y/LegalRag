from __future__ import annotations

import csv
import itertools
import json
import logging
import time
from pathlib import Path
from typing import Any

from core.config import AppConfig
from graph_rerank.graph_search_pipeline import GraphSearchPipeline
from retrieval.schemas import RetrievalResult
from .dataset import EvaluationDataset, EvaluationQuery
from .metrics import RetrievalMetrics


class EvaluationRunner:
    """Tune retrieval hyperparameters and compare five retrieval methods on 300 queries."""

    METHODS = ("bm25", "vector", "graph", "hybrid", "hybrid+graph")

    def __init__(self, config: AppConfig, pipeline: GraphSearchPipeline | None = None) -> None:
        self.config = config
        self.pipeline = pipeline or GraphSearchPipeline(config)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.ks = [int(k) for k in self.config.get("evaluation.ks", [5, 10, 50, 100, 200])]
        self.query_count = int(self.config.get("evaluation.query_count", 4000))
        self.max_k = max(self.ks)
        self.candidate_k = max(
            self.max_k,
            int(self.config.get("evaluation.candidate_k", self.max_k)),
        )
        self.output_dir = self.config.path("evaluation_dir")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._cache: list[dict[str, Any]] = []

    def run(self) -> dict[str, Any]:
        return self.run_experiment()

    def run_experiment(self) -> dict[str, Any]:
        queries = self._load_queries()
        self.pipeline.load()
        self._build_component_cache(queries)

        best_hybrid = self._tune_hybrid()
        best_graph = self._tune_graph(best_hybrid)
        best_hybrid_graph = self._tune_hybrid_graph(best_hybrid)

        method_params: dict[str, dict[str, Any]] = {
            "bm25": {},
            "vector": {},
            "hybrid": best_hybrid,
            "graph": {**best_hybrid, **best_graph},
            "hybrid+graph": {**best_hybrid, **best_hybrid_graph},
        }

        method_runs: list[dict[str, Any]] = []
        for method in self.METHODS:
            self.logger.info("[FINAL] Evaluating %s on %d queries", method, len(queries))
            method_runs.append(self._evaluate_method(method, method_params[method], keep_details=True))

        summary_rows = [run["summary"] for run in method_runs]
        report = {
            "query_count": len(queries),
            "ks": self.ks,
            "miss_definition": f"number of queries with no relevant chunk in top-{self.max_k}",
            "best_hyperparameters": {
                "hybrid": best_hybrid,
                "graph": best_graph,
                "hybrid+graph": best_hybrid_graph,
            },
            "results": summary_rows,
        }
        self._save_outputs(report, method_runs)
        self.print_summary_table(summary_rows)
        return report

    def _load_queries(self) -> list[EvaluationQuery]:
        all_queries = EvaluationDataset.load(self.config.path("queries"))
        seed = int(self.config.get("evaluation.random_seed", self.config.get("project.seed", 42)))
        shuffle = bool(self.config.get("evaluation.shuffle", True))
        selected, _ = EvaluationDataset.split(
            all_queries,
            train_size=self.query_count,
            test_size=0,
            seed=seed,
            shuffle=shuffle,
        )
        self.logger.info("Selected %d evaluation queries from %d available queries", len(selected), len(all_queries))
        return selected

    def _build_component_cache(self, queries: list[EvaluationQuery]) -> None:
        """Encode each query once and cache BM25/vector rankings for all tuning trials."""
        self._cache = []
        log_every = int(self.config.get("evaluation.log_every", 25))
        started = time.perf_counter()
        for index, item in enumerate(queries, start=1):
            vector, bm25 = self.pipeline.retriever.search_components(
                item.query,
                vector_k=self.candidate_k,
                bm25_k=self.candidate_k,
            )
            self._cache.append(
                {
                    "item": item,
                    "vector": vector,
                    "bm25": bm25,
                }
            )
            if index == 1 or index % log_every == 0 or index == len(queries):
                self.logger.info("[CACHE] %d/%d queries", index, len(queries))
        self.logger.info("Component cache completed in %.2f seconds", time.perf_counter() - started)

    def _tune_hybrid(self) -> dict[str, Any]:
        grid = self.config.get("evaluation.hyperparameter_grid.hybrid", {})
        semantic_weights = grid.get("semantic_weight", [0.5, 1.0, 1.5])
        lexical_weights = grid.get("lexical_weight", [0.5, 1.0, 1.5])
        rrf_values = grid.get("rrf_k", [30, 60, 90])

        trials = [
            {
                "semantic_weight": float(sw),
                "lexical_weight": float(lw),
                "rrf_k": int(rrf_k),
            }
            for sw, lw, rrf_k in itertools.product(semantic_weights, lexical_weights, rrf_values)
        ]
        return self._select_best("hybrid", trials)

    def _tune_graph(self, hybrid_params: dict[str, Any]) -> dict[str, Any]:
        grid = self.config.get("evaluation.hyperparameter_grid.graph", {})
        trials = [
            {
                **hybrid_params,
                "max_hops": int(max_hops),
                "hop_decay": float(hop_decay),
            }
            for max_hops, hop_decay in itertools.product(
                grid.get("max_hops", [1, 2]),
                grid.get("hop_decay", [0.5, 0.65, 0.8]),
            )
        ]
        best = self._select_best("graph", trials)
        return {key: best[key] for key in ("max_hops", "hop_decay")}

    def _tune_hybrid_graph(self, hybrid_params: dict[str, Any]) -> dict[str, Any]:
        grid = self.config.get("evaluation.hyperparameter_grid.hybrid_graph", {})
        trials = []
        for graph_weight, max_hops, hop_decay in itertools.product(
            grid.get("graph_weight", [0.1, 0.2, 0.3]),
            grid.get("max_hops", [1, 2]),
            grid.get("hop_decay", [0.5, 0.65, 0.8]),
        ):
            graph_weight = float(graph_weight)
            trials.append(
                {
                    **hybrid_params,
                    "retrieval_weight": 1.0 - graph_weight,
                    "graph_weight": graph_weight,
                    "max_hops": int(max_hops),
                    "hop_decay": float(hop_decay),
                }
            )
        best = self._select_best("hybrid+graph", trials)
        return {
            key: best[key]
            for key in ("retrieval_weight", "graph_weight", "max_hops", "hop_decay")
        }

    def _select_best(self, method: str, trials: list[dict[str, Any]]) -> dict[str, Any]:
        best_params: dict[str, Any] | None = None
        best_score = float("-inf")
        self.logger.info("Tuning %s with %d hyperparameter combinations", method, len(trials))
        for index, params in enumerate(trials, start=1):
            run = self._evaluate_method(method, params, keep_details=False)
            score = float(run["objective"])
            self.logger.info("[TUNE %s %d/%d] objective=%.6f params=%s", method, index, len(trials), score, params)
            if score > best_score:
                best_score = score
                best_params = dict(params)
        if best_params is None:
            raise RuntimeError(f"No valid hyperparameter trial for method {method}")
        self.logger.info("Best %s parameters: objective=%.6f %s", method, best_score, best_params)
        return best_params

    def _rank_cached(self, cached: dict[str, Any], method: str, params: dict[str, Any]) -> tuple[list[RetrievalResult], dict[str, Any]]:
        vector: list[RetrievalResult] = cached["vector"]
        bm25: list[RetrievalResult] = cached["bm25"]
        empty_debug = {"mapped_zero": False, "graph_result_zero": False}

        if method == "vector":
            return vector[: self.max_k], empty_debug
        if method == "bm25":
            return bm25[: self.max_k], empty_debug

        hybrid = self.pipeline.retriever.fuse(
            vector,
            bm25,
            semantic_weight=float(params.get("semantic_weight", 1.0)),
            lexical_weight=float(params.get("lexical_weight", 1.0)),
            rrf_k=int(params.get("rrf_k", 60)),
            top_k=self.candidate_k,
        )
        if method == "hybrid":
            return hybrid[: self.max_k], empty_debug
        if method == "graph":
            return self.pipeline._graph_only(hybrid, self.max_k, params)
        if method == "hybrid+graph":
            return self.pipeline._hybrid_graph(hybrid, self.max_k, params)
        raise ValueError(f"Unsupported method: {method}")

    def _evaluate_method(
        self,
        method: str,
        params: dict[str, Any],
        keep_details: bool,
    ) -> dict[str, Any]:
        sums = {
            f"{metric}@{k}": 0.0
            for k in self.ks
            for metric in ("hit", "recall", "mrr")
        }
        query_details: list[dict[str, Any]] = []
        miss = 0
        mapped_zero = 0
        graph_zero = 0
        started = time.perf_counter()

        for cached in self._cache:
            item: EvaluationQuery = cached["item"]
            results, debug = self._rank_cached(cached, method, params)
            ranked = [str(result.chunk_id).strip() for result in results]
            relevant = {str(chunk_id).strip() for chunk_id in item.relevant_chunk_ids}
            per_query: dict[str, float] = {}
            for k in self.ks:
                per_query[f"hit@{k}"] = RetrievalMetrics.hit_at_k(ranked, relevant, k)
                per_query[f"recall@{k}"] = RetrievalMetrics.recall_at_k(ranked, relevant, k)
                per_query[f"mrr@{k}"] = RetrievalMetrics.reciprocal_rank(ranked, relevant, k)
                for key in (f"hit@{k}", f"recall@{k}", f"mrr@{k}"):
                    sums[key] += per_query[key]

            is_miss = per_query[f"hit@{self.max_k}"] == 0.0
            miss += int(is_miss)
            mapped_zero += int(bool(debug.get("mapped_zero", False)))
            graph_zero += int(bool(debug.get("graph_result_zero", False)))

            if keep_details:
                query_details.append(
                    {
                        "method": method,
                        "query_id": item.query_id,
                        "query": item.query,
                        "relevant_chunk_ids": sorted(relevant),
                        "ranked_chunk_ids": ranked,
                        "miss": is_miss,
                        **per_query,
                    }
                )

        count = len(self._cache)
        averages = {key: value / count if count else 0.0 for key, value in sums.items()}
        objective = self._objective(averages)
        summary = {
            "method": method,
            "query": count,
            "miss": miss,
            **averages,
            "mapped_zero": mapped_zero,
            "graph_zero": graph_zero,
            "objective": objective,
            "elapsed_seconds": time.perf_counter() - started,
            "hyperparameters": params,
        }
        return {
            "method": method,
            "params": params,
            "objective": objective,
            "summary": summary,
            "query_results": query_details,
        }

    def _objective(self, averages: dict[str, float]) -> float:
        """Average Recall@K and MRR@K across every requested cutoff."""
        values = []
        for k in self.ks:
            values.append(float(averages[f"recall@{k}"]))
            values.append(float(averages[f"mrr@{k}"]))
        return sum(values) / len(values) if values else 0.0

    def _save_outputs(self, report: dict[str, Any], method_runs: list[dict[str, Any]]) -> None:
        (self.output_dir / "evaluation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        summary_path = self.output_dir / "method_comparison.csv"
        fieldnames = ["method", "query", "miss"]
        for k in self.ks:
            fieldnames.extend([f"hit@{k}", f"recall@{k}", f"mrr@{k}"])
        fieldnames.extend(["mapped_zero", "graph_zero", "objective", "elapsed_seconds", "hyperparameters"])
        with summary_path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for run in method_runs:
                row = dict(run["summary"])
                row["hyperparameters"] = json.dumps(row["hyperparameters"], ensure_ascii=False)
                writer.writerow({name: row.get(name) for name in fieldnames})

        detail_path = self.output_dir / "query_level_results.csv"
        detail_fields = ["method", "query_id", "query", "relevant_chunk_ids", "ranked_chunk_ids", "miss"]
        for k in self.ks:
            detail_fields.extend([f"hit@{k}", f"recall@{k}", f"mrr@{k}"])
        with detail_path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=detail_fields)
            writer.writeheader()
            for run in method_runs:
                for detail in run["query_results"]:
                    row = dict(detail)
                    row["relevant_chunk_ids"] = json.dumps(row["relevant_chunk_ids"], ensure_ascii=False)
                    row["ranked_chunk_ids"] = json.dumps(row["ranked_chunk_ids"], ensure_ascii=False)
                    writer.writerow({name: row.get(name) for name in detail_fields})

    def print_summary_table(self, rows: list[dict[str, Any]]) -> None:
        columns = ["method", "query", "miss"]
        for k in self.ks:
            columns.extend([f"hit@{k}", f"recall@{k}", f"mrr@{k}"])

        display_rows = []
        for row in rows:
            display = {}
            for column in columns:
                value = row.get(column, "")
                display[column] = f"{value:.4f}" if isinstance(value, float) else str(value)
            display_rows.append(display)

        widths = {
            column: max(len(column), *(len(row[column]) for row in display_rows))
            for column in columns
        }
        separator = "+-" + "-+-".join("-" * widths[column] for column in columns) + "-+"
        header = "| " + " | ".join(column.ljust(widths[column]) for column in columns) + " |"
        print("\nFINAL METHOD COMPARISON")
        print(separator)
        print(header)
        print(separator)
        for row in display_rows:
            print("| " + " | ".join(row[column].ljust(widths[column]) for column in columns) + " |")
        print(separator)
        print(f"miss = number of queries with hit@{self.max_k} = 0")
        print(f"Saved: {self.output_dir / 'method_comparison.csv'}")
        print(f"Saved: {self.output_dir / 'query_level_results.csv'}")
