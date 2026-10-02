from __future__ import annotations

from dataclasses import replace

from core.config import AppConfig
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.schemas import RetrievalResult
from .graph_expander import GraphExpander


class GraphSearchPipeline:
    """Expose BM25, vector, graph, hybrid and hybrid+graph rankings."""

    SUPPORTED_METHODS = ("bm25", "vector", "graph", "hybrid", "hybrid+graph")

    def __init__(
        self,
        config: AppConfig,
        retriever: HybridRetriever | None = None,
        expander: GraphExpander | None = None,
    ) -> None:
        self.config = config
        self.retriever = retriever or HybridRetriever(config)
        self.expander = expander or GraphExpander(config)
        self.loaded = False

    def load(self) -> "GraphSearchPipeline":
        if not self.loaded:
            self.retriever.load()
            self.expander.load()
            self.loaded = True
        return self

    def search_method(
        self,
        query: str,
        method: str,
        top_k: int,
        params: dict[str, float | int] | None = None,
        return_debug: bool = False,
    ) -> list[RetrievalResult] | tuple[list[RetrievalResult], dict[str, object]]:
        self.load()
        params = params or {}
        method = method.lower().strip()
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Unknown retrieval method {method!r}. Supported: {self.SUPPORTED_METHODS}")

        candidate_k = max(
            int(top_k),
            int(params.get("candidate_k", self.config.get("evaluation.candidate_k", top_k))),
        )
        vector, bm25 = self.retriever.search_components(query, candidate_k, candidate_k)

        if method == "vector":
            results = vector[:top_k]
            debug = self._empty_debug(method, vector)
        elif method == "bm25":
            results = bm25[:top_k]
            debug = self._empty_debug(method, bm25)
        else:
            hybrid = self.retriever.fuse(
                vector,
                bm25,
                semantic_weight=float(params.get("semantic_weight", 1.0)),
                lexical_weight=float(params.get("lexical_weight", 1.0)),
                rrf_k=int(params.get("rrf_k", 60)),
                top_k=candidate_k,
            )
            if method == "hybrid":
                results = hybrid[:top_k]
                debug = self._empty_debug(method, hybrid)
            elif method == "graph":
                results, debug = self._graph_only(hybrid, top_k, params)
            else:
                results, debug = self._hybrid_graph(hybrid, top_k, params)

        return (results, debug) if return_debug else results

    def search(
        self,
        query: str,
        top_k: int | None = None,
        retrieval_weight: float | None = None,
        graph_weight: float | None = None,
        max_hops: int | None = None,
        hop_decay: float | None = None,
        return_debug: bool = False,
    ) -> list[RetrievalResult] | tuple[list[RetrievalResult], dict[str, object]]:
        """Backward-compatible default: hybrid+graph."""
        params = {
            "semantic_weight": self.config.get("retrieval.semantic_weight", 1.0),
            "lexical_weight": self.config.get("retrieval.lexical_weight", 1.0),
            "rrf_k": self.config.get("retrieval.rrf_k", 60),
            "retrieval_weight": self.config.get("graph.retrieval_weight", 0.8) if retrieval_weight is None else retrieval_weight,
            "graph_weight": self.config.get("graph.graph_weight", 0.2) if graph_weight is None else graph_weight,
            "max_hops": self.config.get("graph.max_hops", 2) if max_hops is None else max_hops,
            "hop_decay": self.config.get("graph.hop_decay", 0.65) if hop_decay is None else hop_decay,
        }
        return self.search_method(
            query,
            method="hybrid+graph",
            top_k=int(top_k or self.config.get("retrieval.final_top_k", 200)),
            params=params,
            return_debug=return_debug,
        )

    def _graph_only(
        self,
        seeds: list[RetrievalResult],
        top_k: int,
        params: dict[str, float | int],
    ) -> tuple[list[RetrievalResult], dict[str, object]]:
        graph_scores, debug = self._expand(seeds, params)
        ranked = [
            RetrievalResult(chunk_id=chunk_id, score=float(score), source="graph")
            for chunk_id, score in graph_scores.items()
        ]
        ranked.sort(key=lambda item: item.score, reverse=True)
        debug["method"] = "graph"
        return ranked[:top_k], debug

    def _hybrid_graph(
        self,
        hybrid: list[RetrievalResult],
        top_k: int,
        params: dict[str, float | int],
    ) -> tuple[list[RetrievalResult], dict[str, object]]:
        graph_scores, debug = self._expand(hybrid, params)
        retrieval_weight = float(params.get("retrieval_weight", 0.8))
        graph_weight = float(params.get("graph_weight", 0.2))

        combined: dict[str, RetrievalResult] = {
            item.chunk_id: replace(item, score=retrieval_weight * float(item.score))
            for item in hybrid
        }
        for chunk_id, graph_score in graph_scores.items():
            if chunk_id in combined:
                combined[chunk_id].score += graph_weight * float(graph_score)
                combined[chunk_id].source = self._merge_source(combined[chunk_id].source, "graph")
            else:
                combined[chunk_id] = RetrievalResult(
                    chunk_id=chunk_id,
                    score=graph_weight * float(graph_score),
                    source="graph",
                )
        ranked = sorted(combined.values(), key=lambda item: float(item.score), reverse=True)
        debug.update(
            {
                "method": "hybrid+graph",
                "retrieval_weight": retrieval_weight,
                "graph_weight": graph_weight,
            }
        )
        return ranked[:top_k], debug

    def _expand(
        self,
        seeds: list[RetrievalResult],
        params: dict[str, float | int],
    ) -> tuple[dict[str, float], dict[str, object]]:
        retrieval_ids = [str(item.chunk_id).strip() for item in seeds]
        mapped_seed_ids = [chunk_id for chunk_id in retrieval_ids if chunk_id in self.expander.chunk_to_nodes]
        old_max_hops = self.expander.settings.get("max_hops", 2)
        old_hop_decay = self.expander.settings.get("hop_decay", 0.65)
        self.expander.settings["max_hops"] = int(params.get("max_hops", old_max_hops))
        self.expander.settings["hop_decay"] = float(params.get("hop_decay", old_hop_decay))
        try:
            graph_scores = self.expander.expand(seeds)
        finally:
            self.expander.settings["max_hops"] = old_max_hops
            self.expander.settings["hop_decay"] = old_hop_decay
        return graph_scores, {
            "retrieval_seed_ids": retrieval_ids,
            "mapped_seed_ids": mapped_seed_ids,
            "graph_result_ids": list(graph_scores),
            "mapped_zero": len(mapped_seed_ids) == 0,
            "graph_result_zero": len(graph_scores) == 0,
            "max_hops": int(params.get("max_hops", old_max_hops)),
            "hop_decay": float(params.get("hop_decay", old_hop_decay)),
        }

    @staticmethod
    def _empty_debug(method: str, results: list[RetrievalResult]) -> dict[str, object]:
        return {
            "method": method,
            "retrieval_seed_ids": [str(item.chunk_id) for item in results],
            "mapped_seed_ids": [],
            "graph_result_ids": [],
            "mapped_zero": False,
            "graph_result_zero": False,
        }

    @staticmethod
    def _merge_source(left: str, right: str) -> str:
        values = [value for value in (left, right) if value]
        return "+".join(dict.fromkeys(values))
