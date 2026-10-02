from __future__ import annotations

from collections import defaultdict

from .schemas import RetrievalResult


class ReciprocalRankFusion:
    def __init__(self, k: int = 60) -> None:
        self.k = k

    def fuse(self, ranked_lists: list[tuple[list[RetrievalResult], float]]) -> list[RetrievalResult]:
        scores: dict[str, float] = defaultdict(float)
        best: dict[str, RetrievalResult] = {}
        sources: dict[str, set[str]] = defaultdict(set)
        for results, weight in ranked_lists:
            for rank, result in enumerate(results, start=1):
                scores[result.chunk_id] += weight / (self.k + rank)
                best.setdefault(result.chunk_id, result)
                sources[result.chunk_id].add(result.source)
        fused = []
        for chunk_id, score in scores.items():
            item = best[chunk_id]
            fused.append(RetrievalResult(chunk_id, score, item.text, "+".join(sorted(sources[chunk_id])), item.metadata))
        return sorted(fused, key=lambda item: item.score, reverse=True)
