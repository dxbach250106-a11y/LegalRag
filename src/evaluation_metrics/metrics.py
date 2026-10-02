from __future__ import annotations


class RetrievalMetrics:
    @staticmethod
    def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
        return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0

    @staticmethod
    def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
        return len(set(ranked[:k]) & relevant) / k if k > 0 else 0.0

    @staticmethod
    def hit_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
        return float(bool(set(ranked[:k]) & relevant))

    @staticmethod
    def reciprocal_rank(ranked: list[str], relevant: set[str], k: int) -> float:
        for rank, chunk_id in enumerate(ranked[:k], start=1):
            if chunk_id in relevant:
                return 1.0 / rank
        return 0.0
