from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class GraphSearchResult:
    chunk_id: str
    final_score: float
    retrieval_score: float
    graph_score: float
    hop: int | None
    source_seed_ids: list[str] = field(default_factory=list)
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "final_score": self.final_score,
            "retrieval_score": self.retrieval_score,
            "graph_score": self.graph_score,
            "hop": self.hop,
            "source_seed_ids": self.source_seed_ids,
            "text": self.text,
            "metadata": self.metadata,
        }
