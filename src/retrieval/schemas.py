from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalResult:
    chunk_id: str
    score: float
    text: str = ""
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
