from __future__ import annotations

import ast
import json
import logging
import math
from collections import defaultdict, deque
from typing import Iterable

import pandas as pd

from core.config import AppConfig
from retrieval.schemas import RetrievalResult


class GraphExpander:
    """Load a parquet graph and expand retrieval seeds through its edges.

    The loader supports both the cleaned graph schema and older graph files with
    endpoint columns such as ``source``/``target`` or ``src``/``dst``.
    """

    SOURCE_COLUMN_CANDIDATES = (
        "source_node_id",
        "source",
        "src",
        "from_node_id",
        "from_node",
        "from",
        "u",
    )
    TARGET_COLUMN_CANDIDATES = (
        "target_node_id",
        "target",
        "dst",
        "to_node_id",
        "to_node",
        "to",
        "v",
    )
    WEIGHT_COLUMN_CANDIDATES = (
        "reference_count",
        "weight",
        "edge_weight",
        "score",
    )

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.settings = config.section("graph")
        self.logger = logging.getLogger(self.__class__.__name__)
        self.nodes: pd.DataFrame | None = None
        self.edges: pd.DataFrame | None = None
        self.chunk_to_nodes: dict[str, list[str]] = defaultdict(list)
        self.node_to_chunks: dict[str, list[str]] = defaultdict(list)
        self.adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)

    def load(self) -> "GraphExpander":
        self.nodes = pd.read_parquet(self.config.path("graph_nodes"))
        self.edges = pd.read_parquet(self.config.path("graph_edges"))

        self._validate_nodes(self.nodes)
        self._reset_indexes()
        self._build_chunk_mapping(self.nodes)

        source_col = self._resolve_column(
            frame=self.edges,
            configured=self.settings.get("source_column"),
            candidates=self.SOURCE_COLUMN_CANDIDATES,
            role="source edge",
        )
        target_col = self._resolve_column(
            frame=self.edges,
            configured=self.settings.get("target_column"),
            candidates=self.TARGET_COLUMN_CANDIDATES,
            role="target edge",
        )
        weight_col = self._resolve_optional_column(
            frame=self.edges,
            configured=self.settings.get("edge_weight_key"),
            candidates=self.WEIGHT_COLUMN_CANDIDATES,
        )

        direction = str(self.settings.get("direction", "both")).lower().strip()
        if direction not in {"out", "in", "both"}:
            raise ValueError(
                "graph.direction must be one of: 'out', 'in', 'both'; "
                f"received {direction!r}."
            )

        skipped_edges = 0
        for _, edge in self.edges.iterrows():
            source = self._normalise_identifier(edge.get(source_col))
            target = self._normalise_identifier(edge.get(target_col))
            if not source or not target:
                skipped_edges += 1
                continue

            weight = self._parse_weight(edge.get(weight_col)) if weight_col else 1.0
            if direction in {"out", "both"}:
                self.adjacency[source].append((target, weight))
            if direction in {"in", "both"}:
                self.adjacency[target].append((source, weight))

        self.logger.info(
            "Loaded graph: nodes=%d, edges=%d, mapped_chunks=%d, "
            "source_column=%s, target_column=%s, weight_column=%s, skipped_edges=%d",
            len(self.nodes),
            len(self.edges),
            len(self.chunk_to_nodes),
            source_col,
            target_col,
            weight_col or "<default=1.0>",
            skipped_edges,
        )
        return self

    def _reset_indexes(self) -> None:
        self.chunk_to_nodes = defaultdict(list)
        self.node_to_chunks = defaultdict(list)
        self.adjacency = defaultdict(list)

    @staticmethod
    def _validate_nodes(nodes: pd.DataFrame) -> None:
        required = {"node_id", "source_chunk_ids"}
        missing = required - set(nodes.columns)
        if missing:
            raise ValueError(
                "nodes.parquet is missing required columns "
                f"{sorted(missing)}. Available columns: {list(nodes.columns)}"
            )

    def _build_chunk_mapping(self, nodes: pd.DataFrame) -> None:
        for _, row in nodes.iterrows():
            node_id = self._normalise_identifier(row.get("node_id"))
            if not node_id:
                continue
            for chunk_id in self.parse_chunk_ids(row.get("source_chunk_ids")):
                if node_id not in self.chunk_to_nodes[chunk_id]:
                    self.chunk_to_nodes[chunk_id].append(node_id)
                if chunk_id not in self.node_to_chunks[node_id]:
                    self.node_to_chunks[node_id].append(chunk_id)

    @classmethod
    def _resolve_column(
        cls,
        frame: pd.DataFrame,
        configured: object,
        candidates: Iterable[str],
        role: str,
    ) -> str:
        available = list(frame.columns)
        configured_name = str(configured).strip() if configured else ""

        if configured_name and configured_name in frame.columns:
            return configured_name

        for candidate in candidates:
            if candidate in frame.columns:
                if configured_name and configured_name != candidate:
                    logging.getLogger(cls.__name__).warning(
                        "Configured %s column %r was not found; using detected column %r.",
                        role,
                        configured_name,
                        candidate,
                    )
                return candidate

        raise ValueError(
            f"Cannot detect the {role} column in graph edges. "
            f"Configured value: {configured_name or '<not set>'}. "
            f"Available columns: {available}. "
            f"Supported aliases: {list(candidates)}."
        )

    @classmethod
    def _resolve_optional_column(
        cls,
        frame: pd.DataFrame,
        configured: object,
        candidates: Iterable[str],
    ) -> str | None:
        configured_name = str(configured).strip() if configured else ""
        if configured_name and configured_name in frame.columns:
            return configured_name
        for candidate in candidates:
            if candidate in frame.columns:
                return candidate
        return None

    @staticmethod
    def _normalise_identifier(value: object) -> str:
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
        return str(value).strip()

    @staticmethod
    def _parse_weight(value: object) -> float:
        try:
            weight = float(value)
        except (TypeError, ValueError):
            return 1.0
        if not math.isfinite(weight):
            return 1.0
        return weight

    @staticmethod
    def parse_chunk_ids(value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        try:
            if pd.isna(value):
                return []
        except (TypeError, ValueError):
            pass

        text = str(value).strip()
        if not text:
            return []

        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                if isinstance(parsed, (list, tuple, set)):
                    return [str(item).strip() for item in parsed if str(item).strip()]
                return [str(parsed).strip()] if parsed is not None else []
            except (ValueError, TypeError, SyntaxError, json.JSONDecodeError):
                continue

        return [
            item.strip().strip("'\"")
            for item in text.strip("[]").split(",")
            if item.strip().strip("'\"")
        ]

    def expand(self, seeds: list[RetrievalResult]) -> dict[str, float]:
        if self.nodes is None or self.edges is None:
            self.load()

        scores: dict[str, float] = defaultdict(float)
        max_hops = int(self.settings.get("max_hops", 2))
        decay = float(self.settings.get("hop_decay", 0.65))

        for seed in seeds:
            seed_id = self._normalise_identifier(seed.chunk_id)
            for node_id in self.chunk_to_nodes.get(seed_id, []):
                queue = deque([(node_id, 0, float(seed.score))])
                visited = {node_id}

                while queue:
                    current, hop, score = queue.popleft()
                    for chunk_id in self.node_to_chunks.get(current, []):
                        scores[chunk_id] = max(scores[chunk_id], score)

                    if hop >= max_hops:
                        continue

                    for neighbor, weight in self.adjacency.get(current, []):
                        if neighbor in visited:
                            continue
                        visited.add(neighbor)
                        queue.append((neighbor, hop + 1, score * decay * weight))

        return dict(scores)
