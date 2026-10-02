"""Class-based legal hierarchy graph builder.

The public interface is unchanged::

    LegalHierarchyGraphBuilder(config).build()

Internally this preserves the original graph-building logic:

    DOCUMENT -> PART -> CHAPTER -> SECTION -> ARTICLE

It also creates DOCUMENT_LINK edges directly from the ``link_to`` column.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from core.config import AppConfig


REQUIRED_COLUMNS = {
    "id",
    "document_id",
    "document_number",
    "title",
    "part",
    "chapter",
    "section",
    "articles",
    "content",
    "link_to",
}

OPTIONAL_DOCUMENT_COLUMNS = [
    "url",
    "legal_type",
    "legal_sectors",
    "issuing_authority",
    "issuance_date",
    "signers",
]

VIETNAMESE_LEVEL_WORDS = (
    "một|mot|hai|ba|bốn|bon|năm|nam|sáu|sau|bảy|bay|"
    "tám|tam|chín|chin|mười|muoi"
)

PART_RE = re.compile(
    rf"^\s*Phần\s+(?:(?:thứ\s+)?(?:{VIETNAMESE_LEVEL_WORDS}|[IVXLCDM]+|\d+|[A-ZĐ]))"
    r"(?:\s*[:.\-(]|\s+|$)",
    flags=re.IGNORECASE,
)
CHAPTER_RE = re.compile(
    r"^\s*Chương\s+(?:[IVXLCDM]+|\d+|[A-ZĐ])(?:\s*[:.\-]|\s+|$)",
    flags=re.IGNORECASE,
)
SECTION_RE = re.compile(
    r"^\s*Mục\s+(?:[IVXLCDM]+|\d+|[A-ZĐ])(?:\s*[:.\-]|\s+|$)",
    flags=re.IGNORECASE,
)
ARTICLE_RE = re.compile(
    r"^\s*Điều\s+(\d+[A-Za-zĐđ]?)\b",
    flags=re.IGNORECASE,
)

VIRTUAL_PART = "__NO_PART__"
VIRTUAL_CHAPTER = "__NO_CHAPTER__"
VIRTUAL_SECTION = "__NO_SECTION__"
VIRTUAL_LABELS = {
    VIRTUAL_PART: "Không có Phần",
    VIRTUAL_CHAPTER: "Không có Chương",
    VIRTUAL_SECTION: "Không có Mục",
}


def is_missing(value: Any) -> bool:
    """Return True for scalar null/blank values without failing on arrays."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        result = pd.isna(value)
        return bool(result) if isinstance(result, bool) else False
    except (TypeError, ValueError):
        return False


def clean_text(value: Any) -> str:
    if is_missing(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_document_number(value: Any) -> str:
    """Normalize legal document numbers for link resolution."""
    return re.sub(r"\s+", "", clean_text(value).upper())


def normalize_relation(value: Any) -> str:
    relation = clean_text(value).upper()
    relation = re.sub(r"[^A-Z0-9_]+", "_", relation).strip("_")
    return relation or "REFERS_TO"


def stable_id(prefix: str, *values: Any) -> str:
    """Create an ID that remains stable across graph rebuilds."""
    raw = "\x1f".join(clean_text(value) for value in values)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}:{digest}"


def first_non_empty(series: pd.Series) -> Any:
    for value in series:
        if not is_missing(value):
            return value
    return None


def valid_heading(value: Any, pattern: re.Pattern[str]) -> str:
    text = clean_text(value)
    return text if text and pattern.match(text) else ""


def list_to_json(values: Iterable[Any]) -> str:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = clean_text(value)
        if text and text not in seen:
            seen.add(text)
            cleaned.append(text)
    return json.dumps(cleaned, ensure_ascii=False)


def iter_link_items(value: Any) -> Iterable[dict[str, Any]]:
    """Safely iterate list/array/dict values read from a Parquet column."""
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if hasattr(value, "tolist") and not isinstance(value, str):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, dict)]
    return []


class LegalHierarchyGraphBuilder:
    """Build the legal hierarchy graph and document reference graph.

    Public interface intentionally remains compatible with the current
    application pipeline:

        builder = LegalHierarchyGraphBuilder(config)
        nodes_path, edges_path = builder.build()
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.settings = config.section("graph")
        self.logger = logging.getLogger(self.__class__.__name__)

        self.input_path = self.config.path("legal_chunks")
        self.nodes_path = self.config.path("graph_nodes")
        self.edges_path = self.config.path("graph_edges")
        self.stats_path = self._resolve_stats_path()

        self.df: pd.DataFrame | None = None
        self.nodes: dict[str, dict[str, Any]] = {}
        self.hierarchy_edges: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.document_edge_accumulator: dict[
            tuple[str, str, str], dict[str, Any]
        ] = {}
        self.document_node_by_id: dict[int, str] = {}
        self.document_ids_by_number: dict[str, list[int]] = defaultdict(list)

    def _resolve_stats_path(self) -> Path:
        raw_graph_dir = self.config.get("paths.graph_dir")
        if raw_graph_dir is None:
            return self.nodes_path.parent / "graph_stats.json"
        return self.config.path("graph_dir") / "graph_stats.json"

    def _reset_state(self) -> None:
        """Allow build() to be called more than once on the same instance."""
        self.df = None
        self.nodes = {}
        self.hierarchy_edges = {}
        self.document_edge_accumulator = {}
        self.document_node_by_id = {}
        self.document_ids_by_number = defaultdict(list)

    def load(self) -> pd.DataFrame:
        if not self.input_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {self.input_path}")

        self.logger.info("Đọc dữ liệu từ %s", self.input_path)
        self.df = pd.read_parquet(self.input_path, engine="pyarrow")

        missing_columns = REQUIRED_COLUMNS - set(self.df.columns)
        if missing_columns:
            raise ValueError(
                "File Parquet thiếu các cột bắt buộc: "
                + ", ".join(sorted(missing_columns))
            )

        self.logger.info(
            "Đã đọc %s dòng thuộc %s document",
            f"{len(self.df):,}",
            f"{self.df['document_id'].nunique():,}",
        )
        return self.df

    def add_node(self, node_id: str, **attributes: Any) -> None:
        if node_id not in self.nodes:
            self.nodes[node_id] = {"node_id": node_id, **attributes}
            return

        # Fill only missing attributes; never overwrite an existing value.
        current = self.nodes[node_id]
        for key, value in attributes.items():
            if key not in current or is_missing(current[key]):
                current[key] = value

    def add_hierarchy_edge(
        self,
        source: str,
        target: str,
        relationship: str,
    ) -> None:
        key = (source, target, relationship)
        if key not in self.hierarchy_edges:
            self.hierarchy_edges[key] = {
                "source": source,
                "target": target,
                "edge_type": "HIERARCHY",
                "relationship": relationship,
                "reference_count": 1,
                "scopes": "[]",
                "evidence": "[]",
                "resolution_status": "internal",
            }

    def build_document_nodes(self) -> None:
        assert self.df is not None

        document_columns = [
            "document_id",
            "document_number",
            "title",
            *[
                column
                for column in OPTIONAL_DOCUMENT_COLUMNS
                if column in self.df.columns
            ],
        ]
        grouped = self.df[document_columns].groupby("document_id", sort=False)

        for document_id, group in grouped:
            document_id_int = int(document_id)
            node_id = f"doc:{document_id_int}"
            attributes: dict[str, Any] = {
                "node_type": "document",
                "label": clean_text(first_non_empty(group["title"])),
                "document_id": document_id_int,
                "document_number": clean_text(
                    first_non_empty(group["document_number"])
                ),
                "parent_node_id": "",
                "is_virtual": False,
                "article_number": "",
                "content": "",
                "source_chunk_ids": "[]",
                "resolution_status": "internal",
            }

            for column in OPTIONAL_DOCUMENT_COLUMNS:
                attributes[column] = (
                    clean_text(first_non_empty(group[column]))
                    if column in group.columns
                    else ""
                )

            self.add_node(node_id, **attributes)
            self.document_node_by_id[document_id_int] = node_id

            number_key = normalize_document_number(attributes["document_number"])
            if number_key:
                self.document_ids_by_number[number_key].append(document_id_int)

    def get_or_create_level_node(
        self,
        node_type: str,
        label: str,
        document_id: int,
        document_number: str,
        parent_node_id: str,
        path_values: tuple[Any, ...],
        is_virtual: bool,
    ) -> str:
        node_id = stable_id(node_type, document_id, *path_values)
        self.add_node(
            node_id,
            node_type=node_type,
            label=label,
            document_id=document_id,
            document_number=document_number,
            parent_node_id=parent_node_id,
            is_virtual=is_virtual,
            article_number="",
            content="",
            source_chunk_ids="[]",
            resolution_status="internal",
            url="",
            legal_type="",
            legal_sectors="",
            issuing_authority="",
            issuance_date="",
            signers="",
        )
        return node_id

    def build_hierarchy(self) -> None:
        assert self.df is not None

        # Preserve the original rule: only Điều rows enter this hierarchy.
        # Phụ lục and Bảng rows are intentionally not represented here.
        article_mask = (
            self.df["articles"]
            .fillna("")
            .astype(str)
            .str.match(ARTICLE_RE)
        )
        article_rows = self.df.loc[article_mask].copy()

        self.logger.info(
            "Tạo hierarchy từ %s dòng Điều",
            f"{len(article_rows):,}",
        )

        article_accumulator: dict[str, dict[str, Any]] = {}

        for row in article_rows.itertuples(index=False):
            document_id = int(row.document_id)
            document_node = self.document_node_by_id[document_id]
            document_number = clean_text(row.document_number)
            article_heading = clean_text(row.articles)
            article_match = ARTICLE_RE.match(article_heading)
            if article_match is None:
                continue

            article_number = article_match.group(1).upper()

            part_label = valid_heading(row.part, PART_RE)
            chapter_label = valid_heading(row.chapter, CHAPTER_RE)
            section_label = valid_heading(row.section, SECTION_RE)

            part_key = part_label or VIRTUAL_PART
            chapter_key = chapter_label or VIRTUAL_CHAPTER
            section_key = section_label or VIRTUAL_SECTION

            part_node = self.get_or_create_level_node(
                node_type="part",
                label=part_label or VIRTUAL_LABELS[VIRTUAL_PART],
                document_id=document_id,
                document_number=document_number,
                parent_node_id=document_node,
                path_values=(part_key,),
                is_virtual=not bool(part_label),
            )
            self.add_hierarchy_edge(
                document_node,
                part_node,
                "CONTAINS_PART",
            )

            chapter_node = self.get_or_create_level_node(
                node_type="chapter",
                label=chapter_label or VIRTUAL_LABELS[VIRTUAL_CHAPTER],
                document_id=document_id,
                document_number=document_number,
                parent_node_id=part_node,
                path_values=(part_key, chapter_key),
                is_virtual=not bool(chapter_label),
            )
            self.add_hierarchy_edge(
                part_node,
                chapter_node,
                "CONTAINS_CHAPTER",
            )

            section_node = self.get_or_create_level_node(
                node_type="section",
                label=section_label or VIRTUAL_LABELS[VIRTUAL_SECTION],
                document_id=document_id,
                document_number=document_number,
                parent_node_id=chapter_node,
                path_values=(part_key, chapter_key, section_key),
                is_virtual=not bool(section_label),
            )
            self.add_hierarchy_edge(
                chapter_node,
                section_node,
                "CONTAINS_SECTION",
            )

            article_node = stable_id(
                "article",
                document_id,
                part_key,
                chapter_key,
                section_key,
                article_number,
            )

            if article_node not in article_accumulator:
                article_accumulator[article_node] = {
                    "node_id": article_node,
                    "node_type": "article",
                    "label": article_heading,
                    "document_id": document_id,
                    "document_number": document_number,
                    "parent_node_id": section_node,
                    "is_virtual": False,
                    "article_number": article_number,
                    "contents": [],
                    "chunk_ids": [],
                    "resolution_status": "internal",
                    "url": "",
                    "legal_type": "",
                    "legal_sectors": "",
                    "issuing_authority": "",
                    "issuance_date": "",
                    "signers": "",
                }

            accumulator = article_accumulator[article_node]
            content = clean_text(row.content)
            chunk_id = clean_text(row.id)

            if content and content not in accumulator["contents"]:
                accumulator["contents"].append(content)
            if chunk_id and chunk_id not in accumulator["chunk_ids"]:
                accumulator["chunk_ids"].append(chunk_id)

            self.add_hierarchy_edge(
                section_node,
                article_node,
                "CONTAINS_ARTICLE",
            )

        for article_node, record in article_accumulator.items():
            contents = record.pop("contents")
            chunk_ids = record.pop("chunk_ids")
            record["content"] = "\n\n".join(contents)
            record["source_chunk_ids"] = json.dumps(
                chunk_ids,
                ensure_ascii=False,
            )
            self.add_node(
                article_node,
                **{key: value for key, value in record.items() if key != "node_id"},
            )

    def resolve_target_document(
        self,
        document_number: str,
    ) -> tuple[str, str]:
        """Resolve a reference to one internal or synthetic external node."""
        number_key = normalize_document_number(document_number)
        candidates = self.document_ids_by_number.get(number_key, [])

        if len(candidates) == 1:
            return self.document_node_by_id[candidates[0]], "internal"

        status = "not_found" if not candidates else "ambiguous"
        external_node = stable_id("external_doc", number_key)
        self.add_node(
            external_node,
            node_type="external_document",
            label=document_number,
            document_id=None,
            document_number=document_number,
            parent_node_id="",
            is_virtual=False,
            article_number="",
            content="",
            source_chunk_ids="[]",
            resolution_status=status,
            url="",
            legal_type="",
            legal_sectors="",
            issuing_authority="",
            issuance_date="",
            signers="",
        )
        return external_node, status

    def build_document_edges(self) -> None:
        assert self.df is not None

        self.logger.info("Tạo cạnh document-to-document từ link_to")

        for row in self.df.itertuples(index=False):
            source_document_id = int(row.document_id)
            source_node = self.document_node_by_id[source_document_id]
            source_number = normalize_document_number(row.document_number)

            for link in iter_link_items(row.link_to):
                target_number_raw = clean_text(link.get("document_number"))
                target_number = normalize_document_number(target_number_raw)
                if not target_number:
                    continue

                # Do not create a document-level self-loop for an internal ref.
                if target_number == source_number:
                    continue

                target_node, resolution_status = self.resolve_target_document(
                    target_number_raw
                )
                if source_node == target_node:
                    continue

                relationship = normalize_relation(link.get("relationship"))
                key = (source_node, target_node, relationship)

                if key not in self.document_edge_accumulator:
                    self.document_edge_accumulator[key] = {
                        "source": source_node,
                        "target": target_node,
                        "edge_type": "DOCUMENT_LINK",
                        "relationship": relationship,
                        "reference_count": 0,
                        "scopes": set(),
                        "evidence": [],
                        "resolution_status": resolution_status,
                    }

                record = self.document_edge_accumulator[key]
                record["reference_count"] += 1

                scope = clean_text(link.get("scope"))
                if scope:
                    record["scopes"].add(scope)

                raw_text = clean_text(link.get("raw_text"))
                if raw_text and raw_text not in record["evidence"]:
                    # Preserve the original size guard for edge evidence.
                    if len(record["evidence"]) < 10:
                        record["evidence"].append(raw_text)

    def make_nodes_dataframe(self) -> pd.DataFrame:
        node_columns = [
            "node_id",
            "node_type",
            "label",
            "document_id",
            "document_number",
            "parent_node_id",
            "is_virtual",
            "article_number",
            "content",
            "source_chunk_ids",
            "resolution_status",
            "url",
            "legal_type",
            "legal_sectors",
            "issuing_authority",
            "issuance_date",
            "signers",
        ]

        nodes_df = pd.DataFrame(self.nodes.values())
        for column in node_columns:
            if column not in nodes_df.columns:
                nodes_df[column] = ""

        return (
            nodes_df[node_columns]
            .sort_values(
                ["node_type", "document_id", "node_id"],
                kind="stable",
                na_position="last",
            )
            .reset_index(drop=True)
        )

    def make_edges_dataframe(self) -> pd.DataFrame:
        records = list(self.hierarchy_edges.values())

        for record in self.document_edge_accumulator.values():
            records.append(
                {
                    **record,
                    "scopes": list_to_json(sorted(record["scopes"])),
                    "evidence": list_to_json(record["evidence"]),
                }
            )

        edges_df = pd.DataFrame(records)
        if edges_df.empty:
            return pd.DataFrame(
                columns=[
                    "edge_id",
                    "source",
                    "target",
                    "edge_type",
                    "relationship",
                    "reference_count",
                    "scopes",
                    "evidence",
                    "resolution_status",
                ]
            )

        edges_df.insert(
            0,
            "edge_id",
            [
                stable_id("edge", row.source, row.target, row.relationship)
                for row in edges_df.itertuples(index=False)
            ],
        )

        return (
            edges_df.sort_values(
                ["edge_type", "relationship", "source", "target"],
                kind="stable",
            )
            .reset_index(drop=True)
        )

    @staticmethod
    def validate_graph(
        nodes_df: pd.DataFrame,
        edges_df: pd.DataFrame,
    ) -> None:
        if nodes_df["node_id"].duplicated().any():
            raise ValueError("Graph có node_id bị trùng.")

        node_ids = set(nodes_df["node_id"])
        invalid_sources = set(edges_df["source"]) - node_ids
        invalid_targets = set(edges_df["target"]) - node_ids

        if invalid_sources or invalid_targets:
            raise ValueError(
                "Graph có edge trỏ tới node không tồn tại. "
                f"invalid_sources={len(invalid_sources)}, "
                f"invalid_targets={len(invalid_targets)}"
            )

        if (edges_df["source"] == edges_df["target"]).any():
            raise ValueError("Graph còn self-loop không mong muốn.")

    def export(
        self,
        nodes_df: pd.DataFrame,
        edges_df: pd.DataFrame,
    ) -> dict[str, Path]:
        self.nodes_path.parent.mkdir(parents=True, exist_ok=True)
        self.edges_path.parent.mkdir(parents=True, exist_ok=True)
        self.stats_path.parent.mkdir(parents=True, exist_ok=True)

        nodes_df.to_parquet(self.nodes_path, index=False, engine="pyarrow")
        edges_df.to_parquet(self.edges_path, index=False, engine="pyarrow")

        stats = {
            "input_file": str(self.input_path),
            "node_count": int(len(nodes_df)),
            "edge_count": int(len(edges_df)),
            "nodes_by_type": {
                str(key): int(value)
                for key, value in (
                    nodes_df["node_type"].value_counts().sort_index().items()
                )
            },
            "edges_by_relationship": {
                str(key): int(value)
                for key, value in (
                    edges_df["relationship"].value_counts().sort_index().items()
                )
            },
            "document_link_resolution": {
                str(key): int(value)
                for key, value in (
                    edges_df.loc[
                        edges_df["edge_type"] == "DOCUMENT_LINK",
                        "resolution_status",
                    ]
                    .value_counts()
                    .sort_index()
                    .items()
                )
            },
        }

        with self.stats_path.open("w", encoding="utf-8") as file:
            json.dump(stats, file, ensure_ascii=False, indent=2)

        return {
            "nodes": self.nodes_path,
            "edges": self.edges_path,
            "stats": self.stats_path,
        }

    def run(self) -> dict[str, Path]:
        """Internal alias matching the original standalone builder flow."""
        self._reset_state()
        self.load()
        self.build_document_nodes()
        self.build_hierarchy()
        self.build_document_edges()

        nodes_df = self.make_nodes_dataframe()
        edges_df = self.make_edges_dataframe()
        self.validate_graph(nodes_df, edges_df)
        paths = self.export(nodes_df, edges_df)

        self.logger.info(
            "Hoàn thành: %s nodes, %s edges",
            f"{len(nodes_df):,}",
            f"{len(edges_df):,}",
        )
        return paths

    def build(self) -> tuple[Path, Path]:
        """Build and save the graph while preserving the existing interface."""
        paths = self.run()
        return paths["nodes"], paths["edges"]
