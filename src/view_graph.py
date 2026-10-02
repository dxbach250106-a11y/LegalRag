from __future__ import annotations

import html
import json
import math
import webbrowser
from collections import Counter, deque
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import pandas as pd
from pyvis.network import Network


# =============================================================================
# PATHS / SETTINGS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

GRAPH_DIR = ROOT_DIR / "graph"
NODES_PATH = GRAPH_DIR / "nodes.parquet"
EDGES_PATH = GRAPH_DIR / "edges.parquet"
OUTPUT_PATH = ROOT_DIR / "legal_graph.html"

# Không nên render toàn bộ graph lớn trong trình duyệt.
# Tăng các giá trị này nếu máy đủ mạnh.
MAX_NODES = 700
MAX_EDGES = 1800

OPEN_BROWSER = True
SHOW_EDGE_LABELS = True
USE_PHYSICS = True

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

NODE_ID_COLUMN_CANDIDATES = (
    "node_id",
    "id",
    "graph_node_id",
)

NODE_TYPE_COLUMN_CANDIDATES = (
    "node_type",
    "type",
    "entity_type",
)

NODE_LABEL_COLUMN_CANDIDATES = (
    "label",
    "name",
    "title",
    "document_number",
)

EDGE_TYPE_COLUMN_CANDIDATES = (
    "relationship",
    "edge_type",
    "relation",
    "type",
)


# =============================================================================
# NORMALIZATION
# =============================================================================

def first_existing_column(
    frame: pd.DataFrame,
    candidates: Iterable[str],
    *,
    required: bool = True,
    description: str = "column",
) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

    if required:
        raise ValueError(
            f"Không tìm thấy {description}. "
            f"Các cột hiện có: {list(frame.columns)}"
        )

    return None


def is_missing(value: Any) -> bool:
    if value is None:
        return True

    try:
        result = pd.isna(value)
        if isinstance(result, bool):
            return result
    except (TypeError, ValueError):
        pass

    return False


def normalize_node_id(value: Any) -> str | None:
    """
    Chuẩn hóa ID để ID ở nodes.parquet và edges.parquet khớp nhau.

    Ví dụ:
        123       -> "123"
        123.0     -> "123"
        " 123 "   -> "123"
    """

    if is_missing(value):
        return None

    if isinstance(value, bool):
        return str(value)

    if isinstance(value, int):
        return str(value)

    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        if value.is_integer():
            return str(int(value))
        return str(value).strip()

    node_id = str(value).strip()

    if not node_id:
        return None

    # Chuẩn hóa chuỗi "123.0" thành "123".
    if node_id.endswith(".0"):
        prefix = node_id[:-2]
        if prefix.lstrip("-").isdigit():
            return prefix

    return node_id


def json_safe(value: Any) -> Any:
    """Chuyển metadata thành kiểu PyVis/JSON có thể serialize."""

    if is_missing(value):
        return None

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]

    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except (TypeError, ValueError):
            pass

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def row_to_metadata(
    row: pd.Series,
    excluded_columns: set[str],
) -> dict[str, Any]:
    return {
        str(key): json_safe(value)
        for key, value in row.items()
        if key not in excluded_columns and not is_missing(value)
    }


def get_first_value(
    metadata: dict[str, Any],
    candidates: Iterable[str],
    default: Any = None,
) -> Any:
    for candidate in candidates:
        value = metadata.get(candidate)

        if value is None:
            continue

        text = str(value).strip()
        if text:
            return value

    return default


# =============================================================================
# LOAD / VALIDATE
# =============================================================================

def load_graph_frames() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    str,
    str,
    str,
]:
    if not NODES_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy nodes.parquet:\n{NODES_PATH.resolve()}"
        )

    if not EDGES_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy edges.parquet:\n{EDGES_PATH.resolve()}"
        )

    nodes = pd.read_parquet(NODES_PATH)
    edges = pd.read_parquet(EDGES_PATH)

    node_id_col = first_existing_column(
        nodes,
        NODE_ID_COLUMN_CANDIDATES,
        description="cột ID trong nodes.parquet",
    )

    source_col = first_existing_column(
        edges,
        SOURCE_COLUMN_CANDIDATES,
        description="cột source trong edges.parquet",
    )

    target_col = first_existing_column(
        edges,
        TARGET_COLUMN_CANDIDATES,
        description="cột target trong edges.parquet",
    )

    print("=" * 100)
    print("THÔNG TIN DỮ LIỆU GRAPH")
    print("=" * 100)
    print(f"Nodes path       : {NODES_PATH.resolve()}")
    print(f"Edges path       : {EDGES_PATH.resolve()}")
    print(f"Node rows        : {len(nodes):,}")
    print(f"Edge rows        : {len(edges):,}")
    print(f"Node ID column   : {node_id_col}")
    print(f"Source column    : {source_col}")
    print(f"Target column    : {target_col}")
    print(f"Node columns     : {list(nodes.columns)}")
    print(f"Edge columns     : {list(edges.columns)}")

    if nodes.empty:
        raise RuntimeError("nodes.parquet đang rỗng.")

    if edges.empty:
        raise RuntimeError(
            "edges.parquet đang rỗng nên không thể hiển thị cạnh."
        )

    print("\n5 CẠNH ĐẦU TIÊN:")
    print(
        edges[[source_col, target_col]]
        .head()
        .to_string(index=False)
    )

    return nodes, edges, node_id_col, source_col, target_col


def inspect_id_consistency(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    node_id_col: str,
    source_col: str,
    target_col: str,
) -> None:
    node_ids = {
        normalized
        for normalized in (
            normalize_node_id(value)
            for value in nodes[node_id_col]
        )
        if normalized is not None
    }

    source_ids = {
        normalized
        for normalized in (
            normalize_node_id(value)
            for value in edges[source_col]
        )
        if normalized is not None
    }

    target_ids = {
        normalized
        for normalized in (
            normalize_node_id(value)
            for value in edges[target_col]
        )
        if normalized is not None
    }

    source_matches = source_ids & node_ids
    target_matches = target_ids & node_ids

    print("\n" + "=" * 100)
    print("KIỂM TRA ID GIỮA NODES VÀ EDGES")
    print("=" * 100)
    print(f"Unique node IDs         : {len(node_ids):,}")
    print(f"Unique source IDs       : {len(source_ids):,}")
    print(f"Unique target IDs       : {len(target_ids):,}")
    print(f"Source IDs khớp nodes   : {len(source_matches):,}")
    print(f"Target IDs khớp nodes   : {len(target_matches):,}")
    print(
        f"Source không có metadata: "
        f"{len(source_ids - node_ids):,}"
    )
    print(
        f"Target không có metadata: "
        f"{len(target_ids - node_ids):,}"
    )

    missing_sources = list(source_ids - node_ids)[:10]
    missing_targets = list(target_ids - node_ids)[:10]

    if missing_sources:
        print("Ví dụ source thiếu metadata:", missing_sources)

    if missing_targets:
        print("Ví dụ target thiếu metadata:", missing_targets)

    if not source_matches and not target_matches:
        print(
            "\nCẢNH BÁO: ID trong nodes.parquet và edges.parquet "
            "không khớp. Graph vẫn được tạo từ edges, nhưng nhiều node "
            "sẽ chỉ có nhãn placeholder."
        )


# =============================================================================
# BUILD GRAPH
# =============================================================================

def build_graph(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    node_id_col: str,
    source_col: str,
    target_col: str,
) -> nx.DiGraph:
    graph = nx.DiGraph()

    node_metadata: dict[str, dict[str, Any]] = {}

    for _, row in nodes.iterrows():
        node_id = normalize_node_id(row.get(node_id_col))

        if node_id is None:
            continue

        metadata = row_to_metadata(
            row,
            excluded_columns={node_id_col},
        )

        node_metadata[node_id] = metadata
        graph.add_node(node_id, **metadata)

    skipped_edges = 0
    self_loops = 0

    for _, row in edges.iterrows():
        source = normalize_node_id(row.get(source_col))
        target = normalize_node_id(row.get(target_col))

        if source is None or target is None:
            skipped_edges += 1
            continue

        if source == target:
            self_loops += 1

        if source not in graph:
            graph.add_node(
                source,
                node_type="missing_metadata",
                label=source,
            )

        if target not in graph:
            graph.add_node(
                target,
                node_type="missing_metadata",
                label=target,
            )

        edge_metadata = row_to_metadata(
            row,
            excluded_columns={source_col, target_col},
        )

        # DiGraph chỉ giữ một edge cho mỗi cặp source-target.
        # Nếu có edge trùng, gộp số lần xuất hiện vào duplicate_count.
        if graph.has_edge(source, target):
            existing = graph[source][target]
            existing["duplicate_count"] = (
                int(existing.get("duplicate_count", 1)) + 1
            )

            relationships = existing.setdefault(
                "relationships",
                [],
            )

            relationship = get_first_value(
                edge_metadata,
                EDGE_TYPE_COLUMN_CANDIDATES,
            )

            if (
                relationship is not None
                and relationship not in relationships
            ):
                relationships.append(relationship)

            # Không overwrite metadata gốc bằng giá trị rỗng.
            for key, value in edge_metadata.items():
                if key not in existing and value is not None:
                    existing[key] = value
        else:
            edge_metadata["duplicate_count"] = 1
            graph.add_edge(
                source,
                target,
                **edge_metadata,
            )

    print("\n" + "=" * 100)
    print("KẾT QUẢ BUILD NETWORKX GRAPH")
    print("=" * 100)
    print(f"Graph nodes     : {graph.number_of_nodes():,}")
    print(f"Graph edges     : {graph.number_of_edges():,}")
    print(f"Skipped edges   : {skipped_edges:,}")
    print(f"Self-loop rows  : {self_loops:,}")

    if graph.number_of_edges() == 0:
        raise RuntimeError(
            "Graph không có cạnh sau khi đọc edges.parquet. "
            "Hãy kiểm tra giá trị source/target."
        )

    relationship_counter: Counter[str] = Counter()

    for _, _, metadata in graph.edges(data=True):
        relationship = str(
            get_first_value(
                metadata,
                EDGE_TYPE_COLUMN_CANDIDATES,
                "RELATED_TO",
            )
        )
        relationship_counter[relationship] += 1

    print("\nCác loại edge phổ biến:")
    for relationship, count in relationship_counter.most_common(15):
        print(f"  {relationship}: {count:,}")

    print("\n10 CẠNH ĐẦU TIÊN TRONG NETWORKX:")
    for index, (source, target, metadata) in enumerate(
        graph.edges(data=True),
        start=1,
    ):
        relationship = get_first_value(
            metadata,
            EDGE_TYPE_COLUMN_CANDIDATES,
            "RELATED_TO",
        )
        print(f"  {source} --[{relationship}]--> {target}")

        if index >= 10:
            break

    return graph


# =============================================================================
# SELECT CONNECTED PREVIEW
# =============================================================================

def largest_weak_component(graph: nx.DiGraph) -> set[str]:
    components = list(nx.weakly_connected_components(graph))

    if not components:
        return set()

    return max(
        components,
        key=lambda component: (
            graph.subgraph(component).number_of_edges(),
            len(component),
        ),
    )


def bfs_nodes(
    graph: nx.DiGraph,
    seed: str,
    max_nodes: int,
) -> list[str]:
    undirected = graph.to_undirected(as_view=True)

    selected: list[str] = []
    visited = {seed}
    queue: deque[str] = deque([seed])

    while queue and len(selected) < max_nodes:
        current = queue.popleft()
        selected.append(current)

        neighbors = sorted(
            undirected.neighbors(current),
            key=lambda node: undirected.degree(node),
            reverse=True,
        )

        for neighbor in neighbors:
            if neighbor in visited:
                continue

            visited.add(neighbor)
            queue.append(neighbor)

            if len(visited) >= max_nodes * 2:
                # Tránh queue tăng quá lớn ở hub.
                break

    return selected


def cap_edges(
    graph: nx.DiGraph,
    max_edges: int,
) -> nx.DiGraph:
    if graph.number_of_edges() <= max_edges:
        return graph.copy()

    preview = nx.DiGraph()

    # Ưu tiên edge nối các node bậc cao để giữ cấu trúc chính.
    sorted_edges = sorted(
        graph.edges(data=True),
        key=lambda item: (
            graph.degree(item[0]) + graph.degree(item[1]),
            int(item[2].get("duplicate_count", 1)),
        ),
        reverse=True,
    )

    for source, target, metadata in sorted_edges[:max_edges]:
        if source not in preview:
            preview.add_node(
                source,
                **dict(graph.nodes[source]),
            )

        if target not in preview:
            preview.add_node(
                target,
                **dict(graph.nodes[target]),
            )

        preview.add_edge(
            source,
            target,
            **dict(metadata),
        )

    return preview


def select_visual_subgraph(
    graph: nx.DiGraph,
    max_nodes: int = MAX_NODES,
    max_edges: int = MAX_EDGES,
) -> nx.DiGraph:
    """
    Lấy một subgraph thật sự có liên kết:

    1. Chọn weakly-connected component lớn nhất theo số edge.
    2. Bắt đầu từ node có degree cao nhất.
    3. BFS để lấy các node lân cận.
    4. Giới hạn số edge nếu cần.

    Không dùng list(graph.nodes())[:N], vì cách đó có thể chọn các node
    không có cạnh nội bộ.
    """

    component_nodes = largest_weak_component(graph)

    if not component_nodes:
        raise RuntimeError("Không tìm thấy connected component nào.")

    component = graph.subgraph(component_nodes).copy()

    seed = max(
        component.nodes(),
        key=lambda node: component.degree(node),
    )

    selected_nodes = bfs_nodes(
        component,
        seed=seed,
        max_nodes=max_nodes,
    )

    selected = component.subgraph(selected_nodes).copy()
    selected = cap_edges(selected, max_edges=max_edges)

    # Xóa isolates có thể sinh ra sau khi cap edge.
    isolates = list(nx.isolates(selected))
    selected.remove_nodes_from(isolates)

    print("\n" + "=" * 100)
    print("SUBGRAPH DÙNG ĐỂ HIỂN THỊ")
    print("=" * 100)
    print(f"Largest component nodes : {component.number_of_nodes():,}")
    print(f"Largest component edges : {component.number_of_edges():,}")
    print(f"Seed node               : {seed}")
    print(f"Preview nodes           : {selected.number_of_nodes():,}")
    print(f"Preview edges           : {selected.number_of_edges():,}")
    print(f"Removed isolates        : {len(isolates):,}")

    if selected.number_of_edges() == 0:
        raise RuntimeError(
            "Subgraph hiển thị không có edge. "
            "Hãy giảm MAX_NODES/MAX_EDGES hoặc kiểm tra cấu trúc graph."
        )

    return selected


# =============================================================================
# PYVIS
# =============================================================================

def make_node_label(
    node_id: str,
    metadata: dict[str, Any],
) -> str:
    label = get_first_value(
        metadata,
        NODE_LABEL_COLUMN_CANDIDATES,
        node_id,
    )

    label_text = str(label).strip()

    if len(label_text) > 70:
        return label_text[:67] + "..."

    return label_text


def make_node_tooltip(
    node_id: str,
    metadata: dict[str, Any],
) -> str:
    lines = [f"<b>Node ID:</b> {html.escape(node_id)}"]

    for key, value in metadata.items():
        if value is None:
            continue

        value_text = html.escape(str(value))

        if len(value_text) > 1500:
            value_text = value_text[:1500] + "..."

        lines.append(
            f"<b>{html.escape(str(key))}:</b> {value_text}"
        )

    return "<br>".join(lines)


def make_edge_tooltip(
    source: str,
    target: str,
    metadata: dict[str, Any],
) -> str:
    relationship = get_first_value(
        metadata,
        EDGE_TYPE_COLUMN_CANDIDATES,
        "RELATED_TO",
    )

    lines = [
        f"<b>Source:</b> {html.escape(source)}",
        f"<b>Target:</b> {html.escape(target)}",
        (
            "<b>Relationship:</b> "
            f"{html.escape(str(relationship))}"
        ),
    ]

    for key, value in metadata.items():
        if value is None:
            continue

        lines.append(
            f"<b>{html.escape(str(key))}:</b> "
            f"{html.escape(str(value))}"
        )

    return "<br>".join(lines)


def create_pyvis_graph(
    graph: nx.DiGraph,
    output_path: Path = OUTPUT_PATH,
) -> None:
    network = Network(
        height="900px",
        width="100%",
        directed=True,
        notebook=False,
        bgcolor="#ffffff",
        font_color="#222222",
        cdn_resources="in_line",
        select_menu=True,
        filter_menu=True,
    )

    for node_id, metadata in graph.nodes(data=True):
        node_type = str(
            get_first_value(
                metadata,
                NODE_TYPE_COLUMN_CANDIDATES,
                "unknown",
            )
        )

        degree = graph.degree(node_id)
        node_size = min(
            42,
            max(12, 11 + math.sqrt(max(degree, 1)) * 3),
        )

        network.add_node(
            node_id,
            label=make_node_label(node_id, metadata),
            title=make_node_tooltip(node_id, metadata),
            group=node_type,
            size=node_size,
            borderWidth=1,
        )

    for source, target, metadata in graph.edges(data=True):
        relationship = str(
            get_first_value(
                metadata,
                EDGE_TYPE_COLUMN_CANDIDATES,
                "RELATED_TO",
            )
        )

        duplicate_count = int(
            metadata.get("duplicate_count", 1)
        )

        edge_width = min(
            8,
            1.8 + math.log2(max(duplicate_count, 1)),
        )

        edge_kwargs: dict[str, Any] = {
            "title": make_edge_tooltip(
                source,
                target,
                metadata,
            ),
            "arrows": "to",
            "width": edge_width,
            "color": {
                "color": "#606770",
                "highlight": "#d62728",
                "hover": "#1f77b4",
                "opacity": 0.85,
            },
            "smooth": {
                "enabled": True,
                "type": "dynamic",
                "roundness": 0.25,
            },
        }

        if SHOW_EDGE_LABELS:
            edge_kwargs["label"] = relationship

        network.add_edge(
            source,
            target,
            **edge_kwargs,
        )

    options = {
        "interaction": {
            "hover": True,
            "navigationButtons": True,
            "keyboard": True,
            "multiselect": True,
            "tooltipDelay": 120,
            "hideEdgesOnDrag": False,
            "hideEdgesOnZoom": False,
        },
        "nodes": {
            "shape": "dot",
            "font": {
                "size": 13,
                "face": "Arial",
            },
        },
        "edges": {
            "arrows": {
                "to": {
                    "enabled": True,
                    "scaleFactor": 0.8,
                }
            },
            "font": {
                "size": 9,
                "align": "middle",
                "strokeWidth": 3,
                "strokeColor": "#ffffff",
            },
            "selectionWidth": 3,
        },
        "physics": {
            "enabled": USE_PHYSICS,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
                "gravitationalConstant": -55,
                "centralGravity": 0.015,
                "springLength": 135,
                "springConstant": 0.06,
                "damping": 0.55,
                "avoidOverlap": 0.35,
            },
            "stabilization": {
                "enabled": USE_PHYSICS,
                "iterations": 350,
                "updateInterval": 25,
                "fit": True,
            },
        },
    }

    network.set_options(
        json.dumps(
            options,
            ensure_ascii=False,
        )
    )

    # Không dùng network.write_html() vì trên Windows hàm này có thể
    # ghi bằng cp1252 và gây UnicodeEncodeError với tiếng Việt.
    html_content = network.generate_html(
        notebook=False,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        html_content,
        encoding="utf-8",
    )

    # Kiểm tra nhanh HTML có dữ liệu node/edge hay chưa.
    html_has_edges = (
        '"edges":' in html_content
        or "edges = new vis.DataSet" in html_content
        or "edges = new vis.DataSet(" in html_content
    )

    print("\n" + "=" * 100)
    print("KẾT QUẢ XUẤT HTML")
    print("=" * 100)
    print(f"Output path       : {output_path.resolve()}")
    print(f"HTML size         : {output_path.stat().st_size:,} bytes")
    print(f"HTML contains edge: {html_has_edges}")
    print(f"Rendered nodes    : {graph.number_of_nodes():,}")
    print(f"Rendered edges    : {graph.number_of_edges():,}")

    if OPEN_BROWSER:
        webbrowser.open(
            output_path.resolve().as_uri()
        )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    (
        nodes,
        edges,
        node_id_col,
        source_col,
        target_col,
    ) = load_graph_frames()

    inspect_id_consistency(
        nodes=nodes,
        edges=edges,
        node_id_col=node_id_col,
        source_col=source_col,
        target_col=target_col,
    )

    full_graph = build_graph(
        nodes=nodes,
        edges=edges,
        node_id_col=node_id_col,
        source_col=source_col,
        target_col=target_col,
    )

    preview_graph = select_visual_subgraph(
        full_graph,
        max_nodes=MAX_NODES,
        max_edges=MAX_EDGES,
    )

    create_pyvis_graph(
        preview_graph,
        output_path=OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
