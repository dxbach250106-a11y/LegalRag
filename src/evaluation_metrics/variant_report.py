from __future__ import annotations

import ast
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd


K_VALUES = (5, 10, 100, 200)
TOKEN_PATTERN = re.compile(r"""['"]([^'"]+)['"]""")


def _parse_ids(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]

    if hasattr(value, "tolist"):
        try:
            converted = value.tolist()
            if isinstance(converted, list):
                return [str(x).strip() for x in converted if str(x).strip()]
        except Exception:
            pass

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []

        try:
            parsed = json.loads(text)
            if isinstance(parsed, (list, tuple, set)):
                return [str(x).strip() for x in parsed if str(x).strip()]
            return [str(parsed).strip()]
        except Exception:
            pass

        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                return [str(x).strip() for x in parsed if str(x).strip()]
            return [str(parsed).strip()]
        except Exception:
            pass

        tokens = TOKEN_PATTERN.findall(text)
        if tokens:
            return [x.strip() for x in tokens if x.strip()]

    return [str(value).strip()]


def _value(result: Any, key: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


def _plain_ids(results: list[Any]) -> list[str]:
    output, seen = [], set()

    for result in results:
        chunk_id = _value(result, "chunk_id")
        if chunk_id is None:
            continue

        chunk_id = str(chunk_id).strip()
        if chunk_id and chunk_id not in seen:
            seen.add(chunk_id)
            output.append(chunk_id)

    return output


def _graph_ids(results: list[Any]) -> list[str]:
    """
    Mỗi graph result lấy đúng một source chunk đại diện.
    Không đưa article:..., document:... vào metric.
    """
    output, seen = [], set()

    for result in results:
        metadata = _value(result, "metadata", {}) or {}
        selected = None

        if isinstance(metadata, dict):
            original = _parse_ids(metadata.get("original_seed_chunk_id"))
            source_ids = _parse_ids(metadata.get("source_chunk_ids"))

            if original:
                selected = original[0]
            elif source_ids:
                selected = source_ids[0]

        if selected is None:
            result_id = _value(result, "chunk_id")
            if result_id is not None:
                result_id = str(result_id).strip()
                if ":" not in result_id:
                    selected = result_id

        if selected:
            selected = str(selected).strip()
            if selected not in seen:
                seen.add(selected)
                output.append(selected)

    return output


def _recall(ranked: list[str], relevant: list[str], k: int) -> float:
    relevant_set = set(map(str, relevant))
    if not relevant_set:
        return 0.0
    return len(set(ranked[:k]) & relevant_set) / len(relevant_set)


def _mrr(ranked: list[str], relevant: list[str], k: int = 10) -> float:
    relevant_set = set(map(str, relevant))

    for rank, chunk_id in enumerate(ranked[:k], start=1):
        if chunk_id in relevant_set:
            return 1.0 / rank

    return 0.0


class VariantReporter:
    def __init__(
        self,
        retriever: Any,
        graph_expander: Any,
        *,
        output_dir: str | Path,
        log_every: int = 10,
    ) -> None:
        self.retriever = retriever
        self.graph_expander = graph_expander
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_every = log_every
        self.max_k = max(K_VALUES)

    def _call_method(
        self,
        names: tuple[str, ...],
        query: str,
        top_k: int,
    ) -> list[Any]:
        last_error = None

        for name in names:
            method = getattr(self.retriever, name, None)
            if not callable(method):
                continue

            attempts = (
                lambda: method(query=query, top_k=top_k),
                lambda: method(query, top_k),
                lambda: method(query=query, k=top_k),
            )

            for attempt in attempts:
                try:
                    return list(attempt())
                except TypeError as error:
                    last_error = error

        raise AttributeError(
            f"Không tìm thấy method trong retriever: {names}. "
            f"Lỗi cuối: {last_error}"
        )

    def _bm25(self, query: str, params: dict[str, Any]) -> list[Any]:
        return self._call_method(
            ("search_bm25", "bm25_search", "_search_bm25", "search_lexical"),
            query,
            self.max_k,
        )

    def _vector(self, query: str, params: dict[str, Any]) -> list[Any]:
        return self._call_method(
            ("search_faiss", "faiss_search", "_search_faiss", "search_vector", "vector_search"),
            query,
            self.max_k,
        )

    def _hybrid(self, query: str, params: dict[str, Any]) -> list[Any]:
        return list(
            self.retriever.search(
                query=query,
                top_k=self.max_k,
                bm25_top_k=max(int(params.get("bm25_top_k", self.max_k)), self.max_k),
                faiss_top_k=max(int(params.get("faiss_top_k", self.max_k)), self.max_k),
                rrf_k=int(params.get("rrf_k", 60)),
            )
        )

    def _hybrid_graph(self, query: str, params: dict[str, Any]) -> list[Any]:
        graph_weight = float(params.get("graph_weight", 0.1))
        self.graph_expander.graph_weight = graph_weight
        self.graph_expander.retrieval_weight = 1.0 - graph_weight
        self.graph_expander.hop_decay = float(params.get("hop_decay", 0.5))

        seeds = list(
            self.retriever.search(
                query=query,
                top_k=max(int(params.get("seed_top_k", self.max_k)), self.max_k),
                bm25_top_k=max(int(params.get("bm25_top_k", self.max_k)), self.max_k),
                faiss_top_k=max(int(params.get("faiss_top_k", self.max_k)), self.max_k),
                rrf_k=int(params.get("rrf_k", 60)),
            )
        )

        return list(
            self.graph_expander.expand(
                seeds,
                final_top_k=self.max_k,
                max_hops=int(params.get("max_hops", 1)),
                max_neighbors_per_node=int(params.get("max_neighbors_per_node", 10)),
                include_seeds=True,
            )
        )

    def _evaluate_variant(
        self,
        test_df: pd.DataFrame,
        name: str,
        search_fn: Callable[[str, dict[str, Any]], list[Any]],
        id_fn: Callable[[list[Any]], list[str]],
        best_params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        rows = []
        missing = 0
        start = time.perf_counter()

        print()
        print("=" * 110)
        print(f"TEST VARIANT: {name}")
        print("=" * 110)

        for index, (_, sample) in enumerate(test_df.iterrows(), start=1):
            try:
                results = search_fn(str(sample["query"]), best_params)
                ranked_ids = id_fn(results)
                is_missing = int(len(ranked_ids) == 0)
            except Exception as error:
                ranked_ids = []
                is_missing = 1
                print(
                    f"[{name}] Query {index} lỗi: "
                    f"{type(error).__name__}: {error}"
                )

            missing += is_missing
            relevant = _parse_ids(sample["relevant_chunk_ids"])

            row = {
                "variant": name,
                "query_id": str(sample["query_id"]),
                "missing": is_missing,
                "recall@5": _recall(ranked_ids, relevant, 5),
                "recall@10": _recall(ranked_ids, relevant, 10),
                "recall@100": _recall(ranked_ids, relevant, 100),
                "recall@200": _recall(ranked_ids, relevant, 200),
                "mrr@10": _mrr(ranked_ids, relevant, 10),
            }
            rows.append(row)

            if index == 1 or index % self.log_every == 0 or index == len(test_df):
                elapsed = time.perf_counter() - start
                eta = elapsed / index * (len(test_df) - index)

                print(
                    f"[{name}] {index:>3}/{len(test_df)} | "
                    f"{index / len(test_df) * 100:>6.2f}% | "
                    f"missing={missing} | ETA={eta / 60:>7.2f} phút"
                )

        return rows

    def run(
        self,
        test_df: pd.DataFrame,
        *,
        best_params: dict[str, Any],
    ) -> pd.DataFrame:
        variants = [
            ("bm25", self._bm25, _plain_ids),
            ("vector", self._vector, _plain_ids),
            ("hybrid", self._hybrid, _plain_ids),
            ("hybrid_graph", self._hybrid_graph, _graph_ids),
        ]

        all_rows = []

        for name, search_fn, id_fn in variants:
            all_rows.extend(
                self._evaluate_variant(
                    test_df,
                    name,
                    search_fn,
                    id_fn,
                    best_params,
                )
            )

        detail_df = pd.DataFrame(all_rows)

        report = (
            detail_df.groupby("variant", sort=False)
            .agg(
                queries=("query_id", "count"),
                missing=("missing", "sum"),
                **{
                    "recall@5": ("recall@5", "mean"),
                    "recall@10": ("recall@10", "mean"),
                    "recall@100": ("recall@100", "mean"),
                    "recall@200": ("recall@200", "mean"),
                    "mrr@10": ("mrr@10", "mean"),
                },
            )
            .reset_index()
            .rename(columns={"variant": "Variant"})
        )

        detail_df.to_csv(
            self.output_dir / "variant_per_query_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )

        report.to_csv(
            self.output_dir / "retrieval_variant_report.csv",
            index=False,
            encoding="utf-8-sig",
        )

        report.to_latex(
            self.output_dir / "retrieval_variant_report.tex",
            index=False,
            float_format="%.4f",
            escape=True,
        )

        print()
        print("=" * 120)
        print("RETRIEVAL PERFORMANCE OF DIFFERENT SYSTEM VARIANTS")
        print("=" * 120)
        print(
            report.to_string(
                index=False,
                float_format=lambda value: f"{value:.4f}",
            )
        )
        print("=" * 120)

        return report
