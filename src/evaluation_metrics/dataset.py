from __future__ import annotations

import ast
import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvaluationQuery:
    query_id: str
    query: str
    relevant_chunk_ids: list[str]


class EvaluationDataset:
    """Load and deterministically split the query relevance dataset."""

    @classmethod
    def load(cls, path: str | Path) -> list[EvaluationQuery]:
        source = Path(path)
        suffix = source.suffix.lower()
        if suffix == ".jsonl":
            return cls.load_jsonl(source)
        if suffix == ".json":
            return cls.load_json(source)
        if suffix == ".csv":
            return cls.load_csv(source)
        raise ValueError(f"Unsupported evaluation dataset format: {source.suffix}")

    @classmethod
    def load_jsonl(cls, path: str | Path) -> list[EvaluationQuery]:
        items: list[EvaluationQuery] = []
        with Path(path).open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                items.append(cls._parse_item(raw, fallback_id=str(line_number)))
        return items

    @classmethod
    def load_json(cls, path: str | Path) -> list[EvaluationQuery]:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = raw.get("data", raw) if isinstance(raw, dict) else raw
        if not isinstance(rows, list):
            raise ValueError("JSON evaluation file must contain a list or {'data': [...]}.")
        return [cls._parse_item(item, str(index)) for index, item in enumerate(rows)]

    @classmethod
    def load_csv(cls, path: str | Path) -> list[EvaluationQuery]:
        import pandas as pd

        frame = pd.read_csv(path)
        return [
            cls._parse_item(row.to_dict(), str(index))
            for index, row in frame.iterrows()
        ]

    @staticmethod
    def _parse_item(raw: dict, fallback_id: str) -> EvaluationQuery:
        query = raw.get("query", raw.get("question", raw.get("text")))
        if query is None or not str(query).strip():
            raise ValueError(f"Evaluation item {fallback_id} has no query/question/text field.")
        relevant = raw.get(
            "relevant_chunk_ids",
            raw.get("relevant_ids", raw.get("ground_truth", raw.get("answers", []))),
        )
        relevant_ids = EvaluationDataset.parse_relevant_ids(relevant)
        return EvaluationQuery(
            query_id=str(raw.get("query_id", raw.get("id", fallback_id))),
            query=str(query).strip(),
            relevant_chunk_ids=relevant_ids,
        )

    @staticmethod
    def parse_relevant_ids(value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        if not text:
            return []
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                if isinstance(parsed, (list, tuple, set)):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except (ValueError, TypeError, SyntaxError, json.JSONDecodeError):
                pass
        separators = ["|", ";", ","]
        for separator in separators:
            if separator in text:
                return [part.strip().strip("'\"") for part in text.split(separator) if part.strip()]
        return [text.strip("'\"")]

    @staticmethod
    def split(
        items: list[EvaluationQuery],
        train_size: int,
        test_size: int,
        seed: int,
        shuffle: bool = True,
    ) -> tuple[list[EvaluationQuery], list[EvaluationQuery]]:
        copied = list(items)
        if shuffle:
            random.Random(seed).shuffle(copied)
        required = train_size + test_size
        if required > len(copied):
            raise ValueError(
                f"Need {required} queries for train/test, but dataset has only {len(copied)}."
            )
        return copied[:train_size], copied[train_size:required]
