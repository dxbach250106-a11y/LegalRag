from __future__ import annotations

import argparse
from copy import deepcopy
import logging
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from core.config import AppConfig
from graph_rerank.graph_search_pipeline import GraphSearchPipeline
from models.answer_generator import LLMAnswerGenerator
from retrieval.schemas import RetrievalResult
from utils.logger import configure_logging


SUPPORTED_METHODS = tuple(GraphSearchPipeline.SUPPORTED_METHODS)
DEMO_VERSION = "2026-06-29-general-answer-v5"


class LegalRAGDemo:
    """
    Demo hỏi đáp tương tác cho LegalRAG.

    Pipeline:
        1. Nhận query từ người dùng.
        2. Retrieval bằng một trong các phương pháp được hỗ trợ.
        3. Hydrate nội dung và metadata của top-k chunk.
        4. In các chunk cuối cùng.
        5. Sinh câu trả lời tóm tắt, có trích dẫn nội tuyến.
        6. Tự gắn danh sách nguồn và trích đoạn ngắn ở cuối câu trả lời.
        7. Retry hoặc dùng fallback trích xuất nếu model chỉ sao chép context.
    """

    SYSTEM_PROMPT = """
Bạn là trợ lý hỏi đáp pháp luật Việt Nam. Các đoạn nguồn đã được hệ thống truy
xuất vì có liên quan đến câu hỏi. Hãy đọc nguồn, xác định thông tin trực tiếp
trả lời câu hỏi và diễn đạt lại bằng tiếng Việt tự nhiên.

Quy tắc bắt buộc:
- Trả lời trực tiếp ngay phần đầu, không chép nguyên văn toàn bộ nguồn.
- Tổng hợp đủ đối tượng, điều kiện, mức tiền, khoảng cách, thời hạn, thủ tục,
  ngoại lệ hoặc phạm vi áp dụng nếu nguồn có nêu.
- Chỉ sử dụng dữ kiện có trong nguồn; không bịa số liệu hay điều khoản.
- Nếu quy định chỉ áp dụng cho một địa phương hoặc thời kỳ, phải nói rõ.
- Không tự viết danh sách nguồn, URL hoặc nhãn [n]; Python sẽ gắn nguồn sau.
- Không viết quá trình suy luận, không nhắc lại prompt và không lặp câu.
- Không được chỉ trả lời "không có câu trả lời" khi nguồn đã chứa thông tin
  liên quan. Nếu nguồn chỉ trả lời được một phần, hãy trả lời phần xác định
  được và nêu rõ giới hạn đó.

Định dạng:
Kết luận:
<một đoạn 2 đến 5 câu trả lời trực tiếp>

Thông tin chính:
- <ý quan trọng 1>
- <ý quan trọng 2>
- <ý khác nếu cần>

Độ dài thông thường 90 đến 260 từ.
""".strip()

    RETRY_PROMPT = """
Câu trả lời trước không đạt vì đã từ chối trả lời, lặp nội dung hoặc chép nguồn.
Hãy viết lại từ đầu. Các đoạn nguồn dưới đây có thông tin liên quan; hãy trả lời
phần có thể xác nhận được, tóm tắt rõ ràng và không tự tạo phần nguồn.
""".strip()

    STOPWORDS = {
        "ai", "bao", "bị", "bởi", "các", "cái", "cho", "có", "của",
        "đã", "đang", "đến", "để", "gì", "hay", "là", "làm", "một",
        "những", "ở", "ra", "sau", "sẽ", "theo", "thì", "trong",
        "từ", "và", "về", "với",
    }

    def __init__(
        self,
        config_path: str | Path = "config.json",
        method: str = "hybrid+graph",
        top_k: int = 5,
        show_full_text: bool = True,
        auto_enable_generation: bool = True,
    ) -> None:
        self.config = AppConfig.load(config_path)
        configure_logging(self.config.get("project.log_level", "INFO"))
        self.logger = logging.getLogger(self.__class__.__name__)

        method = method.lower().strip()
        if method not in SUPPORTED_METHODS:
            raise ValueError(
                f"Phương pháp {method!r} không hợp lệ. "
                f"Các phương pháp hỗ trợ: {SUPPORTED_METHODS}"
            )

        if top_k <= 0:
            raise ValueError("top_k phải lớn hơn 0.")

        self.method = method
        self.top_k = top_k
        self.show_full_text = show_full_text

        if auto_enable_generation and not bool(
            self.config.get("generation.enabled", False)
        ):
            self.config.data.setdefault("generation", {})["enabled"] = True
            self.logger.warning(
                "generation.enabled đang là false; demo đã tạm bật generation "
                "trong bộ nhớ. File config.json không bị thay đổi."
            )

        self.pipeline = GraphSearchPipeline(self.config)
        self.generator = LLMAnswerGenerator(self.config)

        self.chunk_lookup: pd.DataFrame | None = None
        self.id_column = str(self.config.get("embedding.id_column", "id"))
        self.text_column = str(self.config.get("embedding.text_column", "content"))

        self.max_new_tokens = int(
            self.config.get("generation.max_new_tokens", 420)
        )
        self.max_chars_per_source = int(
            self.config.get("generation.max_chars_per_source", 3200)
        )
        self.max_context_chars = int(
            self.config.get("generation.max_context_chars", 13000)
        )

    # ------------------------------------------------------------------
    # LOAD / RETRIEVAL
    # ------------------------------------------------------------------

    def load(self) -> "LegalRAGDemo":
        self.logger.info("Đang load retrieval indexes và graph...")
        self.pipeline.load()

        chunk_path = self.config.path("legal_chunks")
        if not chunk_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file chunk: {chunk_path}\n"
                "Hãy build/chunk dữ liệu trước khi chạy demo."
            )

        chunks = pd.read_parquet(chunk_path)

        if self.id_column not in chunks.columns:
            raise KeyError(
                f"File {chunk_path} không có cột ID {self.id_column!r}. "
                f"Các cột hiện có: {list(chunks.columns)}"
            )

        if self.text_column not in chunks.columns:
            raise KeyError(
                f"File {chunk_path} không có cột nội dung {self.text_column!r}. "
                f"Các cột hiện có: {list(chunks.columns)}"
            )

        chunks[self.id_column] = chunks[self.id_column].astype(str).str.strip()
        chunks = chunks.drop_duplicates(subset=[self.id_column], keep="first")
        self.chunk_lookup = chunks.set_index(self.id_column, drop=False)

        self.logger.info(
            "Đã load %d chunks để tra cứu nội dung.",
            len(self.chunk_lookup),
        )
        return self

    @staticmethod
    def _clean_value(value: Any) -> Any:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        return value

    def _hydrate_result(self, result: RetrievalResult) -> RetrievalResult:
        chunk_id = str(result.chunk_id).strip()

        if self.chunk_lookup is None:
            raise RuntimeError("Demo chưa được load. Hãy gọi load() trước.")

        if chunk_id not in self.chunk_lookup.index:
            self.logger.warning(
                "Không tìm thấy chunk_id=%s trong legal_chunks.parquet.",
                chunk_id,
            )
            return result

        row = self.chunk_lookup.loc[chunk_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        row_metadata = {
            str(key): self._clean_value(value)
            for key, value in row.to_dict().items()
        }
        merged_metadata = dict(row_metadata)
        merged_metadata.update(result.metadata or {})

        text = str(result.text or "").strip()
        if not text:
            text = str(row.get(self.text_column, "") or "").strip()

        return RetrievalResult(
            chunk_id=chunk_id,
            score=float(result.score),
            text=text,
            source=str(result.source or ""),
            metadata=merged_metadata,
        )

    def retrieve(self, query: str) -> list[RetrievalResult]:
        query = query.strip()
        if not query:
            raise ValueError("Query không được để trống.")

        raw_results = self.pipeline.search_method(
            query=query,
            method=self.method,
            top_k=self.top_k,
        )

        return [self._hydrate_result(result) for result in raw_results]

    # ------------------------------------------------------------------
    # DISPLAY
    # ------------------------------------------------------------------

    def _display_text(self, text: str) -> str:
        if self.show_full_text:
            return text
        if len(text) <= 1200:
            return text
        return text[:1200].rstrip() + "..."

    def print_results(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> None:
        print("\n" + "=" * 110)
        print("KẾT QUẢ RETRIEVAL CUỐI CÙNG")
        print("=" * 110)
        print(f"Query       : {query}")
        print(f"Method      : {self.method}")
        print(f"Top-k       : {len(results)}")
        print("=" * 110)

        if not results:
            print("Không tìm thấy chunk phù hợp.")
            return

        for rank, item in enumerate(results, start=1):
            metadata = item.metadata or {}
            document_number = self._clean_value(metadata.get("document_number"))
            title = self._clean_value(metadata.get("title"))
            article = self._clean_value(
                metadata.get("articles", metadata.get("article"))
            )
            url = self._clean_value(metadata.get("url"))

            print(f"\n{'-' * 110}")
            print(f"RANK       : {rank}")
            print(f"SCORE      : {float(item.score):.8f}")
            print(f"SOURCE     : {item.source or 'unknown'}")
            print(f"CHUNK ID   : {item.chunk_id}")

            if document_number:
                print(f"VĂN BẢN    : {document_number}")
            if title:
                print(f"TIÊU ĐỀ    : {title}")
            if article:
                print(f"ĐIỀU/MỤC   : {article}")
            if url:
                print(f"URL        : {url}")

            print("NỘI DUNG:")
            print(self._display_text(item.text) or "[Chunk không có nội dung]")

        print("\n" + "=" * 110)

    # ------------------------------------------------------------------
    # PROMPT / GENERATION
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_space(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _query_terms(self, query: str) -> set[str]:
        tokens = re.findall(r"[0-9A-Za-zÀ-ỹĐđ]+", query.casefold())
        return {
            token
            for token in tokens
            if len(token) > 1 and token not in self.STOPWORDS
        }

    def _split_sentences(self, text: str) -> list[str]:
        """Tách văn bản pháp luật theo câu, khoản, điểm và bullet."""
        text = self._normalize_space(text)
        if not text:
            return []

        text = re.sub(
            r"\s+(?=(?:\d{1,3}[.)]|[a-zđ][.)]|[-–•])\s+)",
            "\n",
            text,
            flags=re.IGNORECASE,
        )
        parts = re.split(
            r"(?:\n+|(?<=[.!?;:])\s+(?=[A-ZÀ-ỸĐ0-9]))",
            text,
        )

        cleaned: list[str] = []
        for part in parts:
            part = self._normalize_space(part.strip(" -–•"))
            if len(part) >= 18:
                cleaned.append(part)
        return cleaned

    def _query_phrases(self, query: str) -> set[str]:
        tokens = [
            token
            for token in re.findall(r"[0-9A-Za-zÀ-ỹĐđ]+", query.casefold())
            if len(token) > 1 and token not in self.STOPWORDS
        ]
        phrases: set[str] = set()
        for size in (2, 3, 4):
            for index in range(len(tokens) - size + 1):
                phrases.add(" ".join(tokens[index:index + size]))
        return phrases

    def _segment_relevance(
        self,
        query: str,
        segment: str,
        title: str = "",
        source_rank: int = 1,
    ) -> float:
        query_terms = self._query_terms(query)
        segment_terms = self._query_terms(segment)
        title_terms = self._query_terms(title)

        term_recall = len(query_terms & segment_terms) / max(len(query_terms), 1)
        title_recall = len(query_terms & title_terms) / max(len(query_terms), 1)

        normalized_segment = self._normalize_space(segment.casefold())
        phrases = self._query_phrases(query)
        phrase_hits = sum(1 for phrase in phrases if phrase in normalized_segment)
        phrase_score = min(phrase_hits / 3.0, 1.0)

        legal_signal = 0.0
        if re.search(
            r"\b(?:điều|khoản|điểm|đối tượng|điều kiện|mức|thời hạn|"
            r"thủ tục|hồ sơ|khoảng cách|trường hợp|quy định)\b",
            normalized_segment,
        ):
            legal_signal = 1.0

        rank_bonus = 1.0 / max(source_rank, 1)
        length_quality = min(len(segment), 320) / 320

        return (
            0.52 * term_recall
            + 0.16 * title_recall
            + 0.14 * phrase_score
            + 0.08 * legal_signal
            + 0.06 * rank_bonus
            + 0.04 * length_quality
        )

    def _rank_evidence_segments(
        self,
        query: str,
        results: list[RetrievalResult],
        limit: int = 8,
    ) -> list[tuple[float, int, str]]:
        candidates: list[tuple[float, int, str]] = []

        for source_id, item in enumerate(results, start=1):
            metadata = item.metadata or {}
            title = " ".join(
                [
                    str(metadata.get("title", "")),
                    str(metadata.get("articles", metadata.get("article", ""))),
                ]
            )
            segments = self._split_sentences(str(item.text or ""))

            if not segments:
                content = self._normalize_space(str(item.text or ""))
                if content:
                    segments = [content[:600]]

            for segment in segments:
                score = self._segment_relevance(
                    query=query,
                    segment=segment,
                    title=title,
                    source_rank=source_id,
                )
                candidates.append((score, source_id, segment))

        candidates.sort(key=lambda row: row[0], reverse=True)

        selected: list[tuple[float, int, str]] = []
        seen: set[str] = set()
        per_source: dict[int, int] = {}
        for score, source_id, segment in candidates:
            normalized = self._normalize_space(segment.casefold())
            if not normalized or normalized in seen:
                continue
            if per_source.get(source_id, 0) >= 3:
                continue
            seen.add(normalized)
            per_source[source_id] = per_source.get(source_id, 0) + 1
            selected.append((score, source_id, self._normalize_space(segment)))
            if len(selected) >= limit:
                break

        return selected

    def _retrieval_has_evidence(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> bool:
        ranked = self._rank_evidence_segments(query, results, limit=3)
        if not ranked:
            return False
        return ranked[0][0] >= 0.16

    @staticmethod
    def _contains_refusal(answer: str) -> bool:
        normalized = re.sub(r"\s+", " ", answer.casefold()).strip()
        refusal_patterns = (
            "không có câu trả lời",
            "không có đáp án",
            "không tìm thấy câu trả lời",
            "không tìm thấy thông tin",
            "không đủ thông tin",
            "chưa đủ thông tin",
            "không đủ căn cứ",
            "chưa đủ căn cứ",
            "không thể trả lời",
            "không thể xác định",
            "tài liệu không đề cập",
            "nguồn không đề cập",
        )
        return any(pattern in normalized for pattern in refusal_patterns)

    def _best_evidence_sentence(self, query: str, text: str) -> str:
        temporary = RetrievalResult(
            chunk_id="temporary",
            score=0.0,
            text=text,
            source="temporary",
            metadata={},
        )
        ranked = self._rank_evidence_segments(query, [temporary], limit=1)
        if not ranked:
            return self._normalize_space(text)[:360]
        best = ranked[0][2]
        if len(best) > 360:
            best = best[:357].rstrip() + "..."
        return best

    def _build_context(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> str:
        """Tạo context ngắn, tập trung vào các đoạn liên quan nhất."""
        ranked = self._rank_evidence_segments(query, results, limit=10)
        grouped: dict[int, list[str]] = {}
        for _, source_id, segment in ranked:
            grouped.setdefault(source_id, []).append(segment)

        sections: list[str] = []
        total_chars = 0

        for source_id, item in enumerate(results, start=1):
            metadata = item.metadata or {}
            document_number = self._clean_value(
                metadata.get("document_number")
            ) or "Không rõ số hiệu"
            title = self._clean_value(metadata.get("title")) or "Không rõ tiêu đề"
            article = self._clean_value(
                metadata.get("articles", metadata.get("article"))
            ) or "Không rõ điều/mục"

            evidence = grouped.get(source_id, [])[:3]
            if not evidence:
                fallback = self._best_evidence_sentence(query, str(item.text or ""))
                evidence = [fallback] if fallback else []
            if not evidence:
                continue

            evidence_lines = [
                f"Đoạn {index}: {segment[:900]}"
                for index, segment in enumerate(evidence, start=1)
            ]
            section = "\n".join(
                [
                    f'<SOURCE id="{source_id}">',
                    f"Số hiệu: {document_number}",
                    f"Tiêu đề: {title}",
                    f"Điều/Mục: {article}",
                    *evidence_lines,
                    "</SOURCE>",
                ]
            )

            if total_chars + len(section) > self.max_context_chars and sections:
                break
            sections.append(section)
            total_chars += len(section)

        return "\n\n".join(sections)

    def _build_user_prompt(
        self,
        query: str,
        context: str,
        retry: bool = False,
    ) -> str:
        retry_text = f"{self.RETRY_PROMPT}\n\n" if retry else ""

        return f"""
{retry_text}<LEGAL_SOURCES>
{context}
</LEGAL_SOURCES>

<USER_QUESTION>
{query}
</USER_QUESTION>

Hãy trả lời câu hỏi bằng cách tổng hợp các quy định liên quan trong nguồn.
Không sao chép nguyên văn nguồn và không tạo danh sách tài liệu tham khảo.
""".strip()

    def _ensure_generator_loaded(self) -> None:
        if self.generator.pipeline is None:
            self.generator.load()

    def _encode_messages(
        self,
        messages: list[dict[str, str]],
    ) -> dict[str, torch.Tensor]:
        self._ensure_generator_loaded()
        tokenizer = self.generator.pipeline.tokenizer

        if getattr(tokenizer, "chat_template", None):
            try:
                encoded = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                    enable_thinking=False,
                )
            except TypeError:
                encoded = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                )

            if isinstance(encoded, torch.Tensor):
                return {
                    "input_ids": encoded,
                    "attention_mask": torch.ones_like(encoded),
                }
            return dict(encoded)

        plain_prompt = (
            f"SYSTEM:\n{messages[0]['content']}\n\n"
            f"USER:\n{messages[1]['content']}\n\nASSISTANT:\n"
        )
        return dict(tokenizer(plain_prompt, return_tensors="pt"))

    def _generate_once(
        self,
        query: str,
        context: str,
        retry: bool = False,
    ) -> str:
        self._ensure_generator_loaded()
        generation_pipeline = self.generator.pipeline
        model = generation_pipeline.model
        tokenizer = generation_pipeline.tokenizer

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": self._build_user_prompt(query, context, retry=retry),
            },
        ]

        inputs = self._encode_messages(messages)
        input_device = next(model.parameters()).device
        inputs = {key: value.to(input_device) for key, value in inputs.items()}
        input_length = inputs["input_ids"].shape[1]

        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id

        eos_token_ids: list[int] = []
        if tokenizer.eos_token_id is not None:
            if isinstance(tokenizer.eos_token_id, list):
                eos_token_ids.extend(tokenizer.eos_token_id)
            else:
                eos_token_ids.append(int(tokenizer.eos_token_id))

        for special_token in ("<|im_end|>", "<|endoftext|>"):
            token_id = tokenizer.convert_tokens_to_ids(special_token)
            if isinstance(token_id, int) and token_id >= 0:
                if token_id not in eos_token_ids:
                    eos_token_ids.append(token_id)

        # Chỉ truyền một GenerationConfig hoàn chỉnh. Cách này loại bỏ cảnh báo
        # generation_config + generation kwargs và max_new_tokens + max_length.
        generation_config = deepcopy(model.generation_config)
        generation_config.max_new_tokens = None
        generation_config.max_length = input_length + self.max_new_tokens
        generation_config.do_sample = False
        generation_config.temperature = None
        generation_config.top_p = None
        generation_config.top_k = None
        generation_config.typical_p = None
        generation_config.repetition_penalty = 1.08
        generation_config.no_repeat_ngram_size = 5
        generation_config.renormalize_logits = True
        generation_config.pad_token_id = pad_token_id
        generation_config.eos_token_id = (
            eos_token_ids if len(eos_token_ids) > 1
            else eos_token_ids[0] if eos_token_ids
            else tokenizer.eos_token_id
        )

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                generation_config=generation_config,
            )

        generated = outputs[0, input_length:]
        answer = tokenizer.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return self._clean_generated_answer(answer)

    def _clean_generated_answer(self, answer: str) -> str:
        answer = re.sub(
            r"<think>.*?</think>",
            "",
            answer,
            flags=re.IGNORECASE | re.DOTALL,
        )
        answer = re.sub(
            r"<(?:analysis|reasoning)>.*?</(?:analysis|reasoning)>",
            "",
            answer,
            flags=re.IGNORECASE | re.DOTALL,
        )

        # Bỏ phần prompt nếu model vô tình lặp lại chỉ dẫn hoặc context.
        leakage_markers = (
            "<LEGAL_SOURCES>",
            "</LEGAL_SOURCES>",
            "<USER_QUESTION>",
            "</USER_QUESTION>",
            "Nếu không có đáp án nào phù hợp",
        )
        for marker in leakage_markers:
            if marker.casefold() in answer.casefold():
                position = answer.casefold().find(marker.casefold())
                if position == 0:
                    answer = answer[position + len(marker):]

        answer = answer.strip()
        answer = re.sub(r"^\s*\[(?:n|\d+)\]\s*[.::\-]*\s*", "", answer, flags=re.I)
        answer = re.sub(r"\n{3,}", "\n\n", answer)

        # Cắt vòng lặp dòng hoặc vòng lặp khối hai dòng.
        raw_lines = [line.rstrip() for line in answer.splitlines()]
        kept: list[str] = []
        normalized_kept: list[str] = []

        for line in raw_lines:
            normalized = self._normalize_space(line.casefold())

            if normalized and normalized_kept:
                if normalized == normalized_kept[-1]:
                    continue

                if len(normalized_kept) >= 2 and len(kept) >= 2:
                    previous_pair = normalized_kept[-2:]
                    candidate_pair = [normalized_kept[-1], normalized]
                    if candidate_pair == previous_pair:
                        break

                # Dừng khi một câu có nội dung được lặp lại lần thứ ba.
                if normalized_kept.count(normalized) >= 2:
                    break

            kept.append(line)
            normalized_kept.append(normalized)

        answer = "\n".join(kept).strip()

        # Cắt chuỗi lặp dài theo đoạn văn.
        paragraphs = [
            self._normalize_space(part)
            for part in re.split(r"\n\s*\n", answer)
            if self._normalize_space(part)
        ]
        unique_paragraphs: list[str] = []
        seen: set[str] = set()
        for paragraph in paragraphs:
            key = paragraph.casefold()
            if key in seen:
                break
            seen.add(key)
            unique_paragraphs.append(paragraph)

        return "\n\n".join(unique_paragraphs).strip()

    def _copy_ratio(
        self,
        answer: str,
        results: list[RetrievalResult],
    ) -> float:
        normalized_answer = self._normalize_space(answer.casefold())[:4000]
        if not normalized_answer:
            return 0.0

        ratios = []
        for item in results:
            chunk = self._normalize_space(str(item.text or "").casefold())[:4000]
            if chunk:
                ratios.append(
                    SequenceMatcher(None, normalized_answer, chunk).ratio()
                )
        return max(ratios, default=0.0)

    def _has_degenerate_repetition(self, answer: str) -> bool:
        lines = [
            self._normalize_space(line.casefold())
            for line in answer.splitlines()
            if self._normalize_space(line)
        ]
        if not lines:
            return True

        counts: dict[str, int] = {}
        for line in lines:
            counts[line] = counts.get(line, 0) + 1
            if counts[line] >= 3:
                return True

        words = re.findall(r"[0-9A-Za-zÀ-ỹĐđ]+", answer.casefold())
        if len(words) >= 40:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.23:
                return True

        return False

    def _answer_is_valid(
        self,
        query: str,
        answer: str,
        results: list[RetrievalResult],
    ) -> bool:
        if not answer or len(answer.split()) < 16:
            return False

        lowered = answer.casefold()
        forbidden_fragments = (
            "<think>",
            "[n]",
            "<legal_sources>",
            "<user_question>",
            "nếu không có đáp án nào phù hợp",
        )
        if any(fragment in lowered for fragment in forbidden_fragments):
            return False

        if self._contains_refusal(answer) and self._retrieval_has_evidence(
            query,
            results,
        ):
            return False

        if self._has_degenerate_repetition(answer):
            return False

        first_line = answer.splitlines()[0].strip()
        if re.match(r"^(?:\[(?:n|\d+)\]|Điều\s+\d+)", first_line, flags=re.I):
            return False

        if self._copy_ratio(answer, results) >= 0.86:
            return False

        if len(answer.split()) > 450:
            return False

        return True

    def _strip_model_source_section(self, answer: str) -> str:
        patterns = [
            r"\n\s*(?:Nguồn|Nguồn trích dẫn|Tài liệu tham khảo)\s*:\s*\n.*$",
            r"\n\s*#{1,4}\s*(?:Nguồn|Nguồn trích dẫn|Tài liệu tham khảo).*?$",
        ]
        cleaned = answer
        for pattern in patterns:
            cleaned = re.sub(
                pattern,
                "",
                cleaned,
                flags=re.IGNORECASE | re.DOTALL,
            )
        return cleaned.strip()

    def _referenced_source_ids(
        self,
        answer: str,
        results: list[RetrievalResult],
    ) -> list[int]:
        """Chọn nguồn có nội dung gần nhất với answer; không phụ thuộc model."""

        answer_terms = self._query_terms(answer)
        scored: list[tuple[float, int]] = []

        for source_id, item in enumerate(results, start=1):
            metadata = item.metadata or {}
            searchable = " ".join(
                [
                    str(metadata.get("title", "")),
                    str(metadata.get("articles", metadata.get("article", ""))),
                    str(item.text or ""),
                ]
            )
            source_terms = self._query_terms(searchable)
            overlap = len(answer_terms & source_terms) / max(len(answer_terms), 1)
            retrieval_bonus = 1.0 / source_id
            score = 0.85 * overlap + 0.15 * retrieval_bonus
            scored.append((score, source_id))

        scored.sort(reverse=True)
        selected = [source_id for score, source_id in scored if score > 0][:3]
        return selected or list(range(1, min(len(results), 2) + 1))

    def _build_source_section(
        self,
        query: str,
        answer: str,
        results: list[RetrievalResult],
    ) -> str:
        lines = ["Nguồn trích dẫn:"]

        for source_id in self._referenced_source_ids(answer, results):
            item = results[source_id - 1]
            metadata = item.metadata or {}

            document_number = self._clean_value(
                metadata.get("document_number")
            ) or "Không rõ số hiệu"
            title = self._clean_value(metadata.get("title")) or "Không rõ tiêu đề"
            article = self._clean_value(
                metadata.get("articles", metadata.get("article"))
            ) or "Không rõ điều/mục"
            url = self._clean_value(metadata.get("url"))
            quote = self._best_evidence_sentence(query, str(item.text or ""))

            lines.append(
                f"- [Nguồn {source_id}] {document_number} — {article}."
            )
            lines.append(f"  Trích đoạn: “{quote}”")
            if title:
                lines.append(f"  Văn bản: {title}")
            if url:
                lines.append(f"  URL: {url}")

        return "\n".join(lines)

    def _extractive_fallback(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> str:
        """Fallback luôn trả lại phần thông tin tốt nhất nếu có chunk."""
        ranked = self._rank_evidence_segments(query, results, limit=7)

        if not ranked:
            for source_id, item in enumerate(results, start=1):
                content = self._normalize_space(str(item.text or ""))
                if content:
                    ranked.append((0.0, source_id, content[:650]))
                    break

        if not ranked:
            return (
                "Kết luận:\nKhông có nội dung văn bản để tổng hợp.\n\n"
                "Thông tin chính:\n- Kết quả retrieval không chứa nội dung chunk."
            )

        selected: list[tuple[int, str]] = []
        seen: set[str] = set()
        for _, source_id, segment in ranked:
            segment = self._normalize_space(segment)
            key = segment.casefold()
            if not segment or key in seen:
                continue
            seen.add(key)
            if len(segment) > 360:
                segment = segment[:357].rstrip() + "..."
            selected.append((source_id, segment))
            if len(selected) >= 4:
                break

        _, first_segment = selected[0]
        lines = [
            "Kết luận:",
            f"Dựa trên tài liệu được truy xuất, {first_segment.rstrip('.')}.",
            "",
            "Thông tin chính:",
        ]

        for _, segment in selected[1:]:
            lines.append(f"- {segment.rstrip('.')}.")

        if len(selected) == 1:
            lines.append(
                "- Phạm vi áp dụng cần được đối chiếu với số hiệu, địa phương "
                "và thời kỳ của văn bản trong phần nguồn trích dẫn."
            )

        return "\n".join(lines)

    def generate_answer(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> str:
        usable_results = [
            item for item in results if str(item.text or "").strip()
        ]

        if not usable_results:
            return (
                "Không thể sinh câu trả lời vì các kết quả retrieval "
                "không có nội dung chunk."
            )

        context = self._build_context(query, usable_results)
        answer = self._generate_once(query, context, retry=False)

        if not self._answer_is_valid(query, answer, usable_results):
            self.logger.warning(
                "Answer lần đầu chỉ chép context, thiếu nguồn hoặc thiếu phần "
                "trả lời. Đang sinh lại một lần."
            )
            answer = self._generate_once(query, context, retry=True)

        if not self._answer_is_valid(query, answer, usable_results):
            self.logger.warning(
                "Answer lần hai vẫn không đạt; sử dụng fallback trích xuất "
                "để bảo đảm có câu trả lời và nguồn."
            )
            answer = self._extractive_fallback(query, usable_results)

        if self._contains_refusal(answer) and self._retrieval_has_evidence(
            query,
            usable_results,
        ):
            answer = self._extractive_fallback(query, usable_results)

        answer = self._strip_model_source_section(answer)
        source_section = self._build_source_section(
            query=query,
            answer=answer,
            results=usable_results,
        )

        return f"{answer}\n\n{source_section}".strip()

    # ------------------------------------------------------------------
    # INTERACTIVE
    # ------------------------------------------------------------------

    def ask(self, query: str) -> str:
        results = self.retrieve(query)
        self.print_results(query, results)

        print("\nĐang sinh câu trả lời từ các chunk phía trên...\n")
        answer = self.generate_answer(query, results)

        print("=" * 110)
        print("CÂU TRẢ LỜI")
        print("=" * 110)
        print(answer)
        print("=" * 110)
        return answer

    def interactive(self) -> None:
        print("\n" + "=" * 110)
        print("LEGAL RAG INTERACTIVE DEMO")
        print(f"Demo version     : {DEMO_VERSION}")
        print("=" * 110)
        print(f"Retrieval method : {self.method}")
        print(f"Final top-k      : {self.top_k}")
        print("Nhập 'exit', 'quit' hoặc 'q' để thoát.")
        print("=" * 110)

        while True:
            try:
                query = input("\nNhập câu hỏi pháp luật: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nĐã thoát demo.")
                break

            if query.casefold() in {"exit", "quit", "q"}:
                print("Đã thoát demo.")
                break

            if not query:
                print("Query không được để trống.")
                continue

            try:
                self.ask(query)
            except Exception as error:
                self.logger.exception("Không thể xử lý query.")
                print(f"\nLỗi: {error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive LegalRAG demo: retrieval top chunks, sinh câu trả lời "
            "tóm tắt và tự gắn nguồn trích dẫn."
        )
    )
    parser.add_argument("--config", default="config.json")
    parser.add_argument(
        "--method",
        choices=SUPPORTED_METHODS,
        default="hybrid+graph",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--query", default=None)
    parser.add_argument("--truncate", action="store_true")
    parser.add_argument("--no-auto-enable-generation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        demo = LegalRAGDemo(
            config_path=args.config,
            method=args.method,
            top_k=args.top_k,
            show_full_text=not args.truncate,
            auto_enable_generation=not args.no_auto_enable_generation,
        ).load()

        if args.query:
            demo.ask(args.query)
        else:
            demo.interactive()

    except Exception as error:
        logging.getLogger("LegalRAGDemo").exception(
            "Demo khởi động thất bại."
        )
        print(f"\nKhông thể chạy demo: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
