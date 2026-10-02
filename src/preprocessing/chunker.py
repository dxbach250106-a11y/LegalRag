from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

LOGGER = logging.getLogger(__name__)


# ============================================================================
# PATTERN NHẬN DIỆN CẤU TRÚC VĂN BẢN
# ============================================================================

PART_PATTERN = re.compile(
    r"^\s*PHẦN(?:\s+THỨ)?\s+"
    r"(?P<number>[^:.\-\n]+?)"
    r"(?:\s*[:.\-]\s*(?P<title>.+))?\s*$",
    re.IGNORECASE,
)

CHAPTER_PATTERN = re.compile(
    r"^\s*CHƯƠNG\s+"
    r"(?P<number>[IVXLCDM]+|\d+[A-Za-z]?)"
    r"(?:\s*[:.\-]\s*(?P<title>.*))?\s*$",
    re.IGNORECASE,
)

SECTION_PATTERN = re.compile(
    r"^\s*MỤC\s+"
    r"(?P<number>[IVXLCDM]+|\d+[A-Za-z]?)"
    r"(?:\s*[:.\-]\s*(?P<title>.*))?\s*$",
    re.IGNORECASE,
)

ARTICLE_PATTERN = re.compile(
    r"^\s*Điều\s+"
    r"(?P<number>\d+[A-Za-z]?)"
    r"(?:\s*[.:]\s*(?P<title>.*))?\s*$",
    re.IGNORECASE,
)

FIRST_ARTICLE_PATTERN = re.compile(
    r"^\s*Điều\s+\d+[A-Za-z]?\s*[.:]?",
    re.IGNORECASE | re.MULTILINE,
)


# Nhận diện phần hành chính cuối văn bản để không gắn vào Điều cuối.
ADMIN_TAIL_PATTERN = re.compile(
    r"""
    ^\s*
    (?:
        Nơi\s+nhận\s*:
        | KT\.\s*[A-ZÀ-ỸĐ\s]+\b
        | TM\.\s*[A-ZÀ-ỸĐ\s]+\b
        | TUQ\.\s*[A-ZÀ-ỸĐ\s]+\b
        | TL\.\s*[A-ZÀ-ỸĐ\s]+\b
        | CHỦ\s+TỊCH\b
        | PHÓ\s+CHỦ\s+TỊCH\b
        | BỘ\s+TRƯỞNG\b
        | THỨ\s+TRƯỞNG\b
        | GIÁM\s+ĐỐC\b
        | PHÓ\s+GIÁM\s+ĐỐC\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Nhận diện phần phụ lục/danh mục thường xuất hiện sau nơi nhận và chữ ký.
APPENDIX_START_PATTERN = re.compile(
    r"""
    ^\s*
    (?:
        PHỤ\s+LỤC\b.*
        | DANH\s+MỤC\b.*
        | NỘI\s+DUNG\s+QUY\s+TRÌNH\b.*
        | QUY\s+TRÌNH\s+GIẢI\s+QUYẾT\b.*
        | \(?\s*Ban\s+hành\s+kèm\s+theo\b.*
    )
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Dùng khi tiêu đề phụ lục nằm giữa một dòng hành chính dài.
APPENDIX_INLINE_PATTERN = re.compile(
    r"\b(?:PHỤ\s+LỤC|DANH\s+MỤC|NỘI\s+DUNG\s+QUY\s+TRÌNH)\b",
    re.IGNORECASE,
)

# Nhận diện Nơi nhận nằm giữa dòng thay vì ở đầu dòng.
ADMIN_INLINE_PATTERN = re.compile(
    r"\bNơi\s+nhận\s*:",
    re.IGNORECASE,
)


# ============================================================================
# PATTERN NHẬN DIỆN TRÍCH DẪN PHÁP LÝ
# ============================================================================

# Nhận các số hiệu như:
#   66/2025/NĐ-CP
#   34/2025/QĐ-UBND
#   1115/QĐ-UBND
DOCUMENT_NUMBER_PATTERN = (
    r"\d+"
    r"(?:/\d{4})?"
    r"/[A-ZĐ0-9]+"
    r"(?:-[A-ZĐ0-9]+)*"
)

DOCUMENT_TYPE_PATTERN = (
    r"Bộ\s+luật"
    r"|Luật"
    r"|Nghị\s+định"
    r"|Thông\s+tư"
    r"|Nghị\s+quyết"
    r"|Quyết\s+định"
    r"|Chỉ\s+thị"
)

# Ví dụ:
#   điểm b khoản 4 Điều 14 Nghị định số 66/2025/NĐ-CP
#   khoản 2, khoản 3 Điều 4 Nghị định số 66/2025/NĐ-CP
#   Quyết định số 1115/QĐ-UBND
EXTERNAL_REFERENCE_PATTERN = re.compile(
    rf"""
    (?P<structure>
        (?:
            điểm\s+
            (?P<point>[a-zđ])
            \s+
        )?

        (?:
            (?P<clauses>
                khoản\s+\d+[a-z]?
                (?:
                    \s*,?\s*
                    (?:và\s+)?
                    khoản\s+\d+[a-z]?
                )*
            )
            \s+
        )?

        (?:
            Điều\s+
            (?P<article>\d+[a-z]?)
            \s+
        )?
    )?

    (?P<document_type>
        {DOCUMENT_TYPE_PATTERN}
    )

    \s+
    (?:số\s+)?

    (?P<document_number>
        {DOCUMENT_NUMBER_PATTERN}
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Ví dụ:
#   điểm b khoản 4 Điều 14
#   khoản 2, khoản 3 Điều 4
#   Điều 2
INTERNAL_REFERENCE_PATTERN = re.compile(
    r"""
    (?:
        điểm\s+
        (?P<point>[a-zđ])
        \s+
    )?

    (?:
        (?P<clauses>
            khoản\s+\d+[a-z]?
            (?:
                \s*,?\s*
                (?:và\s+)?
                khoản\s+\d+[a-z]?
            )*
        )
        \s+
    )?

    Điều\s+
    (?P<article>\d+[a-z]?)
    """,
    re.IGNORECASE | re.VERBOSE,
)

ARTICLE_ONLY_PATTERN = re.compile(
    r"\bĐiều\s+(?P<article>\d+[a-z]?)\b",
    re.IGNORECASE,
)

# Ví dụ:
#   Mục I Phụ lục II
#   Mục 2 Phụ lục 3
APPENDIX_REFERENCE_PATTERN = re.compile(
    r"""
    \b
    Mục\s+
    (?P<section>[IVXLCDM]+|\d+)
    \s+
    Phụ\s+lục\s+
    (?P<appendix>[IVXLCDM]+|\d+)
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

REPEAL_KEYWORD_PATTERN = re.compile(
    r"\b("
    r"hết\s+hiệu\s+lực"
    r"|bãi\s+bỏ"
    r"|chấm\s+dứt\s+hiệu\s+lực"
    r")\b",
    re.IGNORECASE,
)

AMEND_KEYWORD_PATTERN = re.compile(
    r"\b("
    r"sửa\s+đổi"
    r"|bổ\s+sung"
    r"|thay\s+thế"
    r")\b",
    re.IGNORECASE,
)


# ============================================================================
# HÀM TIỆN ÍCH
# ============================================================================

def normalize_text(text: str) -> str:
    """
    Chuẩn hóa Unicode và khoảng trắng nhưng vẫn giữ xuống dòng.

    Xuống dòng được giữ ở bước này để parser có thể nhận diện chính xác
    Phần, Chương, Mục và Điều.
    """

    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = text.replace("\xa0", " ")

    lines: list[str] = []

    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(line)

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def flatten_chunk_text(text: str) -> str:
    """
    Làm phẳng nội dung trước khi lưu và embedding.

    Tất cả ký tự xuống dòng, tab và chuỗi khoảng trắng liên tiếp được đổi
    thành đúng một dấu cách.
    """

    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = text.replace("\xa0", " ")

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    return text.strip()


def clean_table_text(text: str) -> str:
    """
    Chuyển dữ liệu bảng dùng dấu | thành văn bản dễ đọc và dễ embedding.

    Mỗi hàng bảng được giữ riêng về mặt ngữ nghĩa, còn các ô trong hàng
    được nối bằng dấu chấm phẩy thay vì để ký tự | gây nhiễu.
    """

    if not isinstance(text, str):
        return ""

    cleaned_rows: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        if "|" not in line:
            cleaned_rows.append(
                flatten_chunk_text(line)
            )
            continue

        cells = [
            re.sub(r"\s+", " ", cell).strip()
            for cell in re.split(r"\|+", line)
        ]

        cells = [
            cell
            for cell in cells
            if cell
            and not re.fullmatch(r"[-:\s]+", cell)
        ]

        if cells:
            cleaned_rows.append(
                "; ".join(cells)
            )

    result = ". ".join(cleaned_rows)
    result = re.sub(r"\s+", " ", result)
    result = re.sub(r"(?:;\s*){2,}", "; ", result)
    result = re.sub(r"\s+([,.;:!?])", r"\1", result)
    result = re.sub(r"\.{2,}", ".", result)

    return result.strip(" .")


def split_long_text(
    text: str,
    max_chars: int = 3000,
) -> list[str]:
    """Chia văn bản dài theo câu mà không cắt giữa từ."""

    text = flatten_chunk_text(text)

    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    sentences = re.split(
        r"(?<=[.;!?])\s+",
        text,
    )

    chunks: list[str] = []
    buffer = ""

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        candidate = (
            f"{buffer} {sentence}".strip()
            if buffer
            else sentence
        )

        if len(candidate) <= max_chars:
            buffer = candidate
            continue

        if buffer:
            chunks.append(buffer)

        if len(sentence) <= max_chars:
            buffer = sentence
            continue

        # Nếu một câu hoặc một hàng bảng quá dài, cắt tại khoảng trắng
        # gần giới hạn max_chars thay vì cắt giữa từ.
        remaining = sentence

        while len(remaining) > max_chars:
            split_position = remaining.rfind(
                " ",
                0,
                max_chars + 1,
            )

            if split_position <= 0:
                split_position = max_chars

            chunks.append(
                remaining[:split_position].strip()
            )
            remaining = remaining[split_position:].strip()

        buffer = remaining

    if buffer:
        chunks.append(buffer)

    return chunks


def split_appendix_blocks(
    lines: list[str],
) -> list[tuple[str, str]]:
    """
    Tách phụ lục thành các khối văn bản thường và khối bảng.

    Kết quả gồm các tuple:
        ("text", nội dung)
        ("table", nội dung bảng đã làm sạch)
    """

    blocks: list[tuple[str, str]] = []
    current_kind: str | None = None
    current_lines: list[str] = []

    def flush_block() -> None:
        nonlocal current_kind
        nonlocal current_lines

        if not current_lines or current_kind is None:
            current_kind = None
            current_lines = []
            return

        raw_block = "\n".join(current_lines).strip()

        if current_kind == "table":
            cleaned_block = clean_table_text(raw_block)
        else:
            cleaned_block = flatten_chunk_text(raw_block)

        if cleaned_block:
            blocks.append(
                (current_kind, cleaned_block)
            )

        current_kind = None
        current_lines = []

    for line in lines:
        stripped_line = line.strip()

        if not stripped_line:
            continue

        line_kind = (
            "table"
            if stripped_line.count("|") >= 2
            else "text"
        )

        if (
            current_kind is not None
            and line_kind != current_kind
        ):
            flush_block()

        current_kind = line_kind
        current_lines.append(stripped_line)

    flush_block()

    return blocks


def get_data_split(dataset: Dataset | DatasetDict) -> Dataset:
    """Lấy Dataset trực tiếp hoặc split 'data' từ DatasetDict."""

    if isinstance(dataset, Dataset):
        return dataset

    if isinstance(dataset, DatasetDict):
        if "data" not in dataset:
            raise KeyError(
                "DatasetDict không có split 'data'. "
                f"Các split hiện có: {list(dataset.keys())}"
            )

        return dataset["data"]

    raise TypeError("Dữ liệu phải có kiểu Dataset hoặc DatasetDict.")


def normalize_sector(value: Any) -> str:
    """Chuẩn hóa tên lĩnh vực để so sánh."""

    return re.sub(
        r"\s+",
        " ",
        str(value).strip().casefold(),
    )


def sector_matches(legal_sectors: Any, target_legal_sector: str,) -> bool:
    """Kiểm tra target có thuộc legal_sectors hay không."""

    target = normalize_sector(target_legal_sector)

    if not target:
        return False

    if isinstance(legal_sectors, list):
        sectors = [
            normalize_sector(sector)
            for sector in legal_sectors
        ]

        return any(
            target == sector or target in sector
            for sector in sectors
        )

    sector_text = normalize_sector(legal_sectors)

    return target == sector_text or target in sector_text


def normalize_document_number(
    document_number: Any,
) -> str | None:
    """Chuẩn hóa số hiệu văn bản để so sánh và dựng graph."""

    if document_number is None:
        return None

    value = str(document_number).upper().strip()
    value = value.replace("–", "-")
    value = value.replace("—", "-")

    value = re.sub(r"\s*/\s*", "/", value)
    value = re.sub(r"\s*-\s*", "-", value)
    value = re.sub(r"\s+", "", value)

    return value or None


def make_safe_id(value: Any) -> str:
    """Chuyển một giá trị thành thành phần ID an toàn."""

    value = str(value)
    value = unicodedata.normalize("NFKD", value)

    value = "".join(
        character
        for character in value
        if not unicodedata.combining(character)
    )

    value = re.sub(r"[^A-Za-z0-9]+", "_", value)

    return value.strip("_").lower()


def looks_like_title(line: str) -> bool:
    """
    Kiểm tra một dòng có khả năng là tiêu đề nằm sau Phần/Chương/Mục.

    Ví dụ:
        CHƯƠNG I
        QUY ĐỊNH CHUNG
    """

    if not line or len(line) > 250:
        return False

    letters = [
        character
        for character in line
        if character.isalpha()
    ]

    if not letters:
        return False

    uppercase_ratio = sum(
        character.isupper()
        for character in letters
    ) / len(letters)

    return uppercase_ratio >= 0.7


def overlaps(
    start: int,
    end: int,
    spans: list[tuple[int, int]],
) -> bool:
    """Kiểm tra một span có chồng lên span đã xử lý hay không."""

    return any(
        start < used_end and end > used_start
        for used_start, used_end in spans
    )


def get_clause_numbers(clauses_text: str | None) -> list[str]:
    """Lấy tất cả số Khoản từ một chuỗi trích dẫn."""

    if not clauses_text:
        return []

    return [
        value.lower()
        for value in re.findall(
            r"khoản\s+(\d+[a-z]?)",
            clauses_text,
            flags=re.IGNORECASE,
        )
    ]


def detect_document_relationships(text: str) -> dict[str, set[str]]:
    """
    Xác định các văn bản có quan hệ bãi bỏ hoặc sửa đổi trong đoạn text.

    Đây là rule-based extraction nên kết quả phù hợp để tạo graph sơ bộ,
    sau đó vẫn có thể bổ sung bước kiểm tra/rerank.
    """

    relationships: dict[str, set[str]] = {
        "REPEALS": set(),
        "AMENDS": set(),
    }

    # Chia gần đúng theo câu để tránh gán quan hệ cho văn bản ở quá xa.
    sentences = re.split(r"(?<=[.;])\s+", flatten_chunk_text(text))

    for sentence in sentences:
        document_matches = list(
            EXTERNAL_REFERENCE_PATTERN.finditer(sentence)
        )

        if not document_matches:
            continue

        document_numbers = [
            normalize_document_number(
                match.group("document_number")
            )
            for match in document_matches
        ]

        document_numbers = [
            number
            for number in document_numbers
            if number is not None
        ]

        if not document_numbers:
            continue

        if REPEAL_KEYWORD_PATTERN.search(sentence):
            # Trong cấu trúc thông thường:
            # "Quyết định số X ... hết hiệu lực"
            # văn bản đầu tiên là văn bản bị bãi bỏ.
            relationships["REPEALS"].add(document_numbers[0])

        if AMEND_KEYWORD_PATTERN.search(sentence):
            relationships["AMENDS"].add(document_numbers[0])

    return relationships


def extract_link_to(
    text: str,
    current_document_number: str | None,
    scope: str = "article",
) -> list[dict]:
    """
    Trích xuất liên kết pháp lý phục vụ dựng graph.

    scope:
        - document_header: phần mở đầu trước Điều đầu tiên
        - article: nội dung của một Điều
    """

    links: list[dict] = []
    used_spans: list[tuple[int, int]] = []
    seen: set[tuple[Any, ...]] = set()

    text = flatten_chunk_text(text)

    current_document_number = normalize_document_number(
        current_document_number
    )

    special_relationships = detect_document_relationships(text)

    def add_link(
        relationship: str,
        document_number: str | None,
        article: str | None = None,
        clause: str | None = None,
        point: str | None = None,
        appendix: str | None = None,
        section: str | None = None,
        raw_text: str = "",
    ) -> None:
        """Chuẩn hóa, loại trùng rồi thêm một liên kết."""

        normalized_document_number = normalize_document_number(
            document_number
        )

        article_value = (
            str(article).lower()
            if article is not None
            else None
        )

        clause_value = (
            str(clause).lower()
            if clause is not None
            else None
        )

        point_value = (
            str(point).lower()
            if point is not None
            else None
        )

        appendix_value = (
            str(appendix).upper()
            if appendix is not None
            else None
        )

        section_value = (
            str(section).upper()
            if section is not None
            else None
        )

        # Loại self-link chỉ trỏ lại chính văn bản nhưng không chỉ rõ
        # Điều/Khoản/Điểm/Phụ lục nào.
        empty_self_link = (
            normalized_document_number == current_document_number
            and article_value is None
            and clause_value is None
            and point_value is None
            and appendix_value is None
            and section_value is None
        )

        if empty_self_link:
            return

        key = (
            relationship,
            normalized_document_number,
            article_value,
            clause_value,
            point_value,
            appendix_value,
            section_value,
            scope,
        )

        if key in seen:
            return

        seen.add(key)

        links.append(
            {
                "relationship": relationship,
                "document_number": normalized_document_number,
                "article": article_value,
                "clause": clause_value,
                "point": point_value,
                "appendix": appendix_value,
                "section": section_value,
                "raw_text": flatten_chunk_text(raw_text),
                "scope": scope,
            }
        )

    # ------------------------------------------------------------------------
    # A. Trích dẫn tới văn bản có số hiệu cụ thể
    # ------------------------------------------------------------------------

    for match in EXTERNAL_REFERENCE_PATTERN.finditer(text):
        document_number = normalize_document_number(
            match.group("document_number")
        )

        if document_number in special_relationships["REPEALS"]:
            relationship = "REPEALS"

        elif document_number in special_relationships["AMENDS"]:
            relationship = "AMENDS"

        else:
            prefix_start = max(0, match.start() - 150)
            prefix_text = text[prefix_start:match.start()]

            if re.search(r"\bcăn\s+cứ\b", prefix_text, re.IGNORECASE):
                relationship = "BASED_ON"
            else:
                relationship = "REFERS_TO"

        article = match.group("article")
        point = match.group("point")
        clauses = get_clause_numbers(
            match.group("clauses")
        )

        used_spans.append(
            (match.start(), match.end())
        )

        if clauses:
            for clause in clauses:
                add_link(
                    relationship=relationship,
                    document_number=document_number,
                    article=article,
                    clause=clause,
                    point=point,
                    raw_text=match.group(0),
                )
        else:
            add_link(
                relationship=relationship,
                document_number=document_number,
                article=article,
                point=point,
                raw_text=match.group(0),
            )

    # ------------------------------------------------------------------------
    # B. Trích dẫn nội bộ tới Điều/Khoản/Điểm
    # ------------------------------------------------------------------------

    for match in INTERNAL_REFERENCE_PATTERN.finditer(text):
        if overlaps(
            match.start(),
            match.end(),
            used_spans,
        ):
            continue

        clauses = get_clause_numbers(
            match.group("clauses")
        )

        if clauses:
            for clause in clauses:
                add_link(
                    relationship="REFERS_TO",
                    document_number=current_document_number,
                    article=match.group("article"),
                    clause=clause,
                    point=match.group("point"),
                    raw_text=match.group(0),
                )
        else:
            add_link(
                relationship="REFERS_TO",
                document_number=current_document_number,
                article=match.group("article"),
                point=match.group("point"),
                raw_text=match.group(0),
            )

        used_spans.append(
            (match.start(), match.end())
        )

    # ------------------------------------------------------------------------
    # C. Bắt thêm Điều còn lại trong danh sách
    #
    # Ví dụ:
    #   Điều 2 và Điều 3 Quyết định này
    # ------------------------------------------------------------------------

    for match in ARTICLE_ONLY_PATTERN.finditer(text):
        if overlaps(
            match.start(),
            match.end(),
            used_spans,
        ):
            continue

        add_link(
            relationship="REFERS_TO",
            document_number=current_document_number,
            article=match.group("article"),
            raw_text=match.group(0),
        )

        used_spans.append(
            (match.start(), match.end())
        )

    # ------------------------------------------------------------------------
    # D. Trích dẫn nội bộ tới Phụ lục
    # ------------------------------------------------------------------------

    for match in APPENDIX_REFERENCE_PATTERN.finditer(text):
        add_link(
            relationship="REFERS_TO",
            document_number=current_document_number,
            appendix=match.group("appendix"),
            section=match.group("section"),
            raw_text=match.group(0),
        )

    return links


def merge_links(
    first_links: list[dict],
    second_links: list[dict],
) -> list[dict]:
    """Gộp hai danh sách liên kết và loại liên kết trùng."""

    merged_links: list[dict] = []
    seen: set[tuple[Any, ...]] = set()

    for link in first_links + second_links:
        key = (
            link.get("relationship"),
            link.get("document_number"),
            link.get("article"),
            link.get("clause"),
            link.get("point"),
            link.get("appendix"),
            link.get("section"),
            link.get("scope"),
        )

        if key in seen:
            continue

        seen.add(key)
        merged_links.append(link)

    return merged_links


# ============================================================================
# HÀM CHUNK CHÍNH
# ============================================================================

def chunk_dataset(
    ds_content: Dataset | DatasetDict,
    ds_metadata: Dataset | DatasetDict,
    target_legal_sector: str,
    output_path: str = "./data/legal_chunks.parquet",
    min_chunk_chars: int = 50,
    print_all_chunks: bool = True,
) -> None:
    """
    Chia văn bản theo Phần -> Chương -> Mục -> Điều.

    Mỗi Điều là một chunk có schema:
        {
            id,
            document_id,
            document_number,
            title,
            url,
            legal_type,
            legal_sectors,
            issuing_authority,
            issuance_date,
            signers,
            part,
            chapter,
            section,
            articles,
            content,
            link_to
        }

    Trích dẫn ở phần đầu văn bản trước Điều đầu tiên vẫn được quét và gắn
    vào chunk Điều đầu tiên với scope='document_header'.
    """

    content_data = get_data_split(ds_content)
    metadata_data = get_data_split(ds_metadata)

    if len(content_data) != len(metadata_data):
        raise ValueError(
            "Số bản ghi content và metadata không bằng nhau: "
            f"{len(content_data)} != {len(metadata_data)}"
        )

    if not target_legal_sector.strip():
        LOGGER.error("target_legal_sector không được để trống.")
        return []

    target_exists = any(
        sector_matches(
            metadata_data[index].get("legal_sectors", ""),
            target_legal_sector,
        )
        for index in range(len(metadata_data))
    )

    if not target_exists:
        LOGGER.error(
            "Target legal sector '%s' không tồn tại trong dataset.",
            target_legal_sector,
        )
        return []

    chunk_list: list[dict] = []

    matched_document_count = 0
    skipped_document_count = 0

    for index in range(len(content_data)):
        content_item = content_data[index]
        metadata_item = metadata_data[index]

        if not sector_matches(
            metadata_item.get("legal_sectors", ""),
            target_legal_sector,
        ):
            skipped_document_count += 1
            continue

        matched_document_count += 1

        document_id = content_item.get(
            "id",
            metadata_item.get("id", index),
        )

        metadata_document_id = metadata_item.get("id")

        if (
            metadata_document_id is not None
            and str(metadata_document_id) != str(document_id)
        ):
            LOGGER.warning(
                "ID content và metadata khác nhau tại index %d: %s != %s",
                index,
                document_id,
                metadata_document_id,
            )

        current_document_number = normalize_document_number(
            metadata_item.get("document_number")
        )

        parser_content = normalize_text(
            content_item.get("content", "")
        )

        if not parser_content:
            LOGGER.warning(
                "Document %s không có content.",
                document_id,
            )
            continue

        # --------------------------------------------------------------------
        # Quét liên kết ở toàn bộ phần đầu trước Điều đầu tiên.
        # Không cắt bỏ phần này khỏi văn bản nguồn.
        # --------------------------------------------------------------------

        first_article_match = FIRST_ARTICLE_PATTERN.search(
            parser_content
        )

        if first_article_match:
            document_header_text = parser_content[
                :first_article_match.start()
            ]
        else:
            document_header_text = parser_content

        document_header_links = extract_link_to(
            text=document_header_text,
            current_document_number=current_document_number,
            scope="document_header",
        )

        # --------------------------------------------------------------------
        # Trạng thái cấu trúc hiện tại
        # --------------------------------------------------------------------

        current_part: str | None = None
        current_chapter: str | None = None
        current_section: str | None = None

        current_article_number: str | None = None
        current_article_title: str | None = None
        current_article_order = 0

        article_buffer: list[str] = []
        pending_title_level: str | None = None
        document_chunk_count = 0

        # Trạng thái dùng để bỏ phần Nơi nhận/chữ ký và tách phụ lục/bảng.
        inside_admin_tail = False
        inside_appendix = False
        appendix_title: str | None = None
        appendix_buffer: list[str] = []
        appendix_chunk_order = 0
        table_chunk_order = 0

        def flush_article() -> None:
            """Lưu Điều hiện tại thành một chunk."""

            nonlocal article_buffer
            nonlocal current_article_number
            nonlocal current_article_title
            nonlocal document_chunk_count

            if current_article_number is None:
                article_buffer = []
                return

            raw_article_content = "\n".join(
                article_buffer
            ).strip()

            if "|" in raw_article_content:
                article_content = clean_table_text(
                    raw_article_content
                )
            else:
                article_content = flatten_chunk_text(
                    raw_article_content
                )

            if len(article_content) < min_chunk_chars:
                article_buffer = []
                current_article_number = None
                current_article_title = None
                return

            if current_article_title:
                article_value = (
                    f"Điều {current_article_number}. "
                    f"{current_article_title}"
                )
            else:
                article_value = (
                    f"Điều {current_article_number}"
                )

            # Không dùng dòng tiêu đề Điều để tìm link vì nếu dùng,
            # chính Điều hiện tại sẽ bị nhận thành self-reference.
            article_body = flatten_chunk_text(
                " ".join(article_buffer[1:])
            )

            article_links = extract_link_to(
                text=article_body,
                current_document_number=current_document_number,
                scope="article",
            )

            # Chỉ gắn link phần mở đầu vào chunk đầu tiên của văn bản,
            # tránh lặp lại cùng một danh sách link ở mọi Điều.
            if document_chunk_count == 0:
                link_to = merge_links(
                    document_header_links,
                    article_links,
                )
            else:
                link_to = article_links

            chunk_id = (
                f"{make_safe_id(document_id)}"
                f"_dieu_{make_safe_id(current_article_number)}"
                f"_{current_article_order:04d}"
            )

            temp_token = {
                "id": chunk_id,
                "document_id": document_id,
                "document_number": current_document_number,
                "title": metadata_item.get("title"),
                "url": metadata_item.get("url"),
                "legal_type": metadata_item.get("legal_type"),
                "legal_sectors": metadata_item.get("legal_sectors"),
                "issuing_authority": metadata_item.get(
                    "issuing_authority"
                ),
                "issuance_date": metadata_item.get(
                    "issuance_date"
                ),
                "signers": metadata_item.get("signers"),
                "part": current_part,
                "chapter": current_chapter,
                "section": current_section,
                "articles": article_value,
                "content": article_content,
                "link_to": link_to,
            }

            chunk_list.append(temp_token)

            document_chunk_count += 1
            article_buffer = []
            current_article_number = None
            current_article_title = None

        def flush_appendix() -> None:
            """Tạo chunk riêng cho nội dung phụ lục và từng khối bảng."""

            nonlocal appendix_buffer
            nonlocal appendix_chunk_order
            nonlocal table_chunk_order
            nonlocal inside_appendix

            if not appendix_buffer:
                inside_appendix = False
                return

            appendix_blocks = split_appendix_blocks(
                appendix_buffer
            )

            for block_kind, block_content in appendix_blocks:
                block_parts = split_long_text(
                    block_content,
                    max_chars=3000,
                )

                for part_index, block_part in enumerate(
                    block_parts,
                    start=1,
                ):
                    if block_kind == "table":
                        table_chunk_order += 1
                        block_order = table_chunk_order
                        block_label = "Bảng"
                        id_label = "bang"
                    else:
                        appendix_chunk_order += 1
                        block_order = appendix_chunk_order
                        block_label = "Nội dung phụ lục"
                        id_label = "phu_luc"

                    if len(block_parts) > 1:
                        article_value = (
                            f"{appendix_title or 'Phụ lục'} - "
                            f"{block_label} {block_order}, phần {part_index}"
                        )
                    else:
                        article_value = (
                            f"{appendix_title or 'Phụ lục'} - "
                            f"{block_label} {block_order}"
                        )

                    chunk_id = (
                        f"{make_safe_id(document_id)}"
                        f"_{id_label}_{block_order:04d}"
                        f"_{part_index:02d}"
                    )

                    appendix_links = extract_link_to(
                        text=block_part,
                        current_document_number=current_document_number,
                        scope="appendix",
                    )

                    temp_token = {
                        "id": chunk_id,
                        "document_id": document_id,
                        "document_number": current_document_number,
                        "title": metadata_item.get("title"),
                        "url": metadata_item.get("url"),
                        "legal_type": metadata_item.get("legal_type"),
                        "legal_sectors": metadata_item.get("legal_sectors"),
                        "issuing_authority": metadata_item.get(
                            "issuing_authority"
                        ),
                        "issuance_date": metadata_item.get(
                            "issuance_date"
                        ),
                        "signers": metadata_item.get("signers"),
                        "part": current_part,
                        "chapter": current_chapter,
                        "section": current_section,
                        "articles": article_value,
                        "content": block_part,
                        "link_to": appendix_links,
                    }

                    chunk_list.append(temp_token)

            appendix_buffer = []
            inside_appendix = False

        # --------------------------------------------------------------------
        # Duyệt toàn bộ văn bản, bao gồm cả phần mở đầu.
        # Các dòng trước Điều đầu tiên không được thêm vào content của chunk,
        # nhưng đã được quét link ở document_header_text phía trên.
        # --------------------------------------------------------------------

        for line in parser_content.split("\n"):
            line = line.strip()

            if not line:
                continue

            # ------------------------------------------------------------
            # Nếu tiêu đề phụ lục nằm trong một dòng hành chính dài,
            # lấy phần từ tiêu đề phụ lục trở đi và bỏ phần chữ ký phía trước.
            # ------------------------------------------------------------

            inline_appendix_match = APPENDIX_INLINE_PATTERN.search(line)

            if (
                inside_admin_tail
                and inline_appendix_match is not None
            ):
                flush_article()
                inside_admin_tail = False
                inside_appendix = True
                appendix_title = inline_appendix_match.group(0)
                appendix_buffer = [
                    line[inline_appendix_match.start():].strip()
                ]
                continue

            # ------------------------------------------------------------
            # Nhận diện tiêu đề phụ lục/danh mục ở đầu dòng.
            # Dòng DANH MỤC chỉ được coi là phụ lục khi đã đi qua phần
            # hành chính; PHỤ LỤC được nhận diện trực tiếp.
            # ------------------------------------------------------------

            appendix_match = APPENDIX_START_PATTERN.match(line)
            explicit_appendix = bool(
                re.match(
                    r"^\s*PHỤ\s+LỤC\b",
                    line,
                    flags=re.IGNORECASE,
                )
            )

            if appendix_match and (
                inside_admin_tail
                or explicit_appendix
            ):
                flush_article()

                if inside_appendix:
                    flush_appendix()

                inside_admin_tail = False
                inside_appendix = True
                appendix_title = line
                appendix_buffer = [line]
                continue

            # Khi đang đọc phụ lục, giữ nguyên từng dòng để sau đó
            # phân biệt khối văn bản và khối bảng theo dấu |.
            if inside_appendix:
                appendix_buffer.append(line)
                continue

            # ------------------------------------------------------------
            # Nơi nhận có thể nằm giữa cùng một dòng với cuối Điều.
            # Giữ phần trước "Nơi nhận", kết thúc Điều rồi bỏ phần sau.
            # ------------------------------------------------------------

            admin_inline_match = ADMIN_INLINE_PATTERN.search(line)

            if admin_inline_match is not None:
                article_prefix = line[
                    :admin_inline_match.start()
                ].strip()

                if (
                    article_prefix
                    and current_article_number is not None
                ):
                    article_buffer.append(article_prefix)

                flush_article()
                inside_admin_tail = True

                remaining_tail = line[
                    admin_inline_match.end():
                ].strip()

                tail_appendix_match = APPENDIX_INLINE_PATTERN.search(
                    remaining_tail
                )

                if tail_appendix_match is not None:
                    inside_admin_tail = False
                    inside_appendix = True
                    appendix_title = tail_appendix_match.group(0)
                    appendix_buffer = [
                        remaining_tail[
                            tail_appendix_match.start():
                        ].strip()
                    ]

                continue

            # Bắt đầu phần nơi nhận/chữ ký. Phần này không được đưa vào
            # content và cũng không dùng để trích xuất link_to.
            if ADMIN_TAIL_PATTERN.match(line):
                flush_article()
                inside_admin_tail = True
                continue

            # Vẫn duyệt tiếp để có thể gặp phụ lục phía sau chữ ký.
            if inside_admin_tail:
                continue

            part_match = PART_PATTERN.match(line)
            chapter_match = CHAPTER_PATTERN.match(line)
            section_match = SECTION_PATTERN.match(line)
            article_match = ARTICLE_PATTERN.match(line)

            has_structure_match = any(
                [
                    part_match,
                    chapter_match,
                    section_match,
                    article_match,
                ]
            )

            # Tiêu đề ở dòng ngay sau Phần/Chương/Mục.
            if (
                pending_title_level is not None
                and not has_structure_match
            ):
                if looks_like_title(line):
                    if pending_title_level == "part":
                        current_part = f"{current_part}: {line}"

                    elif pending_title_level == "chapter":
                        current_chapter = (
                            f"{current_chapter}: {line}"
                        )

                    elif pending_title_level == "section":
                        current_section = (
                            f"{current_section}: {line}"
                        )

                    pending_title_level = None
                    continue

                pending_title_level = None

            if part_match:
                flush_article()

                # Giữ nguyên cách viết của tiêu đề trong văn bản gốc.
                current_part = line

                title = (
                    part_match.group("title") or ""
                ).strip()

                if not title:
                    pending_title_level = "part"

                current_chapter = None
                current_section = None
                continue

            if chapter_match:
                flush_article()

                current_chapter = line

                title = (
                    chapter_match.group("title") or ""
                ).strip()

                if not title:
                    pending_title_level = "chapter"

                current_section = None
                continue

            if section_match:
                flush_article()

                current_section = line

                title = (
                    section_match.group("title") or ""
                ).strip()

                if not title:
                    pending_title_level = "section"

                continue

            if article_match:
                flush_article()

                current_article_order += 1

                current_article_number = (
                    article_match.group("number").strip()
                )

                current_article_title = (
                    article_match.group("title") or ""
                ).strip() or None

                article_buffer = [line]
                pending_title_level = None
                continue

            if current_article_number is not None:
                article_buffer.append(line)

        # Lưu Điều hoặc phụ lục cuối cùng của văn bản.
        flush_article()
        flush_appendix()

    LOGGER.info(
        "Chunking completed: %d documents phù hợp, "
        "%d documents bị bỏ qua, %d chunks được tạo.",
        matched_document_count,
        skipped_document_count,
        len(chunk_list),
    )

    if not chunk_list:
        LOGGER.warning(
            "Không tạo được chunk nào cho target '%s'.",
            target_legal_sector,
        )
        return []

    # ------------------------------------------------------------------------
    # In đầy đủ 5 chunk đầu tiên để kiểm tra, không cắt content.
    # ------------------------------------------------------------------------

    if print_all_chunks:
        preview_count = min(5, len(chunk_list))

        print("\n" + "=" * 120)
        print(
            f"{preview_count} CHUNK ĐẦU TIÊN "
            f"TRÊN TỔNG SỐ {len(chunk_list)} CHUNK"
        )
        print("=" * 120)

        for position, chunk in enumerate(
            chunk_list[:5],
            start=1,
        ):
            print(
                f"\n{'=' * 40} "
                f"CHUNK {position}/{preview_count} "
                f"{'=' * 40}"
            )

            print(
                json.dumps(
                    chunk,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
            )

    # ------------------------------------------------------------------------
    # Lưu Parquet
    # ------------------------------------------------------------------------

    output_file = Path(output_path)
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_dataset = Dataset.from_list(
        chunk_list
    )

    output_dataset.to_parquet(
        str(output_file)
    )

    LOGGER.info(
        "Đã lưu file chunk thành công: %s",
        output_file,
    )

# ============================================================================
# CLASS CHUNK
# ============================================================================

class Chunk:
    """
    Điều phối toàn bộ pipeline chunk văn bản pháp luật.

    Class này sử dụng nguyên hàm ``chunk_dataset`` phía trên nên giữ đầy đủ
    logic cũ: lọc lĩnh vực, nhận diện Phần/Chương/Mục/Điều, loại Nơi nhận và
    chữ ký, giữ phụ lục, tách bảng, chia đoạn dài, tạo ID và trích xuất link_to.
    """

    def __init__(
        self,
        content_path: str = "./data/ds_content",
        metadata_path: str = "./data/ds_metadata",
        output_path: str = "./data/legal_chunks.parquet",
        target_legal_sector: str = "Giáo dục",
        min_chunk_chars: int = 50,
        print_all_chunks: bool = True,
    ) -> None:
        self.content_path = Path(content_path)
        self.metadata_path = Path(metadata_path)
        self.output_path = Path(output_path)
        self.target_legal_sector = target_legal_sector
        self.min_chunk_chars = min_chunk_chars
        self.print_all_chunks = print_all_chunks

        self.ds_content: Dataset | DatasetDict | None = None
        self.ds_metadata: Dataset | DatasetDict | None = None

    def validate(self) -> None:
        """Kiểm tra cấu hình và đường dẫn đầu vào."""

        if not self.target_legal_sector.strip():
            raise ValueError("target_legal_sector không được để trống.")

        if self.min_chunk_chars < 0:
            raise ValueError("min_chunk_chars phải lớn hơn hoặc bằng 0.")

        if not self.content_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy dataset content: {self.content_path.resolve()}"
            )

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy dataset metadata: {self.metadata_path.resolve()}"
            )

        if self.output_path.suffix.lower() != ".parquet":
            raise ValueError(
                f"output_path phải có đuôi .parquet: {self.output_path}"
            )

    def load_data(self) -> tuple[Dataset | DatasetDict, Dataset | DatasetDict]:
        """Load dataset content và metadata từ disk."""

        self.validate()

        LOGGER.info("Đang load content dataset: %s", self.content_path)
        self.ds_content = load_from_disk(str(self.content_path))

        LOGGER.info("Đang load metadata dataset: %s", self.metadata_path)
        self.ds_metadata = load_from_disk(str(self.metadata_path))

        content_data = get_data_split(self.ds_content)
        metadata_data = get_data_split(self.ds_metadata)

        if len(content_data) != len(metadata_data):
            raise ValueError(
                "Số bản ghi content và metadata không bằng nhau: "
                f"{len(content_data)} != {len(metadata_data)}"
            )

        LOGGER.info(
            "Đã load %d content records và %d metadata records.",
            len(content_data),
            len(metadata_data),
        )

        return self.ds_content, self.ds_metadata

    def chunk(
        self,
        ds_content: Dataset | DatasetDict | None = None,
        ds_metadata: Dataset | DatasetDict | None = None,
    ) -> Path:
        """
        Chạy nguyên logic chunk cũ và trả về đường dẫn file Parquet đầu ra.

        Có thể truyền Dataset trực tiếp. Nếu không truyền, class tự load từ
        ``content_path`` và ``metadata_path``.
        """

        if ds_content is not None:
            self.ds_content = ds_content

        if ds_metadata is not None:
            self.ds_metadata = ds_metadata

        if self.ds_content is None or self.ds_metadata is None:
            self.load_data()

        if self.ds_content is None or self.ds_metadata is None:
            raise RuntimeError("Không thể load dữ liệu đầu vào.")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        chunk_dataset(
            ds_content=self.ds_content,
            ds_metadata=self.ds_metadata,
            target_legal_sector=self.target_legal_sector,
            output_path=str(self.output_path),
            min_chunk_chars=self.min_chunk_chars,
            print_all_chunks=self.print_all_chunks,
        )

        if not self.output_path.exists():
            raise RuntimeError(
                "Chunking kết thúc nhưng không tạo được file đầu ra: "
                f"{self.output_path.resolve()}"
            )

        return self.output_path

    def run(self) -> Path:
        """Load dữ liệu rồi chạy chunking."""

        self.load_data()
        return self.chunk()

    def preview(self, limit: int = 5) -> list[dict[str, Any]]:
        """Đọc và in một số chunk đầu tiên từ file Parquet đã tạo."""

        if limit <= 0:
            return []

        if not self.output_path.exists():
            raise FileNotFoundError(
                f"Chưa có file chunk đầu ra: {self.output_path.resolve()}"
            )

        dataset = Dataset.from_parquet(str(self.output_path))
        preview_count = min(limit, len(dataset))
        rows = [dataset[index] for index in range(preview_count)]

        print("\n" + "=" * 120)
        print(f"{preview_count} CHUNK ĐẦU TIÊN TRÊN TỔNG SỐ {len(dataset)} CHUNK")
        print("=" * 120)

        for index, row in enumerate(rows, start=1):
            print(f"\n{'=' * 40} CHUNK {index}/{preview_count} {'=' * 40}")
            print(json.dumps(row, ensure_ascii=False, indent=2, default=str))

        return rows

    def statistics(self) -> dict[str, Any]:
        """Tính thống kê cơ bản của file chunk đầu ra."""

        if not self.output_path.exists():
            raise FileNotFoundError(
                f"Chưa có file chunk đầu ra: {self.output_path.resolve()}"
            )

        dataset = Dataset.from_parquet(str(self.output_path))
        document_ids: set[str] = set()
        article_chunks = 0
        appendix_chunks = 0
        table_chunks = 0
        content_lengths: list[int] = []

        for row in dataset:
            document_ids.add(str(row.get("document_id")))
            chunk_id = str(row.get("id", "")).casefold()

            if "_dieu_" in chunk_id:
                article_chunks += 1
            elif "_phu_luc_" in chunk_id:
                appendix_chunks += 1
            elif "_bang_" in chunk_id:
                table_chunks += 1

            content_lengths.append(len(str(row.get("content", ""))))

        result = {
            "target_legal_sector": self.target_legal_sector,
            "output_path": str(self.output_path.resolve()),
            "total_documents": len(document_ids),
            "total_chunks": len(dataset),
            "article_chunks": article_chunks,
            "appendix_chunks": appendix_chunks,
            "table_chunks": table_chunks,
            "min_content_chars": min(content_lengths, default=0),
            "max_content_chars": max(content_lengths, default=0),
            "average_content_chars": round(
                sum(content_lengths) / len(content_lengths), 2
            ) if content_lengths else 0.0,
        }

        print("\n" + "=" * 80)
        print("THỐNG KÊ CHUNK")
        print("=" * 80)
        print(json.dumps(result, ensure_ascii=False, indent=2))

        return result


# ============================================================================
# CHẠY THỦ CÔNG
# ============================================================================

if __name__ == "__main__":
    chunker = Chunk(
        content_path="./data/ds_content",
        metadata_path="./data/ds_metadata",
        output_path="./data/legal_chunks.parquet",
        target_legal_sector="Giáo dục",
        min_chunk_chars=50,
        print_all_chunks=True,
    )

    output_file = chunker.run()
    LOGGER.info("File chunk đầu ra: %s", output_file.resolve())
    chunker.statistics()

