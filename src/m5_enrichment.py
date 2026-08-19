from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys, re, json, hashlib
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LLM_PROVIDER, GEMINI_MODEL, ENRICHMENT_CACHE_PATH
from src.llm import chat, chat_json


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    if LLM_PROVIDER != "none":
        summary = chat("Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt.",
                       text, max_tokens=150)
        if summary:
            return summary

    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    return ". ".join(sentences[:2]) + "." if sentences else text


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    if LLM_PROVIDER != "none":
        raw = chat(f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. "
                   "Mỗi câu hỏi trên 1 dòng, kết thúc bằng dấu '?'. "
                   "Không đánh số, không thêm lời dẫn.", text, max_tokens=200)
        questions = _clean_questions(raw, n_questions)
        if questions:
            return questions

    sentences = [s.strip() for s in re.split(r"[.!?\n]", text) if len(s.strip()) > 10]
    return [f"{s.rstrip('.')}?" for s in sentences[:n_questions]]


def _clean_questions(raw: str, n_questions: int) -> list[str]:
    """Chuẩn hoá output LLM thành list câu hỏi sạch.

    LLM hay trả kèm lời dẫn ("Dưới đây là 3 câu hỏi:"), bullet/markdown, hoặc
    quên dấu '?' — chuẩn hoá ở đây để downstream (embed HyQA) luôn nhận câu hỏi.
    """
    if not raw:
        return []
    questions = []
    for line in raw.strip().split("\n"):
        q = line.strip().lstrip("*-•").strip()
        q = re.sub(r"^\d+[.)]\s*", "", q)          # bỏ "1." / "2)"
        q = re.sub(r"^\*\*(.+?)\*\*:?\s*", r"\1", q)  # bỏ **bold**
        q = q.strip().strip("*").strip()
        if len(q) < 5 or q.endswith(":"):           # bỏ lời dẫn kiểu "Câu hỏi:"
            continue
        if not q.endswith("?"):
            q = q.rstrip(".") + "?"
        questions.append(q)
    return questions[:n_questions]


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    if LLM_PROVIDER != "none":
        context = chat("Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và "
                       "nói về chủ đề gì. Chỉ trả về 1 câu.",
                       f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}", max_tokens=80)
        if context:
            return f"{context}\n\n{text}"

    prefix = f"Trích từ {document_title}. " if document_title else ""
    return f"{prefix}{text}"


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    if LLM_PROVIDER != "none":
        meta = chat_json(
            'Trích xuất metadata từ đoạn văn. Trả về JSON: '
            '{"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}',
            text, max_tokens=150)
        if meta:
            return meta

    return {"topic": "general", "entities": [], "category": "policy", "language": "vi"}


# ─── Combined Single-Call Mode ───────────────────────────


_CACHE: dict | None = None


def _load_cache() -> dict:
    """Cache enrichment theo nội dung chunk — chạy lại pipeline không tốn quota API."""
    global _CACHE
    if _CACHE is None:
        try:
            with open(ENRICHMENT_CACHE_PATH, encoding="utf-8") as f:
                _CACHE = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            _CACHE = {}
    return _CACHE


def _save_cache() -> None:
    if _CACHE is None:
        return
    os.makedirs(os.path.dirname(ENRICHMENT_CACHE_PATH), exist_ok=True)
    with open(ENRICHMENT_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(_CACHE, f, ensure_ascii=False)


def _cache_key(text: str, source: str) -> str:
    return hashlib.sha1(f"{GEMINI_MODEL}|{source}|{text}".encode("utf-8")).hexdigest()


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    Cost optimization: 1 API call thay vì 4 calls riêng lẻ, cộng cache trên đĩa
    để lần chạy sau không gọi lại API cho cùng một chunk.
    """
    if LLM_PROVIDER == "none":
        return {}

    cache = _load_cache()
    key = _cache_key(text, source)
    if key in cache:
        return cache[key]

    result = _call_enrich_llm(text, source)
    if result:
        cache[key] = result
        _save_cache()
    return result


def _call_enrich_llm(text: str, source: str) -> dict:
    if LLM_PROVIDER != "none":
        return chat_json(
            'Phân tích đoạn văn và trả về JSON: '
            '{"summary": "tóm tắt 2-3 câu", "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"], '
            '"context": "1 câu mô tả đoạn văn nằm ở đâu trong tài liệu", '
            '"metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}}',
            f"Tài liệu: {source}\n\nĐoạn văn:\n{text}", max_tokens=400)
    return {}


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
