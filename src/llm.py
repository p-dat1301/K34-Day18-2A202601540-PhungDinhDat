from __future__ import annotations

"""LLM helper — dùng chung cho answer generation, RAGAS judge và enrichment.

Chạy trên Gemini qua OpenAI-compat endpoint (generativelanguage.googleapis.com/v1beta/openai/),
nên không cần cài thêm package ngoài openai / langchain-openai đã có trong requirements.
"""

import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (GEMINI_API_KEY, OPENAI_API_KEY, GEMINI_BASE_URL, GEMINI_MODEL,
                    GEMINI_JUDGE_MODEL, LLM_PROVIDER, EMBEDDING_MODEL,
                    RAGAS_EMBEDDING_DEVICE)


MAX_RETRIES = 6


def _is_rate_limit(err: Exception) -> bool:
    text = str(err)
    return "429" in text or "RESOURCE_EXHAUSTED" in text or "rate limit" in text.lower()


def _retry_delay(err: Exception, attempt: int) -> float:
    """Ưu tiên retryDelay API trả về, fallback exponential backoff 4/8/16/32s..."""
    m = re.search(r"retryDelay['\"]?:\s*['\"]?(\d+(?:\.\d+)?)s", str(err))
    if m:
        return float(m.group(1)) + 1.0
    m = re.search(r"retry in (\d+(?:\.\d+)?)s", str(err))
    if m:
        return float(m.group(1)) + 1.0
    return min(4.0 * (2 ** attempt), 64.0)


def chat(system: str, user: str, max_tokens: int = 300) -> str:
    """Gọi LLM 1 lần. Trả về "" khi không có key hoặc lỗi — caller tự fallback."""
    if LLM_PROVIDER == "none":
        return ""
    try:
        if LLM_PROVIDER == "gemini":
            from openai import OpenAI
            client = OpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL)
            model = GEMINI_MODEL
        else:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            model = "gpt-4o-mini"
        kwargs = {}
        if LLM_PROVIDER == "gemini":
            # gemini-3.x là thinking model: reasoning tokens ăn vào max_tokens.
            # reasoning_effort="low" giới hạn phần thinking; nới max_tokens để
            # phần answer thực sự còn chỗ (nếu không -> finish_reason="length",
            # content=None và caller nhận chuỗi rỗng).
            kwargs["reasoning_effort"] = "low"
            max_tokens = max(max_tokens * 4, 1024)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        # Free tier giới hạn requests/phút -> 429. Retry với exponential backoff,
        # ưu tiên retryDelay do API trả về.
        last_err = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = client.chat.completions.create(
                    model=model, messages=messages, max_tokens=max_tokens, **kwargs)
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                last_err = e
                if not _is_rate_limit(e) or attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(_retry_delay(e, attempt))
        raise last_err
    except Exception as e:
        print(f"  ⚠️  LLM call failed ({LLM_PROVIDER}): {str(e)[:200]}")
        return ""


def chat_json(system: str, user: str, max_tokens: int = 400) -> dict:
    """Gọi LLM yêu cầu trả JSON. Trả {} khi lỗi — caller tự fallback."""
    import json

    raw = chat(system, user, max_tokens=max_tokens)
    if not raw:
        return {}
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        return json.loads(raw[start:end]) if start != -1 and end > start else {}
    except Exception:
        return {}


def get_ragas_llm():
    """LangChain LLM wrapper cho RAGAS (0.1.x dùng BaseRagasLLM)."""
    from src import _ragas_compat
    _ragas_compat.install()
    from ragas.llms import LangchainLLMWrapper

    from langchain_openai import ChatOpenAI

    if LLM_PROVIDER == "gemini":
        # Dùng ChatOpenAI trỏ vào endpoint OpenAI-compat của Gemini.
        # ChatGoogleGenerativeAI native không dùng được: RAGAS truyền `temperature`
        # xuống call-time -> TypeError: generate_content() got an unexpected
        # keyword argument 'temperature'.
        _ragas_compat.disable_multiple_completions()
        langchain_llm = ChatOpenAI(
            model=GEMINI_JUDGE_MODEL, api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL,
            temperature=0.0, max_retries=6, timeout=120)
    else:
        langchain_llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0.0)
    return LangchainLLMWrapper(langchain_llm)


def get_ragas_embeddings():
    """Embeddings cho RAGAS — dùng local bge-m3 (không cần API key)."""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    # CPU: GPU đã bị bge-m3 của DenseSearch chiếm -> nếu để chung sẽ CUDA OOM.
    emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": RAGAS_EMBEDDING_DEVICE},
    ))
    return emb