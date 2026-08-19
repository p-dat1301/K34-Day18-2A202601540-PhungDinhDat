from __future__ import annotations

"""Vá tương thích cho RAGAS 0.1.22. Gọi TRƯỚC khi import ragas.

1. RAGAS import cứng `langchain_community.chat_models.vertexai`. Bản
   langchain-community mới đã bỏ module này nhưng RAGAS vẫn cần nó (chỉ để
   isinstance check trong MULTIPLE_COMPLETION_SUPPORTED) -> cung cấp stub.

2. RAGAS coi mọi `ChatOpenAI` là hỗ trợ `n>1` (multiple candidates). Ta trỏ
   ChatOpenAI vào endpoint OpenAI-compat của Gemini, nơi `n>1` trả lỗi
   400 "Multiple candidates is not enabled for this model". Vá hàm
   `is_multiple_completion_supported` -> False để RAGAS lặp prompt n lần
   thay vì gửi `n` lên API.
"""

import sys
import types


def install() -> None:
    if "langchain_community.chat_models.vertexai" in sys.modules:
        return
    try:
        import langchain_community.chat_models  # noqa: F401
    except ImportError:
        return

    stub = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:
        """Stub — RAGAS chỉ dùng để isinstance check, không bao giờ khởi tạo."""

    class VertexAI:
        pass

    stub.ChatVertexAI = ChatVertexAI
    stub.VertexAI = VertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = stub


def disable_multiple_completions() -> None:
    """Buộc RAGAS lặp prompt thay vì gửi `n>1` (Gemini OpenAI-compat không hỗ trợ)."""
    install()
    import ragas.llms.base as ragas_llms_base

    ragas_llms_base.is_multiple_completion_supported = lambda llm: False
