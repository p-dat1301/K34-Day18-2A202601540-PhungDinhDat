from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K, RERANKER_MODEL, RERANKER_DEVICE


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = RERANKER_MODEL, device: str = RERANKER_DEVICE):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load_model(self):
        """Lazy load — chỉ tải model khi thực sự rerank.

        device mặc định "cpu": GPU 3.68 GiB đã dành cho bge-m3 của DenseSearch,
        nạp thêm reranker 2.2 GB lên GPU sẽ CUDA OOM.
        """
        if self._model is None:
            from sentence_transformers import CrossEncoder
            kwargs = {"device": self.device} if self.device else {}
            self._model = CrossEncoder(self.model_name, **kwargs)
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents:
            return []
        model = self._load_model()
        pairs = [(query, doc["text"]) for doc in documents]
        scores = model.predict(pairs)
        if isinstance(scores, (int, float)):
            scores = [scores]
        scored = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        return [
            RerankResult(text=doc["text"], original_score=float(doc.get("score", 0.0)),
                         rerank_score=float(score), metadata=doc.get("metadata", {}), rank=i)
            for i, (score, doc) in enumerate(scored[:top_k])
        ]


class FlashrankReranker:
    """Alternative nhẹ hơn CrossEncoder.

    Model mặc định của flashrank ~4MB (so với 2.2GB của bge-reranker-v2-m3) nên
    chạy CPU vẫn nhanh. Đổi lại chất lượng thấp hơn trên tiếng Việt vì model
    mặc định train chủ yếu trên tiếng Anh — dùng khi latency quan trọng hơn
    precision. So sánh bằng `benchmark_reranker()`.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from flashrank import Ranker
            self._model = Ranker(model_name=self.model_name) if self.model_name else Ranker()
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents:
            return []
        from flashrank import RerankRequest

        model = self._load_model()
        # flashrank trả về passage kèm "id" -> map ngược lại document gốc để
        # giữ nguyên metadata và original_score.
        passages = [{"id": i, "text": doc["text"]} for i, doc in enumerate(documents)]
        ranked = model.rerank(RerankRequest(query=query, passages=passages))
        return [
            RerankResult(
                text=documents[item["id"]]["text"],
                original_score=float(documents[item["id"]].get("score", 0.0)),
                rerank_score=float(item["score"]),
                metadata=documents[item["id"]].get("metadata", {}),
                rank=i,
            )
            for i, item in enumerate(ranked[:top_k])
        ]


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
