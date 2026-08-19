from __future__ import annotations

"""Production RAG Pipeline — Bài tập NHÓM: ghép M1+M2+M3+M4."""

import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1)
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": child.parent_id}})
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({time.time()-t0:.1f}s)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        print(f"  ✓ Enriched {len(enriched)} chunks ({time.time()-t0:.1f}s)", flush=True)
    else:
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    print(f"  ✓ Indexed ({time.time()-t0:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    print(f"  ✓ Reranker ready ({time.time()-t0:.1f}s)", flush=True)

    return search, reranker


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker,
              timings: dict | None = None) -> tuple[str, list[str]]:
    """Run single query through pipeline.

    timings: nếu truyền vào, ghi latency (ms) từng bước để làm latency breakdown.
    """
    t0 = time.perf_counter()
    results = search.search(query)
    t_search = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    t_rerank = (time.perf_counter() - t0) * 1000
    contexts = [r.text for r in reranked] if reranked else [r.text for r in results[:3]]

    from src.llm import chat
    t0 = time.perf_counter()
    if contexts:
        try:
            context_str = "\n\n".join(contexts)
            answer = chat("Trả lời CHỈ dựa trên context. Nếu không có → nói 'Không tìm thấy.'",
                          f"Context:\n{context_str}\n\nCâu hỏi: {query}")
            if not answer:
                answer = contexts[0]
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}", flush=True)
            answer = contexts[0]
    else:
        answer = "Không tìm thấy thông tin."
    t_llm = (time.perf_counter() - t0) * 1000

    if timings is not None:
        timings.setdefault("search_ms", []).append(t_search)
        timings.setdefault("rerank_ms", []).append(t_rerank)
        timings.setdefault("llm_ms", []).append(t_llm)
        timings.setdefault("total_ms", []).append(t_search + t_rerank + t_llm)
    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []
    timings: dict[str, list[float]] = {}

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker, timings)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    print(f"  ✓ RAGAS done ({time.time()-t0:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    latency = _summarize_latency(timings)
    _print_latency(latency)

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures, latency=latency)
    return results


def _summarize_latency(timings: dict[str, list[float]]) -> dict:
    """Trung bình / p50 / p95 (ms) cho từng bước của pipeline."""
    summary = {}
    for stage, values in timings.items():
        if not values:
            continue
        ordered = sorted(values)
        summary[stage] = {
            "avg": round(sum(ordered) / len(ordered), 1),
            "p50": round(ordered[len(ordered) // 2], 1),
            "p95": round(ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)], 1),
            "n": len(ordered),
        }
    return summary


def _print_latency(latency: dict) -> None:
    if not latency:
        return
    print("\n" + "=" * 60)
    print("LATENCY BREAKDOWN (ms/query)")
    print("=" * 60)
    print(f"  {'Stage':<14} {'avg':>9} {'p50':>9} {'p95':>9}")
    labels = {"search_ms": "Hybrid search", "rerank_ms": "Rerank", "llm_ms": "LLM answer",
              "total_ms": "TOTAL"}
    for stage in ["search_ms", "rerank_ms", "llm_ms", "total_ms"]:
        s = latency.get(stage)
        if s:
            print(f"  {labels[stage]:<14} {s['avg']:>9.1f} {s['p50']:>9.1f} {s['p95']:>9.1f}")


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
