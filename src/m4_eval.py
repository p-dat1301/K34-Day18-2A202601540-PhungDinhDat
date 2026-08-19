from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, RAGAS_MAX_WORKERS


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    if not questions:
        return {"faithfulness": 0.0, "answer_relevancy": 0.0,
                "context_precision": 0.0, "context_recall": 0.0, "per_question": []}
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset
        from src.llm import get_ragas_llm, get_ragas_embeddings

        llm = get_ragas_llm()
        embeddings = get_ragas_embeddings()
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        for m in metrics:
            m.llm = llm
            m.embeddings = embeddings

        def _score(row, key):
            """RAGAS trả nan khi 1 metric fail cho câu đó — quy về 0.0 để
            aggregate và failure_analysis không bị nhiễm nan."""
            import math
            try:
                v = float(row.get(key, 0.0))
            except (TypeError, ValueError):
                return 0.0
            return 0.0 if math.isnan(v) else v

        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })
        # Free tier Gemini giới hạn requests/phút. RAGAS mặc định chạy 16 worker
        # song song -> 429 hàng loạt. Hạ max_workers + nới timeout để chạy ổn định.
        from ragas.run_config import RunConfig
        run_config = RunConfig(max_workers=RAGAS_MAX_WORKERS, timeout=300, max_retries=6)
        result = evaluate(dataset, metrics=metrics, llm=llm, embeddings=embeddings,
                          run_config=run_config, raise_exceptions=False)
        df = result.to_pandas()
        per_question = [EvalResult(
            question=row["question"], answer=row["answer"],
            contexts=row["contexts"], ground_truth=row["ground_truth"],
            faithfulness=_score(row, "faithfulness"),
            answer_relevancy=_score(row, "answer_relevancy"),
            context_precision=_score(row, "context_precision"),
            context_recall=_score(row, "context_recall"),
        ) for _, row in df.iterrows()]

        def _agg(key):
            vals = [getattr(r, key) for r in per_question if getattr(r, key) is not None]
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        return {
            "faithfulness": _agg("faithfulness"),
            "answer_relevancy": _agg("answer_relevancy"),
            "context_precision": _agg("context_precision"),
            "context_recall": _agg("context_recall"),
            "per_question": per_question,
        }
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return {"faithfulness": 0.0, "answer_relevancy": 0.0,
                "context_precision": 0.0, "context_recall": 0.0, "per_question": []}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }

    scored = []
    for r in eval_results:
        metric_map = {
            "faithfulness": r.faithfulness,
            "answer_relevancy": r.answer_relevancy,
            "context_precision": r.context_precision,
            "context_recall": r.context_recall,
        }
        avg = sum(metric_map.values()) / max(len(metric_map), 1)
        worst_metric = min(metric_map, key=metric_map.get)
        scored.append({"question": r.question, "avg": avg,
                       "worst_metric": worst_metric, "score": metric_map[worst_metric]})

    scored.sort(key=lambda x: x["avg"])
    failures = []
    for item in scored[:bottom_n]:
        diagnosis, suggested_fix = diagnostic_tree[item["worst_metric"]]
        failures.append({
            "question": item["question"],
            "worst_metric": item["worst_metric"],
            "score": round(item["score"], 4),
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })
    return failures


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json",
                latency: dict | None = None):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    per_question = results.get("per_question", [])
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(per_question),
        "latency_ms": latency or {},
        "failures": failures,
        "per_question": [
            {
                "question": r.question,
                "answer": r.answer,
                "ground_truth": r.ground_truth,
                "contexts": list(r.contexts),
                "faithfulness": r.faithfulness,
                "answer_relevancy": r.answer_relevancy,
                "context_precision": r.context_precision,
                "context_recall": r.context_recall,
            }
            for r in per_question
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
