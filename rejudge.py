"""Chấm lại RAGAS từ report đã lưu, không chạy lại retrieval/enrichment.

Dùng khi lần chạy trước bị NaN vì hết quota judge model: report đã lưu sẵn
question / answer / contexts / ground_truth cho từng câu, nên chỉ cần gọi lại
`evaluate_ragas` với `GEMINI_JUDGE_MODEL` khác (mỗi model có quota ngày riêng).

    GEMINI_JUDGE_MODEL=gemini-3.1-flash-lite python rejudge.py ragas_report.json

Không truyền tham số thì chấm lại cả naive_baseline_report.json và ragas_report.json.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import GEMINI_JUDGE_MODEL
from src.m4_eval import EvalResult, evaluate_ragas, failure_analysis, save_report

DEFAULT_REPORTS = ["naive_baseline_report.json", "ragas_report.json"]


def rejudge(path: str) -> None:
    with open(path, encoding="utf-8") as f:
        report = json.load(f)

    per_question = report.get("per_question", [])
    if not per_question:
        print(f"  ⚠️  {path}: không có per_question để chấm lại — bỏ qua.")
        return

    print(f"\n=== {path}: chấm lại {len(per_question)} câu bằng {GEMINI_JUDGE_MODEL} ===",
          flush=True)
    results = evaluate_ragas(
        [q["question"] for q in per_question],
        [q["answer"] for q in per_question],
        [q["contexts"] for q in per_question],
        [q["ground_truth"] for q in per_question],
    )

    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        measured = results.get("measured", {}).get(m, "?")
        print(f"  {m}: {results.get(m, 0):.4f}  (đo được {measured}/{len(per_question)})")

    new_results: list[EvalResult] = results.get("per_question", [])
    failures = failure_analysis(new_results, bottom_n=5) if new_results else []
    save_report(results, failures, path=path, latency=report.get("latency_ms"))


if __name__ == "__main__":
    targets = sys.argv[1:] or DEFAULT_REPORTS
    for p in targets:
        if os.path.exists(p):
            rejudge(p)
        else:
            print(f"  ⚠️  Không thấy {p} — bỏ qua.")
