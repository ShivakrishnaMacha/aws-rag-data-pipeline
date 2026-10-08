#!/usr/bin/env python3
"""Eval harness: recall@k, MRR, and citation grounding on the synthetic QA set.

Writes eval/report.md. Target: recall@5 >= 0.80 on the synthetic set.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QA_PATH = ROOT / "eval" / "eval_qa.json"
REPORT_PATH = ROOT / "eval" / "report.md"
K = 5


def main() -> dict:
    from retrieval.answer import compose_answer
    from retrieval.search import HybridRetriever

    qa = json.loads(QA_PATH.read_text(encoding="utf-8"))
    retriever = HybridRetriever()
    hits = 0
    reciprocal_ranks: list[float] = []
    cited_correct = 0
    latencies: list[float] = []
    rows: list[str] = []

    for item in qa:
        sr = retriever.search(item["question"], k=K)
        latencies.append(sr["latency_ms"])
        retrieved_ids = [r["doc_id"] for r in sr["results"]]
        expected = item["expected_doc_id"]
        if expected in retrieved_ids:
            hits += 1
            reciprocal_ranks.append(1.0 / (retrieved_ids.index(expected) + 1))
        else:
            reciprocal_ranks.append(0.0)
        ans = compose_answer(sr)
        cited_ids = [c["doc_id"] for c in ans["citations"]]
        ok = expected in cited_ids
        cited_correct += ok
        rows.append(
            f"| {item['question'][:60]}… | {expected} | "
            f"{'✅' if expected in retrieved_ids else '❌'} | "
            f"{'✅' if ok else '❌'} |"
        )

    n = len(qa)
    metrics = {
        "n_questions": n,
        "k": K,
        "recall_at_k": round(hits / n, 3),
        "mrr": round(sum(reciprocal_ranks) / n, 3),
        "citation_accuracy": round(cited_correct / n, 3),
        "p50_latency_ms": round(sorted(latencies)[n // 2], 1),
    }
    report = (
        "# Retrieval eval report\n\n"
        f"Corpus: synthetic ops runbooks. QA set: `{QA_PATH.name}` ({n} questions).\n\n"
        "## Metrics\n\n"
        "| metric | value |\n|---|---|\n"
        + "".join(f"| {k} | {v} |\n" for k, v in metrics.items())
        + "\n## Per-question results\n\n"
        "| question | expected doc | retrieved@k | cited correctly |\n"
        "|---|---|---|---|\n"
        + "\n".join(rows)
        + "\n"
    )
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"report -> {REPORT_PATH}")
    return metrics


if __name__ == "__main__":
    main()
