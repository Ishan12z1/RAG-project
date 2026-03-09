from __future__ import annotations

from evaluation.scripts.golden_loader import load_golden_set
from evaluation.scripts.retrieval_evaluator import RetrievalEvaluator
from evaluation.scripts.ladder import append_ladder_row

def run(retriever, golden_path: str, *, run_tag: str, notes: str = ""):
    examples = load_golden_set(golden_path)
    evaluator = RetrievalEvaluator(examples, ks=(1, 3, 5, 10))

    metrics, _ = evaluator.evaluate(
        retriever,
        max_k=10,
        dump_dir=f"eval/runs/{run_tag.replace('+','plus_')}",
        run_tag=run_tag,
        notes=notes,
    )

    append_ladder_row(
        run_tag=run_tag,
        metrics={
            "recall_at_5": metrics.recall_at_k.get(5, 0.0),
            "mrr": metrics.mrr_at_k.get(5, 0.0),
            "p50_ms": metrics.p50_ms,
            "p95_ms": metrics.p95_ms,
        },
        notes=notes,
    )