"""Assign empirical ONCE/SPLIT labels and export planner SFT examples.

Reads *saved* teacher retrieval results; never calls a model or retriever.
The target for each question is the selected teacher plan, not retrieval scores.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from adaptive_rag.data.teacher_plans import validate_teacher_plans
from adaptive_rag.evaluation.teacher_retrieval import load_teacher_retrieval

SUPPORTED_METRICS = ("recall_at_k", "mean_recall", "ndcg_at_k", "mean_ndcg", "combined")


def _finite_score(record: Mapping[str, Any], strategy: str, metric: str, k: int) -> float:
    key = f"{strategy}_{metric}_at_{k}"
    if key not in record:
        raise ValueError(f"Missing saved metric {key!r} for {record.get('question_id')}")
    value = float(record[key])
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Invalid {key}={value} for {record.get('question_id')}")
    return value


def strategy_score(record: Mapping[str, Any], strategy: str, *, metric: str = "recall_at_k",
                   k: int = 7, alpha: float = 0.6) -> float:
    """Compute a score solely from persisted Recall@k / NDCG@k metrics.

    combined = alpha * Recall@k + (1-alpha) * NDCG@k.
    mean_* metrics average values over cutoffs 1..k.
    """
    if strategy not in ("once", "split"):
        raise ValueError("strategy must be 'once' or 'split'")
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"metric must be one of {SUPPORTED_METRICS}")
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    if metric == "recall_at_k":
        return _finite_score(record, strategy, "recall", k)
    if metric == "ndcg_at_k":
        return _finite_score(record, strategy, "ndcg", k)
    if metric == "mean_recall":
        return sum(_finite_score(record, strategy, "recall", i) for i in range(1, k + 1)) / k
    if metric == "mean_ndcg":
        return sum(_finite_score(record, strategy, "ndcg", i) for i in range(1, k + 1)) / k
    return (alpha * _finite_score(record, strategy, "recall", k)
            + (1 - alpha) * _finite_score(record, strategy, "ndcg", k))


def label_question(record: Mapping[str, Any], *, metric: str = "recall_at_k", k: int = 7,
                   threshold: float = 0.1, alpha: float = 0.6) -> dict[str, Any]:
    """Prefer ONCE on ties and when SPLIT improvement <= threshold."""
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    for field in ("question_id", "question", "once_plan", "split_plan"):
        if field not in record:
            raise ValueError(f"Missing {field!r}")
    if not str(record["question_id"]).strip() or not str(record["question"]).strip():
        raise ValueError("question_id and question must be nonempty")
    plans = {"once": record["once_plan"], "split": record["split_plan"]}
    validate_teacher_plans(plans)
    once_score = strategy_score(record, "once", metric=metric, k=k, alpha=alpha)
    split_score = strategy_score(record, "split", metric=metric, k=k, alpha=alpha)
    improvement = split_score - once_score
    label = "split" if improvement > threshold + 1e-12 else "once"
    return {
        "question_id": str(record["question_id"]),
        "question": str(record["question"]),
        "source": record.get("source"),
        "strategy": label,
        "once_score": once_score,
        "split_score": split_score,
        "improvement": improvement,
        "metric": metric,
        "k": k,
        "threshold": threshold,
        "alpha": alpha if metric == "combined" else None,
        "selected_plan": {**plans[label], "question": str(record["question"])},
    }


def label_retrieval_results(path: str | Path, *, metric: str = "recall_at_k", k: int = 7,
                            threshold: float = 0.1, alpha: float = 0.6) -> list[dict[str, Any]]:
    """Load completed retrieval records and produce labels without retrieval calls."""
    records = load_teacher_retrieval(path)
    if not records:
        raise ValueError("No saved teacher retrieval records found")
    return [label_question(r, metric=metric, k=k, threshold=threshold, alpha=alpha)
            for r in records]


def _write_jsonl(records: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        with temp.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(dict(record), ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return path


def export_labeled_data(labels: Iterable[Mapping[str, Any]], output_path: str | Path) -> Path:
    """Persist labels and their selected plans as JSONL, without changing the originals."""
    return _write_jsonl(labels, output_path)


def export_sft_examples(labels: Iterable[Mapping[str, Any]], output_path: str | Path) -> Path:
    """Write one question->planner JSON target per record.

    ``target`` is JSON text in the existing prompted planner format. The training
    module/notebook decides how to wrap question and target in a chat template.
    """
    def examples():
        for row in labels:
            plan = row["selected_plan"]
            if plan["strategy"] != row["strategy"]:
                raise ValueError(f"Plan/label mismatch: {row['question_id']}")
            yield {
                "question_id": str(row["question_id"]),
                "question": row["question"],
                "strategy": row["strategy"],
                "target": json.dumps(plan, ensure_ascii=False, separators=(",", ":")),
            }
    return _write_jsonl(examples(), output_path)


def summarize_labels(labels: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(labels)
    n_once = sum(r["strategy"] == "once" for r in rows)
    n_split = sum(r["strategy"] == "split" for r in rows)
    by_source: dict[str, dict[str, int]] = {}
    for row in rows:
        source = str(row.get("source") or "unknown")
        counts = by_source.setdefault(source, {"once": 0, "split": 0})
        counts[row["strategy"]] += 1
    return {"total": len(rows), "once": n_once, "split": n_split,
            "split_fraction": n_split / len(rows) if rows else 0.0,
            "by_source": by_source}


def export_label_summary_parquet(labels: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    """Optional flat table for analysis (nested selected_plan omitted)."""
    rows = [{k: v for k, v in row.items() if k != "selected_plan"} for row in labels]
    if not rows:
        raise ValueError("No labels to export")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        pd.DataFrame(rows).to_parquet(temp, index=False)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return path
