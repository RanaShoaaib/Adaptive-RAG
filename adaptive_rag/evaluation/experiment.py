"""Record, summarize, and persist Adaptive RAG experiments.

This module does not run retrieval or generation. It records outputs from the
existing pipeline and saves one directory per experiment.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from adaptive_rag import config as cfg
from adaptive_rag.evaluation.answer import evaluate_answer
from adaptive_rag.evaluation.retrieval import ndcg_at_k, recall_at_k

COST_KEYS = (
    "search_calls",
    "embedding_calls",
    "reranker_calls",
    "reranked_pairs",
    "rewrite_calls",
)


def create_experiment_config(
    experiment_id: str,
    dataset_path: str | Path,
    num_samples: int,
    strategy: str = "prompted",
    *,
    sampling_strategy: str | None = None,
    sampling_seed: int | None = None,
) -> dict[str, Any]:
    """Snapshot experiment settings from adaptive_rag.config and dataset metadata.

    The sampling fields describe how the *saved* subset was created; they are
    optional because the runner cannot infer them from a Parquet file.
    """
    if not experiment_id or Path(experiment_id).name != experiment_id or experiment_id in {".", ".."}:
        raise ValueError("experiment_id must be a non-empty directory name")
    if num_samples < 1:
        raise ValueError("num_samples must be positive")

    path = Path(dataset_path)
    try:
        dataset_display_path = str(path.resolve().relative_to(cfg.ROOT_DIR.resolve()))
    except ValueError:
        dataset_display_path = str(path)

    return {
        "experiment_id": experiment_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "strategy": strategy,
        "dataset": {
            "path": dataset_display_path,
            "num_samples": num_samples,
            "sampling_strategy": sampling_strategy,
            "sampling_seed": sampling_seed,
            "validation_fraction": cfg.VAL_SIZE,
            "sources": {
                "hotpotqa": {"path": cfg.HOTPOTQA_DATASET_PATH, "name": cfg.HOTPOTQA_DATASET_NAME},
                "squad": {"path": cfg.SQUAD_DATASET_PATH},
            },
        },
        "models": {
            "planner": cfg.SLM_MODEL,
            "answering": cfg.ANSWERING_MODEL,
            "embedding": cfg.EMBEDDING_MODEL,
            "reranker": cfg.RERANKING_MODEL,
        },
        "retrieval": {
            "collection_name": cfg.COLLECTION_NAME,
            "hybrid_search_top_k": cfg.HYBRID_SEARCH_TOP_K,
            "hybrid_search_alpha": cfg.HYBRID_SEARCH_ALPHA,
            "relevance_threshold": cfg.RELEVANCE_THRESHOLD,
            "subquery_top_k": cfg.SUBQUERY_TOP_K,
            "final_context_top_k": cfg.FINAL_CONTEXT_TOP_K,
            "query_prefix": cfg.QUERY_PREFIX,
            "passage_prefix": cfg.PASSAGE_PREFIX,
        },
        "generation": {
            "slm_max_new_tokens": cfg.SLM_MAX_NEW_TOKENS,
            "answering_max_new_tokens": cfg.ANSWERING_MAX_NEW_TOKENS,
            "do_sample": False,
        },
    }


def create_result_record(
    question_row: Mapping[str, Any],
    query_plan: Mapping[str, Any],
    retrieved_docs: Sequence[Mapping[str, Any]],
    cost_stats: Mapping[str, int],
    generated_answer: str,
    *,
    max_k: int | None = None,
) -> dict[str, Any]:
    """Combine original question data and one completed pipeline evaluation."""
    if max_k is None:
        max_k = cfg.FINAL_CONTEXT_TOP_K
    if max_k < 1:
        raise ValueError("max_k must be positive")

    record = dict(question_row)
    if "gold_passage_ids" not in record or "answers" not in record:
        raise ValueError("question_row must contain gold_passage_ids and answers")
    if query_plan.get("strategy") not in {"once", "split"}:
        raise ValueError("query_plan must contain strategy 'once' or 'split'")

    record["strategy"] = query_plan["strategy"]
    record["planner_queries"] = query_plan["queries"]
    record["retrieved_passage_ids"] = [doc["uuid"] for doc in retrieved_docs]

    for k in range(1, max_k + 1):
        record[f"recall_at_{k}"] = recall_at_k(record["gold_passage_ids"], list(retrieved_docs), k)
        record[f"ndcg_at_{k}"] = ndcg_at_k(record["gold_passage_ids"], list(retrieved_docs), k)

    for key in COST_KEYS:
        if key not in cost_stats:
            raise ValueError(f"Missing cost counter: {key}")
        record[key] = int(cost_stats[key])

    answer_scores = evaluate_answer(generated_answer, record["answers"])
    record["generated_answer"] = generated_answer
    record["answer_em"] = answer_scores["exact_match"]
    record["answer_f1"] = answer_scores["f1"]
    return record


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_experiment_results(
    records: Sequence[Mapping[str, Any]],
    experiment_config: Mapping[str, Any],
    *,
    results_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Save config.json, results.parquet, results.csv, and summary.json for one experiment.

    Calling again with a larger list of records overwrites the prior snapshot.
    Supply all completed records each time, not only the latest record.
    """
    if not records:
        raise ValueError("At least one completed record is required")

    experiment_id = str(experiment_config["experiment_id"])
    if not experiment_id or Path(experiment_id).name != experiment_id or experiment_id in {".", ".."}:
        raise ValueError("Invalid experiment_id")
    base = Path(results_dir) if results_dir is not None else cfg.ROOT_DIR / "results"
    directory = base / experiment_id
    directory.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame.from_records(records)
    if "id" in df.columns and df["id"].duplicated().any():
        raise ValueError("Duplicate question IDs in experiment results")

    metric_columns = [
        col for col in df.columns
        if col in {"answer_em", "answer_f1", *COST_KEYS}
        or col.startswith("recall_at_")
        or col.startswith("ndcg_at_")
    ]
    summary = {
        "experiment_id": experiment_id,
        "num_examples": len(df),
        "planned_num_examples": experiment_config.get("dataset", {}).get("num_samples"),
        "metrics_mean": {col: float(df[col].mean()) for col in metric_columns},
        "planner_failures": int((df["planner_status"] == "failed").sum()) if "planner_status" in df else 0,
        "planner_success_rate": float((df["planner_status"] == "ok").mean()) if "planner_status" in df else 1.0,
        "strategy_counts": {str(k): int(v) for k, v in df["strategy"].value_counts().items()},
        "source_counts": {str(k): int(v) for k, v in df["source"].value_counts().items()},
    }

    # Replace the Parquet snapshot only after the new file is fully written.
    parquet_path = directory / "results.parquet"
    temporary_parquet = directory / "results.parquet.tmp"
    try:
        df.to_parquet(temporary_parquet, index=False)
        os.replace(temporary_parquet, parquet_path)
    finally:
        temporary_parquet.unlink(missing_ok=True)

    # Save CSV for easy inspection in PyCharm or Excel.
    csv_path = directory / "results.csv"
    temporary_csv = directory / "results.csv.tmp"
    try:
        df.to_csv(temporary_csv, index=False)
        os.replace(temporary_csv, csv_path)
    finally:
        temporary_csv.unlink(missing_ok=True)

    _write_json_atomic(directory / "config.json", experiment_config)
    _write_json_atomic(directory / "summary.json", summary)
    return summary
