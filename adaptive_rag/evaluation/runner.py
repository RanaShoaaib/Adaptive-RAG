import hashlib
from functools import partial
from pathlib import Path

from adaptive_rag.planner.adaptive import load_adaptive_planner, generate_adaptive_query

from adaptive_rag.config import (
    ANSWERING_MODEL,
    FINAL_CONTEXT_TOP_K,
    ROOT_DIR,
    SEED,
    SLM_MODEL
)
from adaptive_rag.data.prepare_data import load_subset
from adaptive_rag.database.weaviate_client import endpoint_fingerprint, weaviate_session
from adaptive_rag.embedding.embedder import get_embedding_model
from adaptive_rag.evaluation.checkpoint import append_record, fingerprint, open_checkpoint
from adaptive_rag.evaluation.experiment import (
    COST_KEYS,
    create_experiment_config,
    create_result_record,
    save_experiment_results,
)
from adaptive_rag.generation.generator import generate_answer
from adaptive_rag.models.lm import get_slm
from adaptive_rag.planner.prompted import (
    generate_always_once_query,
    generate_always_split_query,
    generate_query,
)
from adaptive_rag.retrieval.reranker import get_reranker
from adaptive_rag.retrieval.retriever import retrieve_decomposed, retrieve_single


PLANNERS = {
    "prompted": generate_query,
    "always_once": generate_always_once_query,
    "always_split": generate_always_split_query,
    "adaptive": generate_adaptive_query,
    "adaptive_once": partial(generate_adaptive_query, forced_strategy="once"),
    "adaptive_split": partial(generate_adaptive_query, forced_strategy="split"),
}


def run_evaluation(
    experiment_id: str,
    dataset_path: str | Path,
    planner_strategy: str = "prompted",
    *,
    sampling_strategy: str | None = None,
    sampling_seed: int | None = SEED,
    adapter_dir: str | Path | None = None,
    adaptive_use_4bit: bool = False,
    results_dir: str | Path | None = None,
) -> None:
    """Evaluate a fixed subset, resuming compatible checkpoints for the experiment ID."""
    if planner_strategy not in PLANNERS:
        raise ValueError(f"planner_strategy must be one of {list(PLANNERS)}")

    if planner_strategy.startswith("adaptive"):
        if adapter_dir is None or not Path(adapter_dir).is_dir():
            raise ValueError("adaptive requires an existing adapter_dir")
    elif adapter_dir is not None:
        raise ValueError("adapter_dir is only valid for adaptive strategies")

    subset = load_subset(dataset_path)
    if not len(subset):
        raise ValueError("Selected subset is empty")

    experiment_config = create_experiment_config(
        experiment_id=experiment_id,
        dataset_path=dataset_path,
        num_samples=len(subset),
        strategy=planner_strategy,
        sampling_strategy=sampling_strategy,
        sampling_seed=sampling_seed,
    )
    if adapter_dir is not None:
        experiment_config["models"]["planner_adapter"] = str(Path(adapter_dir).resolve())
        experiment_config["models"]["planner_use_4bit"] = adaptive_use_4bit
        adapter_files = {}
        for path in sorted(Path(adapter_dir).iterdir()):
            if path.is_file():
                with path.open("rb") as stream:
                    adapter_files[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
        experiment_config["models"]["planner_adapter_files"] = adapter_files

    experiment_config["retrieval"]["endpoint_sha256"] = endpoint_fingerprint()
    rows = list(subset)
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question IDs in evaluation dataset")
    experiment_config["dataset"]["rows_sha256"] = fingerprint(rows)
    directory = (Path(results_dir) if results_dir is not None else ROOT_DIR / "results") / experiment_id
    checkpoint = directory / "checkpoint.jsonl"
    manifest = directory / "checkpoint_config.json"

    with open_checkpoint(
        checkpoint, manifest, experiment_config, id_key="id",
        legacy_paths=[directory / "config.json", directory / "results.parquet",
                      directory / "results.csv", directory / "summary.json"],
    ) as (experiment_config, completed):
        if set(completed) - set(ids):
            raise ValueError(f"Checkpoint contains questions outside the selected dataset: {checkpoint}")
        pending = [(i, row) for i, (qid, row) in enumerate(zip(ids, rows), start=1)
                   if qid not in completed]
        if completed:
            print(f"Resuming {experiment_id}: {len(completed)}/{len(rows)} questions complete")
        if pending:
            _evaluate_pending(pending, len(rows), PLANNERS[planner_strategy], adapter_dir,
                              adaptive_use_4bit, checkpoint, completed)
        summary = save_experiment_results(
            [completed[qid] for qid in ids], experiment_config, results_dir=results_dir,
        )

    print(f"\nCompleted experiment: {experiment_id}")
    print(f"Summary: {summary}")


def _evaluate_pending(pending, total, planner_fn, adapter_dir, adaptive_use_4bit,
                      checkpoint, completed):
    with weaviate_session() as client:
        embedding_model = get_embedding_model()
        reranker = get_reranker()
        query_model, query_tokenizer = get_slm(SLM_MODEL)
        if ANSWERING_MODEL == SLM_MODEL:
            answering_model, answering_tokenizer = query_model, query_tokenizer
        else:
            answering_model, answering_tokenizer = get_slm(ANSWERING_MODEL)

        if adapter_dir is not None:
            planner_model, planner_tokenizer = load_adaptive_planner(
                adapter_dir, base_model_name=SLM_MODEL, use_4bit=adaptive_use_4bit
            )
        else:
            planner_model, planner_tokenizer = query_model, query_tokenizer

        for i, question_row in pending:
            print(f"\n[{i}/{total}] {question_row['question']}")
            try:
                query_plan = planner_fn(planner_model, planner_tokenizer, question_row["question"])
            except ValueError as exc:
                # A malformed planner response must not abort the test set.
                # Missing metrics remain NaN rather than counting as zero retrieval.
                print(f"Planner failure: {exc}")
                record = dict(question_row)
                record.update({"strategy": "failed", "planner_status": "failed",
                               "planner_error": str(exc), "planner_attempts": None,
                               "planner_queries": None, "retrieved_passage_ids": None,
                               "generated_answer": None})
                for k in range(1, FINAL_CONTEXT_TOP_K + 1):
                    record[f"recall_at_{k}"] = None
                    record[f"ndcg_at_{k}"] = None
                for key in ("answer_em", "answer_f1", *COST_KEYS):
                    record[key] = None
                append_record(checkpoint, record)
                completed[str(record["id"])] = record
                continue

            retrieve_fn = (
                retrieve_single if query_plan["strategy"] == "once" else retrieve_decomposed
            )
            retrieved_docs, cost_stats = retrieve_fn(
                client, embedding_model, reranker,
                query_model, query_tokenizer, query_plan,
            )
            generated_answer = generate_answer(
                answering_model, answering_tokenizer,
                question_row["question"], retrieved_docs,
            )
            record = create_result_record(
                question_row=question_row,
                query_plan=query_plan,
                retrieved_docs=retrieved_docs,
                cost_stats=cost_stats,
                generated_answer=generated_answer,
            )
            record["planner_status"] = "ok"
            record["planner_error"] = None
            record["planner_attempts"] = query_plan.get("planner_attempts")
            append_record(checkpoint, record)
            completed[str(record["id"])] = record
            k = FINAL_CONTEXT_TOP_K
            print(
                f"Strategy: {query_plan['strategy']} | "
                f"Recall@{k}: {record[f'recall_at_{k}']:.3f} | "
                f"NDCG@{k}: {record[f'ndcg_at_{k}']:.3f} | "
                f"Answer F1: {record['answer_f1']:.3f}"
            )
