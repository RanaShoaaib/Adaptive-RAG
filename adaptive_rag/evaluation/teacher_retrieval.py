"""Run paired teacher plans through the existing retrieval pipeline; save metrics."""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Mapping
import pandas as pd
from adaptive_rag.config import FINAL_CONTEXT_TOP_K, SLM_MODEL
from adaptive_rag.data.prepare_data import load_subset
from adaptive_rag.data.teacher_plans import load_teacher_plans, validate_teacher_plans
from adaptive_rag.database.weaviate_client import endpoint_fingerprint, weaviate_session
from adaptive_rag.embedding.embedder import get_embedding_model
from adaptive_rag.evaluation.checkpoint import append_record, fingerprint, open_checkpoint
from adaptive_rag.evaluation.experiment import COST_KEYS, create_experiment_config
from adaptive_rag.evaluation.retrieval import recall_at_k, ndcg_at_k
from adaptive_rag.models.lm import get_slm
from adaptive_rag.retrieval.reranker import get_reranker
from adaptive_rag.retrieval.retriever import retrieve_single, retrieve_decomposed


def evaluate_plan_pair(row: Mapping[str, Any], teacher: Mapping[str, Any], *, client: Any,
                       embedding_model: Any, reranker: Any, rewrite_model: Any,
                       rewrite_tokenizer: Any, max_k: int = FINAL_CONTEXT_TOP_K) -> dict:
    """Evaluate both strategies with identical resources and original retrieval logic."""
    if not 1 <= max_k <= FINAL_CONTEXT_TOP_K:
        raise ValueError("max_k exceeds configured final context budget")
    qid, question = str(row['id']), str(row['question'])
    if qid != str(teacher['question_id']) or question != teacher['question']:
        raise ValueError(f"Teacher plan/question mismatch: {qid}")
    gold = list(row['gold_passage_ids'])
    if not gold:
        raise ValueError(f"Missing gold evidence for {qid}")
    validate_teacher_plans({'once': teacher['once'], 'split': teacher['split']})
    result = {'question_id': qid, 'question': question, 'source': row.get('source'),
              'gold_passage_ids': gold, 'teacher_model': teacher.get('model'),
              'once_plan': teacher['once'], 'split_plan': teacher['split']}
    for strategy, retrieve in [('once', retrieve_single), ('split', retrieve_decomposed)]:
        plan = {**teacher[strategy], 'question': question}  # original-question reranking
        docs, costs = retrieve(client, embedding_model, reranker, rewrite_model,
                               rewrite_tokenizer, plan)
        if not set(COST_KEYS).issubset(costs):
            raise ValueError(f"Missing cost counters for {strategy}")
        ids = [str(doc['uuid']) for doc in docs]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate retrieved passages for {qid}, {strategy}")
        result[f'{strategy}_retrieved_ids'] = ids
        result[f'{strategy}_cost'] = {key: int(costs[key]) for key in COST_KEYS}
        for k in range(1, max_k + 1):
            result[f'{strategy}_recall_at_{k}'] = recall_at_k(gold, docs, k)
            result[f'{strategy}_ndcg_at_{k}'] = ndcg_at_k(gold, docs, k)
    return result


def load_teacher_retrieval(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    records, seen = [], set()
    with path.open(encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            qid = str(record['question_id'])
            if qid in seen:
                raise ValueError(f"Duplicate question_id {qid} on line {line_no}")
            seen.add(qid)
            records.append(record)
    return records


def save_teacher_retrieval(record: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    append_record(path, dict(record))


def evaluate_teacher_dataset(dataset_path: str | Path, teacher_plans_path: str | Path,
                             output_path: str | Path, *, max_k: int = FINAL_CONTEXT_TOP_K,
                             rewrite_model_name: str = SLM_MODEL) -> dict[str, int]:
    """Incrementally evaluate pairs; reruns skip completed questions."""
    subset = list(load_subset(dataset_path))
    plans = {str(p['question_id']): p for p in load_teacher_plans(teacher_plans_path)}
    if not 1 <= max_k <= FINAL_CONTEXT_TOP_K:
        raise ValueError("max_k exceeds configured final context budget")
    if not subset:
        raise ValueError('Dataset is empty')
    ids = [str(r['id']) for r in subset]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate dataset IDs')
    for row in subset:
        qid = str(row['id'])
        if qid not in plans:
            raise ValueError(f'Missing teacher plan: {qid}')
        if plans[qid]['question'] != row['question']:
            raise ValueError(f'Teacher question text mismatch: {qid}')
    experiment = create_experiment_config("teacher", dataset_path, len(subset))
    experiment["retrieval"]["endpoint_sha256"] = endpoint_fingerprint()
    config = {
        "rows_sha256": fingerprint(subset),
        "plans_sha256": fingerprint([plans[qid] for qid in ids]),
        "max_k": max_k,
        "rewrite_model": rewrite_model_name,
        "rewrite_max_new_tokens": experiment["generation"]["slm_max_new_tokens"],
        "embedding_model": experiment["models"]["embedding"],
        "reranker": experiment["models"]["reranker"],
        "retrieval": experiment["retrieval"],
    }
    output = Path(output_path)
    manifest = output.with_suffix(output.suffix + ".config.json")
    with open_checkpoint(output, manifest, config, id_key="question_id") as (_, previous):
        if set(previous) - set(ids):
            raise ValueError(f"Checkpoint contains questions outside the selected dataset: {output}")
        return _evaluate_pending(subset, plans, previous, output, max_k, rewrite_model_name)


def _evaluate_pending(subset, plans, previous, output_path, max_k, rewrite_model_name):
    pending = [r for r in subset if str(r['id']) not in previous]
    counts = {'evaluated': 0, 'skipped': len(subset) - len(pending)}
    if not pending:
        return counts
    with weaviate_session() as client:
        embedding_model = get_embedding_model()
        reranker = get_reranker()
        model, tokenizer = get_slm(rewrite_model_name)
        for i, row in enumerate(pending, 1):
            result = evaluate_plan_pair(row, plans[str(row['id'])], client=client,
                embedding_model=embedding_model, reranker=reranker,
                rewrite_model=model, rewrite_tokenizer=tokenizer, max_k=max_k)
            save_teacher_retrieval(result, output_path)
            counts['evaluated'] += 1
            print(f"[{i}/{len(pending)}] {row['id']} complete")
    return counts


def export_teacher_retrieval_parquet(jsonl_path: str | Path, parquet_path: str | Path) -> Path:
    """Export saved retrieval results without any new retrieval calls."""
    records = load_teacher_retrieval(jsonl_path)
    if not records:
        raise ValueError('No retrieval records')
    path = Path(parquet_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    try:
        pd.DataFrame.from_records(records).to_parquet(temp, index=False)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return path
