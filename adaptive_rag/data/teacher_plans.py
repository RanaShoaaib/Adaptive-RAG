"""Generate, validate, and persist paired ONCE/SPLIT teacher retrieval plans.

Requires the optional ``openai`` package and OPENAI_API_KEY. No API calls on import.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

from adaptive_rag.planner.prompted import validate_query_plan


SYSTEM_PROMPT = """You generate retrieval plans for a hybrid BM25/vector search system.
For EACH question, produce both an ONCE and a SPLIT plan, regardless of which
strategy you personally prefer. Do not answer the question or invent facts.
ONCE: exactly one lexical_query and semantic_query pair covering the question.
SPLIT: exactly two distinct, complementary pairs. For multi-hop questions,
express independently retrievable information needs without presuming an unknown
intermediate answer. For simple questions, use complementary perspectives rather
than fabricated reasoning hops. lexical_query should contain concise search
keywords/entities; semantic_query should express a natural-language information
need. Preserve the meaning of the question. Return only the requested JSON.
"""

# The wrapper is for teacher generation only; each nested plan follows prompted.py.
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "once": {"$ref": "#/$defs/once"},
        "split": {"$ref": "#/$defs/split"},
    },
    "required": ["once", "split"],
    "additionalProperties": False,
    "$defs": {
        "query": {
            "type": "object",
            "properties": {
                "lexical_query": {"type": "string"},
                "semantic_query": {"type": "string"},
            },
            "required": ["lexical_query", "semantic_query"],
            "additionalProperties": False,
        },
        "once": {
            "type": "object",
            "properties": {
                "strategy": {"type": "string", "enum": ["once"]},
                "queries": {"type": "array", "items": {"$ref": "#/$defs/query"}},
            },
            "required": ["strategy", "queries"],
            "additionalProperties": False,
        },
        "split": {
            "type": "object",
            "properties": {
                "strategy": {"type": "string", "enum": ["split"]},
                "queries": {"type": "array", "items": {"$ref": "#/$defs/query"}},
            },
            "required": ["strategy", "queries"],
            "additionalProperties": False,
        },
    },
}


def validate_teacher_plans(plans: dict[str, Any]) -> None:
    """Check teacher wrapper, exact query counts, and existing planner contract."""
    if not isinstance(plans, dict) or set(plans) != {"once", "split"}:
        raise ValueError("Teacher output must have exactly 'once' and 'split' keys")
    for strategy in ("once", "split"):
        plan = plans[strategy]
        if not isinstance(plan, dict) or set(plan) != {"strategy", "queries"}:
            raise ValueError(f"{strategy} plan must have only strategy and queries")
        if plan.get("strategy") != strategy:
            raise ValueError(f"Expected strategy={strategy!r}")
        validate_query_plan(plan)
        for query in plan["queries"]:
            if set(query) != {"lexical_query", "semantic_query"}:
                raise ValueError("Unexpected query fields")
    if plans["split"]["queries"][0] == plans["split"]["queries"][1]:
        raise ValueError("SPLIT queries must not be identical")


def generate_teacher_plans(
    question: str,
    *,
    client: Any = None,
    model: str,
    max_retries: int = 2,
) -> dict[str, Any]:
    """Generate one paired response with OpenAI Chat Completions structured outputs.

    Model must support json_schema response_format; pass a verified API model ID.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if client is None:
        from openai import OpenAI
        client = OpenAI()
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Question: {question}"},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "paired_retrieval_plans",
                        "strict": True,
                        "schema": RESPONSE_SCHEMA,
                    },
                },
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty teacher response")
            plans = json.loads(content)
            validate_teacher_plans(plans)
            return plans
        except (ValueError, json.JSONDecodeError, IndexError, AttributeError) as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(min(2**attempt, 4))
    raise ValueError(f"Invalid teacher output after {max_retries + 1} attempts: {last_error}")


def load_teacher_plans(path: str | Path) -> list[dict[str, Any]]:
    """Load validated JSONL records; reject duplicate question IDs."""
    path = Path(path)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not all(k in record for k in ("question_id", "question", "once", "split")):
                raise ValueError(f"Malformed record at line {line_number}")
            validate_teacher_plans({"once": record["once"], "split": record["split"]})
            qid = str(record["question_id"])
            if qid in seen:
                raise ValueError(f"Duplicate question_id {qid!r} at line {line_number}")
            seen.add(qid)
            records.append(record)
    return records


def save_teacher_plan(record: dict[str, Any], path: str | Path) -> None:
    """Append a validated record; call through generate_dataset_plans for resumption."""
    validate_teacher_plans({"once": record["once"], "split": record["split"]})
    if not str(record.get("question_id", "")).strip() or not str(record.get("question", "")).strip():
        raise ValueError("question_id and question are required")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def generate_dataset_plans(
    questions: Iterable[dict[str, Any]],
    output_path: str | Path,
    *,
    model: str,
    client: Any = None,
    max_retries: int = 2,
) -> dict[str, int]:
    """Generate and append plans, skipping previously completed question IDs.

    Input records must contain question_id and question; dataset is optional.
    Fail fast on generation errors, preserving earlier successful records.
    """
    existing = load_teacher_plans(output_path)
    completed = {str(row["question_id"]): row["question"] for row in existing}
    counts = {"generated": 0, "skipped": 0}
    for row in questions:
        qid = str(row["question_id"])
        question = str(row["question"])
        if qid in completed:
            if completed[qid] != question:
                raise ValueError(f"Question ID {qid!r} has changed text")
            counts["skipped"] += 1
            continue
        plans = generate_teacher_plans(question, client=client, model=model, max_retries=max_retries)
        record = {"question_id": qid, "question": question, "model": model, **plans}
        if "dataset" in row:
            record["dataset"] = str(row["dataset"])
        save_teacher_plan(record, output_path)
        completed[qid] = question
        counts["generated"] += 1
    return counts
