"""Small, dependency-free test doubles shared by the test modules."""
from __future__ import annotations

import json
from collections import UserDict
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Callable


# --------------------------------------------------------------------------- plans
def query(lex: str = "lex", sem: str = "sem") -> dict[str, str]:
    return {"lexical_query": lex, "semantic_query": sem}


def once_plan(lex: str = "alpha beta", sem: str = "What is alpha beta?") -> dict[str, Any]:
    return {"strategy": "once", "queries": [query(lex, sem)]}


def split_plan() -> dict[str, Any]:
    return {
        "strategy": "split",
        "queries": [query("alpha", "What is alpha?"), query("beta", "What is beta?")],
    }


def teacher_plans() -> dict[str, Any]:
    return {"once": once_plan(), "split": split_plan()}


def plan_json(plan: dict[str, Any]) -> str:
    return json.dumps(plan)


# --------------------------------------------------------------------------- documents
def doc(i: int | str) -> dict[str, str]:
    """A passage whose rendered reranker text ends with its own uuid."""
    return {"uuid": f"u{i}", "title": f"T{i}", "text": f"doc u{i}"}


def docs(*ids: int | str) -> list[dict[str, str]]:
    return [doc(i) for i in ids]


class FakeReranker:
    """CrossEncoder double: ``score_fn(query, passage_text) -> float``."""

    def __init__(self, score_fn: Callable[[str, str], float] | dict[str, float]):
        if isinstance(score_fn, dict):
            table = score_fn
            score_fn = lambda _q, passage: table[passage.split()[-1]]  # noqa: E731
        self.score_fn = score_fn
        self.calls: list[list[list[str]]] = []

    def predict(self, pairs):
        self.calls.append([list(p) for p in pairs])
        return [self.score_fn(q, d) for q, d in pairs]


# --------------------------------------------------------------------------- tokenizer
class FakeBatchEncoding(UserDict):
    """Mimics transformers' BatchEncoding (a dict, *not* a list)."""


class FakeChatTokenizer:
    """Character-level chat tokenizer mimicking transformers >= 5 defaults.

    ``apply_chat_template`` returns a BatchEncoding unless ``return_dict=False``
    (the transformers 5 default). Records every call's keyword arguments.
    """

    eos_token_id = 1
    pad_token_id = 0

    def __init__(self, *, drop_assistant_content: bool = False, mutate_prefix: bool = False):
        self.drop_assistant_content = drop_assistant_content
        self.mutate_prefix = mutate_prefix
        self.calls: list[dict[str, Any]] = []

    def render(self, messages, add_generation_prompt: bool) -> str:
        parts = []
        has_assistant = any(m["role"] == "assistant" for m in messages)
        for m in messages:
            if m["role"] == "assistant" and self.drop_assistant_content:
                parts.append("<|im_start|>assistant\n")  # renders nothing beyond the prompt
                continue
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        text = "".join(parts)
        if self.mutate_prefix and has_assistant:
            text = "<|extra|>" + text  # breaks prefix equality
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text

    def decode(self, ids, skip_special_tokens=False, **kwargs):
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False,
                            return_dict=True, **kwargs):
        self.calls.append({"tokenize": tokenize, "add_generation_prompt": add_generation_prompt,
                           "return_dict": return_dict, **kwargs})
        text = self.render(messages, add_generation_prompt)
        if not tokenize:
            return text
        ids = [ord(ch) for ch in text]
        if return_dict:
            return FakeBatchEncoding({"input_ids": ids, "attention_mask": [1] * len(ids)})
        return ids


# --------------------------------------------------------------------------- OpenAI
class FakeOpenAI:
    """``client.chat.completions.create`` returning queued responses.

    Queue items: a str (message content) or an Exception instance to raise.
    """

    def __init__(self, responses: list[Any]):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=item))])


# --------------------------------------------------------------------------- datasets
class FakeDataset:
    """Minimal stand-in for ``datasets.Dataset`` (rows are plain dicts)."""

    def __init__(self, rows: list[dict[str, Any]], columns: list[str] | None = None):
        self.rows = [dict(r) for r in rows]
        self._columns = list(columns) if columns is not None else (list(self.rows[0]) if self.rows else [])

    @property
    def column_names(self) -> list[str]:
        return list(self._columns)

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self.rows]
        return self.rows[key]

    def select(self, indices):
        return FakeDataset([self.rows[i] for i in indices], self._columns)

    def map(self, fn, batched: bool = False, with_indices: bool = False, **_):
        assert batched, "FakeDataset.map only supports batched=True"
        batch = {c: [row[c] for row in self.rows] for c in self._columns}
        out = fn(batch, list(range(len(self.rows)))) if with_indices else fn(batch)
        new_cols = list(out)
        rows = [{c: out[c][i] for c in new_cols} for i in range(len(self.rows))]
        return FakeDataset(rows, new_cols)


# --------------------------------------------------------------------------- weaviate
class FakeWeaviateCollection:
    def __init__(self, store: "FakeWeaviateClient"):
        self.store = store
        self.batch = SimpleNamespace(failed_objects=[], fixed_size=self._fixed_size)
        self.added: list[dict[str, Any]] = []
        self.aggregate = SimpleNamespace(
            over_all=lambda total_count=True: SimpleNamespace(total_count=store.count)
        )
        self.hybrid_calls: list[dict[str, Any]] = []
        self.query = SimpleNamespace(hybrid=self._hybrid)

    @contextmanager
    def _fixed_size(self, batch_size):
        yield SimpleNamespace(add_object=lambda **kw: self.added.append(kw))

    def _hybrid(self, **kwargs):
        self.hybrid_calls.append(kwargs)
        return SimpleNamespace(objects=self.store.hybrid_objects)


class FakeWeaviateClient:
    def __init__(self, *, exists: bool = False, count: int = 0, hybrid_objects: list[Any] | None = None):
        self.exists_flag = exists
        self.count = count
        self.hybrid_objects = hybrid_objects or []
        self.deleted: list[str] = []
        self.collection = FakeWeaviateCollection(self)
        self.closed = False
        self.collections = SimpleNamespace(
            exists=lambda name: self.exists_flag,
            get=lambda name: self.collection,
            delete=self._delete,
        )

    def _delete(self, name):
        self.deleted.append(name)
        self.exists_flag = False
        self.count = 0

    def close(self):
        self.closed = True
