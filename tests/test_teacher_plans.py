"""Unit tests for adaptive_rag.data.teacher_plans (teacher output validation and I/O)."""
import copy
import json

import pytest

from adaptive_rag.data import teacher_plans as tp
from adaptive_rag.data.teacher_plans import (
    RESPONSE_SCHEMA,
    generate_dataset_plans,
    generate_teacher_plans,
    load_teacher_plans,
    save_teacher_plan,
    validate_teacher_plans,
)
from rag_fakes import FakeOpenAI, once_plan, query, split_plan, teacher_plans


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(tp.time, "sleep", lambda _seconds: None)


def reply(plans=None):
    return json.dumps(plans or teacher_plans())


def record(qid="q1", question="Who?", **extra):
    return {"question_id": qid, "question": question, "model": "m", **teacher_plans(), **extra}


class TestResponseSchema:
    def walk(self, node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                yield node
            for value in node.values():
                yield from self.walk(value)
        elif isinstance(node, list):
            for item in node:
                yield from self.walk(item)

    def test_is_valid_for_openai_strict_structured_outputs(self):
        """Strict mode needs additionalProperties=false and every property required."""
        objects = list(self.walk(RESPONSE_SCHEMA))
        assert len(objects) >= 4
        for obj in objects:
            assert obj["additionalProperties"] is False
            assert sorted(obj["required"]) == sorted(obj["properties"])

    def test_top_level_requires_both_plans(self):
        assert RESPONSE_SCHEMA["required"] == ["once", "split"]


class TestValidateTeacherPlans:
    def test_accepts_valid_pair(self):
        validate_teacher_plans(teacher_plans())

    @pytest.mark.parametrize("bad", [None, [], {}, {"once": once_plan()},
                                     {"once": once_plan(), "split": split_plan(), "extra": 1}])
    def test_requires_exactly_once_and_split_keys(self, bad):
        with pytest.raises(ValueError, match="exactly"):
            validate_teacher_plans(bad)

    def test_plan_shape_and_strategy_labels(self):
        plans = teacher_plans()
        plans["once"]["extra"] = 1
        with pytest.raises(ValueError, match="only strategy and queries"):
            validate_teacher_plans(plans)
        plans = teacher_plans()
        plans["once"]["strategy"] = "split"
        with pytest.raises(ValueError, match="Expected strategy"):
            validate_teacher_plans(plans)

    def test_query_counts_follow_the_planner_contract(self):
        plans = teacher_plans()
        plans["split"]["queries"] = plans["split"]["queries"][:1]
        with pytest.raises(ValueError, match="SPLIT"):
            validate_teacher_plans(plans)

    def test_rejects_unexpected_query_fields(self):
        plans = teacher_plans()
        plans["once"]["queries"][0]["note"] = "x"
        with pytest.raises(ValueError, match="Unexpected query fields"):
            validate_teacher_plans(plans)

    def test_split_queries_must_differ(self):
        plans = teacher_plans()
        plans["split"]["queries"] = [query("a", "b"), query("a", "b")]
        with pytest.raises(ValueError, match="identical"):
            validate_teacher_plans(plans)


class TestGenerateTeacherPlans:
    def test_returns_validated_plans_and_calls_api_with_schema(self):
        client = FakeOpenAI([reply()])
        plans = generate_teacher_plans("Who wrote it?", client=client, model="gpt-x")
        assert plans == teacher_plans()
        (call,) = client.calls
        assert call["model"] == "gpt-x"
        assert call["response_format"]["type"] == "json_schema"
        assert call["response_format"]["json_schema"]["strict"] is True
        user = [m for m in call["messages"] if m["role"] == "user"][0]["content"]
        assert user == "Question: Who wrote it?"

    def test_retries_malformed_then_succeeds(self):
        client = FakeOpenAI(["not json", "", reply()])
        assert generate_teacher_plans("q?", client=client, model="m", max_retries=2) == teacher_plans()
        assert len(client.calls) == 3

    def test_schema_invalid_output_is_retried(self):
        bad = reply({"once": once_plan(), "split": once_plan()})
        client = FakeOpenAI([bad, reply()])
        generate_teacher_plans("q?", client=client, model="m", max_retries=1)
        assert len(client.calls) == 2

    def test_raises_after_all_attempts_fail(self):
        client = FakeOpenAI(["x", "y", "z"])
        with pytest.raises(ValueError, match="after 3 attempts"):
            generate_teacher_plans("q?", client=client, model="m", max_retries=2)
        assert len(client.calls) == 3

    @pytest.mark.parametrize("bad", ["", "  ", None])
    def test_requires_a_question(self, bad):
        client = FakeOpenAI([])
        with pytest.raises(ValueError, match="non-empty"):
            generate_teacher_plans(bad, client=client, model="m")
        assert client.calls == []


class TestPersistence:
    def test_missing_file_loads_as_empty(self, tmp_path):
        assert load_teacher_plans(tmp_path / "none.jsonl") == []

    def test_roundtrip_creates_parent_directories(self, tmp_path):
        path = tmp_path / "nested" / "plans.jsonl"
        save_teacher_plan(record("q1"), path)
        save_teacher_plan(record("q2", question="Other?"), path)
        loaded = load_teacher_plans(path)
        assert [r["question_id"] for r in loaded] == ["q1", "q2"]

    def test_save_validates_before_writing(self, tmp_path):
        bad = record()
        bad["split"]["queries"] = bad["split"]["queries"][:1]
        with pytest.raises(ValueError):
            save_teacher_plan(bad, tmp_path / "p.jsonl")
        assert not (tmp_path / "p.jsonl").exists()
        with pytest.raises(ValueError, match="question_id and question"):
            save_teacher_plan(record(question=" "), tmp_path / "p.jsonl")

    def test_load_rejects_duplicates_and_malformed_records(self, tmp_path):
        path = tmp_path / "p.jsonl"
        path.write_text(json.dumps(record("q1")) + "\n" + json.dumps(record("q1")) + "\n")
        with pytest.raises(ValueError, match="Duplicate question_id"):
            load_teacher_plans(path)
        path.write_text(json.dumps({"question_id": "q1"}) + "\n")
        with pytest.raises(ValueError, match="Malformed record at line 1"):
            load_teacher_plans(path)


class TestGenerateDatasetPlans:
    questions = [{"question_id": "q1", "question": "A?", "dataset": "squad"},
                 {"question_id": "q2", "question": "B?"}]

    def test_generates_and_persists_with_metadata(self, tmp_path):
        path = tmp_path / "p.jsonl"
        client = FakeOpenAI([reply(), reply()])
        counts = generate_dataset_plans(self.questions, path, model="m", client=client)
        assert counts == {"generated": 2, "skipped": 0}
        first, second = load_teacher_plans(path)
        assert first["model"] == "m" and first["dataset"] == "squad"
        assert "dataset" not in second

    def test_rerun_skips_completed_questions_without_api_calls(self, tmp_path):
        path = tmp_path / "p.jsonl"
        generate_dataset_plans(self.questions, path, model="m", client=FakeOpenAI([reply(), reply()]))
        client = FakeOpenAI([])
        counts = generate_dataset_plans(self.questions, path, model="m", client=client)
        assert counts == {"generated": 0, "skipped": 2}
        assert client.calls == []

    def test_resumes_after_partial_progress(self, tmp_path):
        path = tmp_path / "p.jsonl"
        save_teacher_plan(record("q1", question="A?"), path)
        client = FakeOpenAI([reply()])
        counts = generate_dataset_plans(self.questions, path, model="m", client=client)
        assert counts == {"generated": 1, "skipped": 1}

    def test_changed_question_text_for_a_known_id_is_an_error(self, tmp_path):
        path = tmp_path / "p.jsonl"
        save_teacher_plan(record("q1", question="OLD?"), path)
        with pytest.raises(ValueError, match="changed text"):
            generate_dataset_plans(self.questions, path, model="m", client=FakeOpenAI([]))

    def test_fails_fast_but_keeps_earlier_records(self, tmp_path):
        path = tmp_path / "p.jsonl"
        client = FakeOpenAI([reply(), "broken"])
        with pytest.raises(ValueError, match="Invalid teacher output"):
            generate_dataset_plans(self.questions, path, model="m", client=client, max_retries=0)
        assert [r["question_id"] for r in load_teacher_plans(path)] == ["q1"]
