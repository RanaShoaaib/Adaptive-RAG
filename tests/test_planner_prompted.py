"""Unit tests for adaptive_rag.planner.prompted (validation, JSON retry, planners)."""
import json

import pytest

from adaptive_rag.planner import prompted
from adaptive_rag.planner.prompted import (
    generate_always_once_query,
    generate_always_split_query,
    generate_query,
    generate_valid_json,
    rewrite_query,
    validate_query_plan,
    validate_retrieval_query,
)
from rag_fakes import once_plan, plan_json, query, split_plan


class ScriptedGenerate:
    """Stand-in for models.lm.generate returning queued responses and recording calls."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, model, tokenizer, messages, max_new_tokens=None, **kwargs):
        self.calls.append({"messages": messages, "max_new_tokens": max_new_tokens})
        return self.responses.pop(0)


@pytest.fixture
def script(monkeypatch):
    def install(*responses):
        fake = ScriptedGenerate(*responses)
        monkeypatch.setattr(prompted, "generate", fake)
        return fake
    return install


class TestValidateRetrievalQuery:
    def test_accepts_valid(self):
        validate_retrieval_query(query("a", "b"))

    @pytest.mark.parametrize("bad", [None, [], "text", 3])
    def test_rejects_non_objects(self, bad):
        with pytest.raises(ValueError, match="JSON object"):
            validate_retrieval_query(bad)

    @pytest.mark.parametrize("field", ["lexical_query", "semantic_query"])
    @pytest.mark.parametrize("value", [None, "", "   ", 5])
    def test_rejects_missing_or_blank_fields(self, field, value):
        bad = query()
        bad[field] = value
        with pytest.raises(ValueError, match=field):
            validate_retrieval_query(bad)

    def test_rejects_absent_field(self):
        with pytest.raises(ValueError, match="semantic_query"):
            validate_retrieval_query({"lexical_query": "x"})


class TestValidateQueryPlan:
    def test_accepts_once_and_split(self):
        validate_query_plan(once_plan())
        validate_query_plan(split_plan())

    def test_rejects_non_dict_and_bad_strategy(self):
        with pytest.raises(ValueError):
            validate_query_plan([])
        with pytest.raises(ValueError, match="Strategy"):
            validate_query_plan({"strategy": "many", "queries": [query()]})

    def test_queries_must_be_a_list(self):
        with pytest.raises(ValueError, match="list"):
            validate_query_plan({"strategy": "once", "queries": query()})

    def test_once_requires_exactly_one_query(self):
        with pytest.raises(ValueError, match="ONCE"):
            validate_query_plan({"strategy": "once", "queries": [query(), query()]})
        with pytest.raises(ValueError, match="ONCE"):
            validate_query_plan({"strategy": "once", "queries": []})

    def test_split_requires_exactly_two_queries(self):
        with pytest.raises(ValueError, match="SPLIT"):
            validate_query_plan({"strategy": "split", "queries": [query()]})
        with pytest.raises(ValueError, match="SPLIT"):
            validate_query_plan({"strategy": "split", "queries": [query(), query(), query()]})

    def test_each_query_is_validated(self):
        with pytest.raises(ValueError, match="lexical_query"):
            validate_query_plan({"strategy": "split", "queries": [query(), {"semantic_query": "s"}]})

    def test_context_budget_is_enforced(self, monkeypatch):
        monkeypatch.setattr(prompted, "SUBQUERY_TOP_K", 4)  # 2 queries * 4 > 7
        with pytest.raises(ValueError, match="budget"):
            validate_query_plan(split_plan())


class TestGenerateValidJson:
    def test_valid_first_attempt(self, script):
        fake = script(plan_json(once_plan()))
        result = generate_valid_json(None, None, [{"role": "user", "content": "q"}], validate_query_plan)
        assert result == once_plan()
        assert len(fake.calls) == 1
        assert fake.calls[0]["max_new_tokens"] == prompted.SLM_MAX_NEW_TOKENS

    def test_strips_markdown_fences(self, script):
        script("```json\n" + plan_json(once_plan()) + "\n```")
        assert generate_valid_json(None, None, [], validate_query_plan) == once_plan()

    def test_retries_once_with_corrective_message(self, script):
        fake = script("not json at all", plan_json(split_plan()))
        original = [{"role": "user", "content": "q"}]
        result = generate_valid_json(None, None, original, validate_query_plan)
        assert result == split_plan()
        assert len(fake.calls) == 2
        retry_messages = fake.calls[1]["messages"]
        assert len(retry_messages) == len(original) + 1
        assert retry_messages[-1]["role"] == "user"
        assert "previous response was invalid" in retry_messages[-1]["content"]
        assert original == [{"role": "user", "content": "q"}]  # caller's list not mutated

    def test_schema_violation_also_triggers_retry(self, script):
        bad = json.dumps({"strategy": "once", "queries": [query(), query()]})
        fake = script(bad, plan_json(once_plan()))
        assert generate_valid_json(None, None, [], validate_query_plan) == once_plan()
        assert len(fake.calls) == 2

    def test_raises_after_exhausting_retries(self, script):
        fake = script("nope", "still nope")
        with pytest.raises(ValueError, match="Invalid SLM output after 2 attempts"):
            generate_valid_json(None, None, [], validate_query_plan)
        assert len(fake.calls) == 2

    def test_max_retries_zero_makes_single_attempt(self, script):
        fake = script("nope")
        with pytest.raises(ValueError, match="after 1 attempts"):
            generate_valid_json(None, None, [], validate_query_plan, max_retries=0)
        assert len(fake.calls) == 1


class TestPlanners:
    def test_generate_query_attaches_question_and_prompts_with_it(self, script):
        fake = script(plan_json(split_plan()))
        plan = generate_query(None, None, "Which is older, A or B?")
        assert plan["strategy"] == "split" and plan["question"] == "Which is older, A or B?"
        user_prompt = fake.calls[0]["messages"][-1]["content"]
        assert "Which is older, A or B?" in user_prompt
        assert fake.calls[0]["messages"][0]["role"] == "system"

    def test_rewrite_query_returns_validated_pair(self, script):
        fake = script(json.dumps(query("new lex", "new sem")))
        out = rewrite_query(None, None, query("old lex", "old sem"))
        assert out == query("new lex", "new sem")
        prompt = fake.calls[0]["messages"][-1]["content"]
        assert "old lex" in prompt and "old sem" in prompt

    def test_rewrite_query_rejects_invalid_output(self, script):
        script(json.dumps({"semantic_query": "only one"}), json.dumps({"semantic_query": "still"}))
        with pytest.raises(ValueError):
            rewrite_query(None, None, query())

    def test_always_once_accepts_only_once_plans(self, script):
        script(plan_json(once_plan()))
        assert generate_always_once_query(None, None, "q?")["strategy"] == "once"
        script(plan_json(split_plan()), plan_json(split_plan()))
        with pytest.raises(ValueError):
            generate_always_once_query(None, None, "q?")

    def test_always_split_accepts_only_split_plans(self, script):
        script(plan_json(split_plan()))
        out = generate_always_split_query(None, None, "q?")
        assert out["strategy"] == "split" and out["question"] == "q?"
        script(plan_json(once_plan()), plan_json(once_plan()))
        with pytest.raises(ValueError):
            generate_always_split_query(None, None, "q?")

    def test_always_split_needs_budget_for_two_subqueries(self, script, monkeypatch):
        monkeypatch.setattr(prompted, "FINAL_CONTEXT_TOP_K", 3)  # 3 // 2 = 1 < 2
        fake = script()
        with pytest.raises(ValueError, match="context budget"):
            generate_always_split_query(None, None, "q?")
        assert fake.calls == []  # fails before any generation
