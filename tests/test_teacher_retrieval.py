"""Unit tests for adaptive_rag.evaluation.teacher_retrieval (paired ONCE/SPLIT runs)."""
import contextlib
import json

import pytest

from adaptive_rag import config as cfg
from adaptive_rag.data.teacher_plans import save_teacher_plan
from adaptive_rag.evaluation import teacher_retrieval as tr
from adaptive_rag.evaluation.experiment import COST_KEYS
from rag_fakes import doc, teacher_plans

COSTS = {key: 2 for key in COST_KEYS}


def row(qid="q1", gold=("u1",), question="Who?", source="squad"):
    return {"id": qid, "question": question, "source": source, "gold_passage_ids": list(gold)}


def teacher(qid="q1", question="Who?"):
    return {"question_id": qid, "question": question, "model": "teacher-m", **teacher_plans()}


class FakeRetrievers:
    """Records calls to retrieve_single / retrieve_decomposed."""

    def __init__(self, once_ids=("u1", "u2"), split_ids=("u3", "u1"), costs=None):
        self.once_ids, self.split_ids = once_ids, split_ids
        self.costs = costs if costs is not None else COSTS
        self.calls = []

    def single(self, *args):
        self.calls.append(("once", args))
        return [doc(i[1:]) for i in self.once_ids], dict(self.costs)

    def decomposed(self, *args):
        self.calls.append(("split", args))
        return [doc(i[1:]) for i in self.split_ids], dict(self.costs)


@pytest.fixture
def retrievers(monkeypatch):
    fake = FakeRetrievers()
    monkeypatch.setattr(tr, "retrieve_single", fake.single)
    monkeypatch.setattr(tr, "retrieve_decomposed", fake.decomposed)
    return fake


def evaluate(retrievers_unused=None, **overrides):
    kwargs = dict(client="C", embedding_model="E", reranker="R", rewrite_model="M", rewrite_tokenizer="T")
    kwargs.update(overrides)
    return tr.evaluate_plan_pair(overrides.pop("row", None) or row(), overrides.pop("teacher", None) or teacher(), **{
        k: v for k, v in kwargs.items() if k not in ("row", "teacher")})


class TestEvaluatePlanPair:
    def test_scores_both_strategies_against_gold(self, retrievers):
        result = tr.evaluate_plan_pair(row(), teacher(), client="C", embedding_model="E",
                                       reranker="R", rewrite_model="M", rewrite_tokenizer="T")
        assert result["question_id"] == "q1" and result["teacher_model"] == "teacher-m"
        assert result["once_retrieved_ids"] == ["u1", "u2"]
        assert result["split_retrieved_ids"] == ["u3", "u1"]
        assert result["once_recall_at_1"] == 1.0
        assert result["split_recall_at_1"] == 0.0 and result["split_recall_at_2"] == 1.0
        assert result["once_ndcg_at_1"] == pytest.approx(1.0)
        assert result["once_cost"] == {k: 2 for k in COST_KEYS}
        assert f"once_recall_at_{cfg.FINAL_CONTEXT_TOP_K}" in result
        assert result["once_plan"] == teacher_plans()["once"]

    def test_retrievers_receive_shared_resources_and_the_original_question(self, retrievers):
        tr.evaluate_plan_pair(row(question="Original?"), teacher(question="Original?"), client="C",
                              embedding_model="E", reranker="R", rewrite_model="M", rewrite_tokenizer="T")
        assert [name for name, _ in retrievers.calls] == ["once", "split"]
        for name, args in retrievers.calls:
            assert args[:5] == ("C", "E", "R", "M", "T")
            plan = args[5]
            assert plan["strategy"] == name and plan["question"] == "Original?"

    def test_max_k_limits_recorded_cutoffs(self, retrievers):
        result = tr.evaluate_plan_pair(row(), teacher(), client=1, embedding_model=1, reranker=1,
                                       rewrite_model=1, rewrite_tokenizer=1, max_k=2)
        assert "once_recall_at_2" in result and "once_recall_at_3" not in result

    @pytest.mark.parametrize("max_k", [0, cfg.FINAL_CONTEXT_TOP_K + 1])
    def test_max_k_must_fit_the_context_budget(self, retrievers, max_k):
        with pytest.raises(ValueError, match="max_k"):
            tr.evaluate_plan_pair(row(), teacher(), client=1, embedding_model=1, reranker=1,
                                  rewrite_model=1, rewrite_tokenizer=1, max_k=max_k)

    def call(self, r, t):
        return tr.evaluate_plan_pair(r, t, client=1, embedding_model=1, reranker=1,
                                     rewrite_model=1, rewrite_tokenizer=1)

    def test_plan_must_belong_to_the_question(self, retrievers):
        with pytest.raises(ValueError, match="mismatch"):
            self.call(row("q1"), teacher("q2"))
        with pytest.raises(ValueError, match="mismatch"):
            self.call(row(question="A?"), teacher(question="B?"))

    def test_gold_evidence_is_required(self, retrievers):
        with pytest.raises(ValueError, match="Missing gold"):
            self.call(row(gold=()), teacher())

    def test_rejects_invalid_teacher_plans(self, retrievers):
        bad = teacher()
        bad["split"] = bad["once"]
        with pytest.raises(ValueError):
            self.call(row(), bad)

    def test_duplicate_retrieved_passages_are_an_error(self, monkeypatch):
        fake = FakeRetrievers(once_ids=("u1", "u1"))
        monkeypatch.setattr(tr, "retrieve_single", fake.single)
        monkeypatch.setattr(tr, "retrieve_decomposed", fake.decomposed)
        with pytest.raises(ValueError, match="Duplicate retrieved"):
            self.call(row(), teacher())

    def test_missing_cost_counters_are_an_error(self, monkeypatch):
        fake = FakeRetrievers(costs={"search_calls": 1})
        monkeypatch.setattr(tr, "retrieve_single", fake.single)
        monkeypatch.setattr(tr, "retrieve_decomposed", fake.decomposed)
        with pytest.raises(ValueError, match="cost counters"):
            self.call(row(), teacher())


class TestJsonlStorage:
    def test_missing_file_loads_as_empty(self, tmp_path):
        assert tr.load_teacher_retrieval(tmp_path / "x.jsonl") == []

    def test_save_load_roundtrip_with_unicode(self, tmp_path):
        path = tmp_path / "deep" / "r.jsonl"
        tr.save_teacher_retrieval({"question_id": "q1", "question": "Zürich?"}, path)
        tr.save_teacher_retrieval({"question_id": "q2", "question": "B?"}, path)
        loaded = tr.load_teacher_retrieval(path)
        assert [r["question_id"] for r in loaded] == ["q1", "q2"]
        assert loaded[0]["question"] == "Zürich?"

    def test_duplicate_question_ids_are_rejected(self, tmp_path):
        path = tmp_path / "r.jsonl"
        path.write_text(json.dumps({"question_id": "q1"}) + "\n" + json.dumps({"question_id": "q1"}) + "\n")
        with pytest.raises(ValueError, match="Duplicate question_id"):
            tr.load_teacher_retrieval(path)

    def test_parquet_export_needs_records(self, tmp_path):
        with pytest.raises(ValueError, match="No retrieval records"):
            tr.export_teacher_retrieval_parquet(tmp_path / "none.jsonl", tmp_path / "o.parquet")

    def test_parquet_export_roundtrip(self, tmp_path):
        pd = pytest.importorskip("pandas")
        pytest.importorskip("pyarrow")
        src = tmp_path / "r.jsonl"
        tr.save_teacher_retrieval({"question_id": "q1", "once_recall_at_1": 1.0}, src)
        out = tr.export_teacher_retrieval_parquet(src, tmp_path / "o" / "r.parquet")
        assert pd.read_parquet(out)["question_id"].tolist() == ["q1"]
        assert not list(out.parent.glob("*.tmp"))


class TestEvaluateTeacherDataset:
    @pytest.fixture
    def env(self, tmp_path, monkeypatch, retrievers):
        rows = [row("q1", question="A?"), row("q2", gold=("u3",), question="B?")]
        monkeypatch.setattr(tr, "load_subset", lambda path: rows)
        sessions = []

        @contextlib.contextmanager
        def session():
            sessions.append(1)
            yield "client"

        monkeypatch.setattr(tr, "weaviate_session", session)
        monkeypatch.setattr(tr, "endpoint_fingerprint", lambda: "test-endpoint")
        monkeypatch.setattr(tr, "get_embedding_model", lambda: "embedder")
        monkeypatch.setattr(tr, "get_reranker", lambda: "reranker")
        monkeypatch.setattr(tr, "get_slm", lambda name: ("model", "tokenizer"))
        plans = tmp_path / "plans.jsonl"
        for qid, question in (("q1", "A?"), ("q2", "B?")):
            save_teacher_plan({"question_id": qid, "question": question, "model": "m", **teacher_plans()}, plans)
        return {"rows": rows, "plans": plans, "out": tmp_path / "out.jsonl", "sessions": sessions}

    def run(self, env):
        return tr.evaluate_teacher_dataset("ignored.parquet", env["plans"], env["out"])

    def test_evaluates_all_pending_questions(self, env):
        assert self.run(env) == {"evaluated": 2, "skipped": 0}
        assert [r["question_id"] for r in tr.load_teacher_retrieval(env["out"])] == ["q1", "q2"]

    def test_rerun_skips_completed_work_and_never_opens_a_session(self, env):
        self.run(env)
        env["sessions"].clear()
        assert self.run(env) == {"evaluated": 0, "skipped": 2}
        assert env["sessions"] == []

    def test_resumes_remaining_questions(self, env):
        self.run(env)
        first = env["out"].read_text(encoding="utf-8").splitlines()[0]
        env["out"].write_text(first + "\n", encoding="utf-8")
        assert self.run(env) == {"evaluated": 1, "skipped": 1}

    def test_every_question_needs_a_teacher_plan(self, env, tmp_path):
        env["plans"].write_text(env["plans"].read_text().splitlines()[0] + "\n")
        with pytest.raises(ValueError, match="Missing teacher plan: q2"):
            self.run(env)

    def test_teacher_question_text_must_match(self, env, monkeypatch):
        env["rows"][0]["question"] = "CHANGED?"
        with pytest.raises(ValueError, match="text mismatch"):
            self.run(env)

    def test_rejects_legacy_output_without_metadata(self, env):
        tr.save_teacher_retrieval({"question_id": "q1", "question": "A?"}, env["out"])
        with pytest.raises(ValueError, match="no configuration"):
            self.run(env)

    def test_rejects_changed_teacher_plans(self, env):
        self.run(env)
        plans = tr.load_teacher_plans(env["plans"])
        plans[0]["once"]["queries"][0]["lexical_query"] = "updated terms"
        env["plans"].write_text(
            "".join(json.dumps(plan) + "\n" for plan in plans), encoding="utf-8",
        )
        with pytest.raises(ValueError, match="configuration mismatch"):
            self.run(env)

    def test_duplicate_dataset_ids_are_rejected(self, env, monkeypatch):
        monkeypatch.setattr(tr, "load_subset", lambda path: [row("q1", question="A?"), row("q1", question="A?")])
        with pytest.raises(ValueError, match="Duplicate dataset IDs"):
            self.run(env)

    def test_empty_dataset_is_rejected(self, env, monkeypatch):
        monkeypatch.setattr(tr, "load_subset", lambda path: [])
        with pytest.raises(ValueError, match="empty"):
            self.run(env)
