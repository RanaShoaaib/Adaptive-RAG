"""Unit tests for adaptive_rag.evaluation.runner (orchestration, planner-failure handling)."""
import contextlib
import json
from functools import partial

import pytest

pytest.importorskip("pyarrow")

from adaptive_rag import config as cfg
from adaptive_rag.evaluation import runner
from rag_fakes import docs, once_plan, split_plan


def rows(n=2):
    return [{"id": f"q{i}", "source": "squad", "question": f"Question {i}?",
             "answers": ["doc u1"], "gold_passage_ids": ["u1"]} for i in range(1, n + 1)]


COSTS = {"search_calls": 1, "embedding_calls": 1, "reranker_calls": 1,
         "reranked_pairs": 5, "rewrite_calls": 0}


class Env:
    def __init__(self, monkeypatch, tmp_path, subset):
        self.log = {"once": [], "split": [], "answers": [], "sessions": 0, "planner_args": []}
        self.dataset = tmp_path / "test.parquet"
        self.dataset.write_text("x")
        self.results = tmp_path / "results"
        self.failing = set()
        monkeypatch.setattr(runner, "load_subset", lambda path: subset)

        @contextlib.contextmanager
        def session():
            self.log["sessions"] += 1
            yield "client"

        monkeypatch.setattr(runner, "weaviate_session", session)
        monkeypatch.setattr(runner, "endpoint_fingerprint", lambda: "test-endpoint")
        monkeypatch.setattr(runner, "get_embedding_model", lambda: "embedder")
        monkeypatch.setattr(runner, "get_reranker", lambda: "reranker")
        monkeypatch.setattr(runner, "get_slm", lambda name: (f"model:{name}", f"tok:{name}"))
        monkeypatch.setattr(runner, "load_adaptive_planner",
                            lambda adapter, base_model_name, use_4bit: ("adaptive-model", "adaptive-tok"))
        monkeypatch.setattr(runner, "retrieve_single",
                            lambda *a: self.log["once"].append(a[5]) or (docs(1, 2), dict(COSTS)))
        monkeypatch.setattr(runner, "retrieve_decomposed",
                            lambda *a: self.log["split"].append(a[5]) or (docs(1, 2), dict(COSTS)))
        monkeypatch.setattr(runner, "generate_answer",
                            lambda m, t, q, d: self.log["answers"].append((m, q)) or "doc u1")

        def planner(model, tokenizer, question):
            self.log["planner_args"].append((model, tokenizer))
            if question in self.failing:
                raise ValueError("bad json")
            return {"question": question, "strategy": "once", "queries": once_plan()["queries"],
                    "planner_attempts": 2}

        monkeypatch.setitem(runner.PLANNERS, "prompted", planner)

    def run(self, strategy="prompted", **kw):
        runner.run_evaluation("exp1", self.dataset, strategy, results_dir=self.results, **kw)

    def records(self):
        import pandas as pd
        (path,) = self.results.rglob("*.parquet")
        return pd.read_parquet(path)


@pytest.fixture
def env(monkeypatch, tmp_path):
    return Env(monkeypatch, tmp_path, rows())


class TestValidation:
    def test_unknown_strategy(self, env):
        with pytest.raises(ValueError, match="planner_strategy"):
            env.run("magic")

    def test_adaptive_requires_existing_adapter_dir(self, env, tmp_path):
        with pytest.raises(ValueError, match="adapter_dir"):
            env.run("adaptive")
        with pytest.raises(ValueError, match="adapter_dir"):
            env.run("adaptive", adapter_dir=tmp_path / "missing")

    def test_adaptive_requires_a_decision_threshold(self, env, tmp_path):
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        with pytest.raises(ValueError, match="decision_threshold"):
            env.run("adaptive", adapter_dir=adapter)
        with pytest.raises(ValueError, match="decision_threshold"):
            env.run("adaptive", adapter_dir=adapter, decision_threshold=1.5)

    def test_decision_threshold_rejected_for_other_strategies(self, env, tmp_path):
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        with pytest.raises(ValueError, match="only valid"):
            env.run("adaptive_split", adapter_dir=adapter, decision_threshold=0.3)
        with pytest.raises(ValueError, match="only valid"):
            env.run("prompted", decision_threshold=0.3)

    def test_adapter_dir_rejected_for_non_adaptive(self, env, tmp_path):
        with pytest.raises(ValueError, match="only valid"):
            env.run("prompted", adapter_dir=tmp_path)

    def test_empty_subset(self, monkeypatch, tmp_path):
        e = Env(monkeypatch, tmp_path, [])
        with pytest.raises(ValueError, match="empty"):
            e.run()
        assert e.log["sessions"] == 0

    def test_validation_happens_before_any_model_is_loaded(self, env):
        with pytest.raises(ValueError):
            env.run("magic")
        assert env.log["sessions"] == 0


class TestPlannerRegistry:
    def test_all_six_conditions_registered(self):
        assert set(runner.PLANNERS) == {"prompted", "always_once", "always_split",
                                        "adaptive", "adaptive_once", "adaptive_split"}

    def test_forced_variants_bind_the_strategy(self):
        assert isinstance(runner.PLANNERS["adaptive_once"], partial)
        assert runner.PLANNERS["adaptive_once"].keywords == {"forced_strategy": "once"}
        assert runner.PLANNERS["adaptive_split"].keywords == {"forced_strategy": "split"}


class TestRun:
    def test_resumes_after_interruption_and_recovers_partial_line(self, env, monkeypatch):
        generate = runner.generate_answer

        def interrupt(model, tokenizer, question, docs):
            if question == "Question 2?":
                raise RuntimeError("interrupted")
            return generate(model, tokenizer, question, docs)

        monkeypatch.setattr(runner, "generate_answer", interrupt)
        with pytest.raises(RuntimeError, match="interrupted"):
            env.run()
        checkpoint = env.results / "exp1" / "checkpoint.jsonl"
        with checkpoint.open("ab") as stream:
            stream.write(b'{"id": "q2"')
        assert not list(env.results.rglob("*.parquet"))

        monkeypatch.setattr(runner, "generate_answer", generate)
        env.log["once"].clear()
        env.run()
        assert [plan["question"] for plan in env.log["once"]] == ["Question 2?"]
        assert env.records()["id"].tolist() == ["q1", "q2"]

        sessions = env.log["sessions"]
        env.run()
        assert env.log["sessions"] == sessions

    def test_rejects_changed_dataset_before_loading_models(self, env, monkeypatch):
        env.run()
        changed = rows()
        changed[0]["answers"] = ["different reference"]
        monkeypatch.setattr(runner, "load_subset", lambda path: changed)
        with pytest.raises(ValueError, match="configuration mismatch"):
            env.run()
        assert env.log["sessions"] == 1

    def test_metadata_stores_fingerprint_and_rejects_changed_endpoint(self, env, monkeypatch):
        env.run()
        directory = env.results / "exp1"
        for name in ("config.json", "checkpoint_config.json"):
            retrieval = json.loads((directory / name).read_text())["retrieval"]
            assert retrieval["endpoint_sha256"] == "test-endpoint"
            assert not {"http_host", "grpc_host"} & retrieval.keys()
        monkeypatch.setattr(runner, "endpoint_fingerprint", lambda: "changed-endpoint")
        with pytest.raises(ValueError, match="configuration mismatch"):
            env.run()
        assert env.log["sessions"] == 1

    def test_routes_once_plans_to_single_retrieval_and_scores_answers(self, env):
        env.run()
        assert len(env.log["once"]) == 2 and env.log["split"] == []
        frame = env.records()
        assert list(frame["planner_status"]) == ["ok", "ok"]
        assert list(frame["planner_attempts"]) == [2, 2]
        assert frame[f"recall_at_{cfg.FINAL_CONTEXT_TOP_K}"].tolist() == [1.0, 1.0]
        assert frame["answer_em"].tolist() == [1.0, 1.0]

    def test_split_plans_use_decomposed_retrieval(self, env, monkeypatch):
        def planner(model, tok, question):
            return {"question": question, "strategy": "split", "queries": split_plan()["queries"]}
        monkeypatch.setitem(runner.PLANNERS, "prompted", planner)
        env.run()
        assert len(env.log["split"]) == 2 and env.log["once"] == []

    def test_prompted_baselines_share_the_query_model(self, env):
        env.run()
        assert set(env.log["planner_args"]) == {(f"model:{cfg.SLM_MODEL}", f"tok:{cfg.SLM_MODEL}")}

    def test_planner_failure_is_recorded_not_fatal(self, env):
        env.failing = {"Question 1?"}
        env.run()
        frame = env.records()
        assert list(frame["planner_status"]) == ["failed", "ok"]
        assert frame["planner_error"].iloc[0] == "bad json"
        assert len(env.log["once"]) == 1 and len(env.log["answers"]) == 1
        assert frame["answer_f1"].isna().iloc[0]

    def test_all_planners_failing_still_completes(self, env):
        env.failing = {"Question 1?", "Question 2?"}
        env.run()
        assert list(env.records()["planner_status"]) == ["failed", "failed"]
        assert env.log["answers"] == []

    def test_adaptive_loads_adapter_and_records_it(self, env, tmp_path, monkeypatch):
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        captured = {}

        def adaptive(model, tok, question, decision_threshold=None):
            captured["args"] = (model, tok)
            captured["threshold"] = decision_threshold
            return {"question": question, "strategy": "once", "queries": once_plan()["queries"]}

        monkeypatch.setitem(runner.PLANNERS, "adaptive", adaptive)
        env.run("adaptive", adapter_dir=adapter, adaptive_use_4bit=True, decision_threshold=0.3)
        assert captured["args"] == ("adaptive-model", "adaptive-tok")
        assert captured["threshold"] == 0.3
        config_file = next(env.results.rglob("*config*.json"))
        models = json.loads(config_file.read_text())["models"]
        assert models["planner_adapter"] == str(adapter.resolve())
        assert models["planner_use_4bit"] is True
        assert models["planner_decision_threshold"] == 0.3
