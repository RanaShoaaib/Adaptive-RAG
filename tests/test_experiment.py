"""Unit tests for adaptive_rag.evaluation.experiment (records, summaries, persistence)."""
import json

import pandas as pd
import pytest

from adaptive_rag import config as cfg
from adaptive_rag.evaluation.experiment import (
    COST_KEYS,
    create_experiment_config,
    create_result_record,
    save_experiment_results,
)

pytest.importorskip("pyarrow", reason="results.parquet needs pyarrow")

COSTS = {key: 1 for key in COST_KEYS}


def question_row(qid="q1", source="squad", answers=("Paris",), gold=("u1",)):
    return {"id": qid, "source": source, "question": "Capital of France?",
            "answers": list(answers), "gold_passage_ids": list(gold)}


def plan(strategy="once"):
    return {"strategy": strategy, "queries": [{"lexical_query": "l", "semantic_query": "s"}]}


def record(qid="q1", source="squad", strategy="once", answer="Paris", retrieved=("u1", "u2"), max_k=3):
    docs = [{"uuid": u} for u in retrieved]
    return create_result_record(question_row(qid, source), plan(strategy), docs, COSTS, answer, max_k=max_k)


class TestCreateExperimentConfig:
    @pytest.mark.parametrize("bad_id", ["", ".", "..", "a/b", "../x"])
    def test_rejects_unsafe_experiment_ids(self, bad_id, tmp_path):
        with pytest.raises(ValueError):
            create_experiment_config(bad_id, tmp_path / "d.parquet", 5)

    def test_rejects_non_positive_sample_count(self, tmp_path):
        with pytest.raises(ValueError):
            create_experiment_config("exp", tmp_path / "d.parquet", 0)

    def test_snapshots_settings_from_config(self, tmp_path):
        config = create_experiment_config("exp", tmp_path / "d.parquet", 7, "prompted",
                                          sampling_strategy="balanced", sampling_seed=42)
        assert config["experiment_id"] == "exp"
        assert config["strategy"] == "prompted"
        assert config["dataset"]["num_samples"] == 7
        assert config["dataset"]["sampling_seed"] == 42
        assert config["models"]["planner"] == cfg.SLM_MODEL
        assert config["retrieval"]["final_context_top_k"] == cfg.FINAL_CONTEXT_TOP_K
        assert config["generation"]["do_sample"] is False
        json.dumps(config)  # must be JSON-serialisable

    def test_dataset_path_is_relative_inside_repo_and_kept_outside(self, tmp_path):
        inside = create_experiment_config("e", cfg.VAL_SUBSET_PATH, 1)["dataset"]["path"]
        assert inside.replace("\\", "/") == f"{cfg.DATASETS_DIR.name}/val_subset.parquet"
        outside = create_experiment_config("e", tmp_path / "x.parquet", 1)["dataset"]["path"]
        assert outside == str(tmp_path / "x.parquet")


class TestCreateResultRecord:
    def test_scores_retrieval_and_answer(self):
        rec = record(answer="paris", retrieved=("u2", "u1"))
        assert rec["strategy"] == "once"
        assert rec["retrieved_passage_ids"] == ["u2", "u1"]
        assert rec["recall_at_1"] == 0.0 and rec["recall_at_2"] == 1.0
        assert rec["answer_em"] == 1.0 and rec["answer_f1"] == 1.0
        assert rec["generated_answer"] == "paris"
        for key in COST_KEYS:
            assert rec[key] == 1

    def test_creates_metrics_for_every_cutoff(self):
        rec = record(max_k=4)
        assert {f"recall_at_{k}" for k in range(1, 5)} <= set(rec)
        assert {f"ndcg_at_{k}" for k in range(1, 5)} <= set(rec)
        assert "recall_at_5" not in rec

    def test_default_cutoff_is_final_context_budget(self):
        rec = create_result_record(question_row(), plan(), [{"uuid": "u1"}], COSTS, "Paris")
        assert f"recall_at_{cfg.FINAL_CONTEXT_TOP_K}" in rec
        assert f"recall_at_{cfg.FINAL_CONTEXT_TOP_K + 1}" not in rec

    def test_preserves_original_question_fields(self):
        rec = record()
        assert rec["id"] == "q1" and rec["source"] == "squad" and rec["answers"] == ["Paris"]

    def test_requires_gold_and_answers(self):
        row = question_row()
        del row["gold_passage_ids"]
        with pytest.raises(ValueError, match="gold_passage_ids"):
            create_result_record(row, plan(), [], COSTS, "x")

    def test_rejects_unknown_strategy(self):
        with pytest.raises(ValueError, match="strategy"):
            create_result_record(question_row(), plan("weird"), [], COSTS, "x")

    def test_requires_all_cost_counters(self):
        partial = dict(COSTS)
        partial.pop("rewrite_calls")
        with pytest.raises(ValueError, match="rewrite_calls"):
            create_result_record(question_row(), plan(), [], partial, "x")

    def test_rejects_non_positive_max_k(self):
        with pytest.raises(ValueError):
            create_result_record(question_row(), plan(), [], COSTS, "x", max_k=0)


class TestSaveExperimentResults:
    def config(self, name="exp", n=2):
        return {"experiment_id": name, "dataset": {"num_samples": n}}

    def test_writes_all_artifacts_atomically(self, tmp_path):
        save_experiment_results([record()], self.config(), results_dir=tmp_path)
        folder = tmp_path / "exp"
        assert {p.name for p in folder.iterdir()} == {
            "config.json", "results.parquet", "results.csv", "summary.json"}
        assert json.loads((folder / "config.json").read_text())["experiment_id"] == "exp"
        assert len(pd.read_parquet(folder / "results.parquet")) == 1

    def test_summary_means_and_counts(self, tmp_path):
        records = [
            record("q1", "squad", "once", "Paris", ("u1",)),
            record("q2", "hotpotqa", "split", "London", ("x",)),
        ]
        summary = save_experiment_results(records, self.config(n=2), results_dir=tmp_path)
        assert summary["num_examples"] == 2 and summary["planned_num_examples"] == 2
        assert summary["metrics_mean"]["recall_at_1"] == pytest.approx(0.5)
        assert summary["metrics_mean"]["answer_em"] == pytest.approx(0.5)
        assert summary["metrics_mean"]["search_calls"] == pytest.approx(1.0)
        assert summary["strategy_counts"] == {"once": 1, "split": 1}
        assert summary["source_counts"] == {"squad": 1, "hotpotqa": 1}
        assert summary["planner_failures"] == 0 and summary["planner_success_rate"] == 1.0

    def test_resaving_larger_snapshot_overwrites(self, tmp_path):
        save_experiment_results([record("q1")], self.config(), results_dir=tmp_path)
        summary = save_experiment_results([record("q1"), record("q2")], self.config(), results_dir=tmp_path)
        assert summary["num_examples"] == 2
        assert len(pd.read_csv(tmp_path / "exp" / "results.csv")) == 2

    def test_failed_planner_rows_are_counted_and_excluded_from_means(self, tmp_path):
        ok = record("q1", answer="Paris")
        ok.update(planner_status="ok")
        failed = {**question_row("q2"), "strategy": "failed", "planner_status": "failed",
                  "answer_f1": None, "answer_em": None, "recall_at_1": None}
        summary = save_experiment_results([ok, failed], self.config(), results_dir=tmp_path)
        assert summary["planner_failures"] == 1
        assert summary["planner_success_rate"] == pytest.approx(0.5)
        # NaN rows are skipped by the mean: only the successful row contributes.
        assert summary["metrics_mean"]["answer_f1"] == pytest.approx(1.0)
        assert summary["strategy_counts"]["failed"] == 1

    def test_rejects_empty_and_duplicate_records(self, tmp_path):
        with pytest.raises(ValueError, match="At least one"):
            save_experiment_results([], self.config(), results_dir=tmp_path)
        with pytest.raises(ValueError, match="Duplicate"):
            save_experiment_results([record("q1"), record("q1")], self.config(), results_dir=tmp_path)

    @pytest.mark.parametrize("bad_id", ["", "..", "a/b"])
    def test_rejects_unsafe_experiment_id(self, bad_id, tmp_path):
        with pytest.raises(ValueError):
            save_experiment_results([record()], self.config(bad_id), results_dir=tmp_path)

    def test_defaults_to_repo_results_directory(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "ROOT_DIR", tmp_path)
        save_experiment_results([record()], self.config("default_dir"))
        assert (tmp_path / "results" / "default_dir" / "summary.json").is_file()
