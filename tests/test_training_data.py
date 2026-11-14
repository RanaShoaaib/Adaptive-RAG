"""Unit tests for adaptive_rag.data.training_data (strategy labels and SFT export)."""
import json
import math

import pytest

from adaptive_rag.data.training_data import (
    export_label_summary_parquet,
    export_labeled_data,
    export_sft_examples,
    label_question,
    label_retrieval_results,
    strategy_score,
    summarize_labels,
)
from adaptive_rag.evaluation.teacher_retrieval import save_teacher_retrieval
from adaptive_rag.planner.adaptive import load_sft_examples
from rag_fakes import once_plan, split_plan

K = 7


def make_record(once_recall=(0.5,) * K, split_recall=(0.5,) * K,
                once_ndcg=(0.5,) * K, split_ndcg=(0.5,) * K, qid="q1", source="hotpotqa"):
    record = {"question_id": qid, "question": f"Question {qid}?", "source": source,
              "once_plan": once_plan(), "split_plan": split_plan()}
    for k in range(1, K + 1):
        record[f"once_recall_at_{k}"] = once_recall[k - 1]
        record[f"split_recall_at_{k}"] = split_recall[k - 1]
        record[f"once_ndcg_at_{k}"] = once_ndcg[k - 1]
        record[f"split_ndcg_at_{k}"] = split_ndcg[k - 1]
    return record


def flat(value):
    return (value,) * K


class TestStrategyScore:
    record = make_record(
        once_recall=(0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0),
        once_ndcg=(0.0, 0.3, 0.3, 0.3, 0.6, 0.6, 0.8),
    )

    def test_recall_at_k_reads_the_requested_cutoff(self):
        assert strategy_score(self.record, "once", metric="recall_at_k", k=1) == 0.0
        assert strategy_score(self.record, "once", metric="recall_at_k", k=5) == 1.0

    def test_ndcg_at_k(self):
        assert strategy_score(self.record, "once", metric="ndcg_at_k", k=7) == pytest.approx(0.8)

    def test_mean_recall_averages_cutoffs_one_to_k(self):
        expected = (0.0 + 0.5 + 0.5 + 0.5 + 1.0 + 1.0 + 1.0) / 7
        assert strategy_score(self.record, "once", metric="mean_recall", k=7) == pytest.approx(expected)

    def test_mean_ndcg(self):
        expected = (0.0 + 0.3 + 0.3 + 0.3 + 0.6 + 0.6 + 0.8) / 7
        assert strategy_score(self.record, "once", metric="mean_ndcg", k=7) == pytest.approx(expected)

    def test_combined_is_a_weighted_blend(self):
        value = strategy_score(self.record, "once", metric="combined", k=7, alpha=0.25)
        assert value == pytest.approx(0.25 * 1.0 + 0.75 * 0.8)

    @pytest.mark.parametrize("kwargs", [
        {"strategy": "both"}, {"metric": "f1"}, {"k": 0}, {"k": 1.5}, {"alpha": -0.1}, {"alpha": 1.1},
        {"alpha": math.nan},
    ])
    def test_rejects_invalid_arguments(self, kwargs):
        args = {"strategy": "once", "metric": "recall_at_k", "k": 7, "alpha": 0.6}
        args.update(kwargs)
        strategy = args.pop("strategy")
        with pytest.raises(ValueError):
            strategy_score(self.record, strategy, **args)

    def test_missing_metric_is_reported(self):
        incomplete = dict(self.record)
        del incomplete["once_recall_at_3"]
        with pytest.raises(ValueError, match="once_recall_at_3"):
            strategy_score(incomplete, "once", metric="recall_at_k", k=3)

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -0.1, 1.2])
    def test_rejects_out_of_range_or_non_finite_values(self, bad):
        broken = dict(self.record, once_recall_at_7=bad)
        with pytest.raises(ValueError, match="Invalid"):
            strategy_score(broken, "once", metric="recall_at_k", k=7)


class TestLabelQuestion:
    def test_split_only_when_improvement_exceeds_threshold(self):
        record = make_record(once_recall=flat(0.5), split_recall=flat(1.0))
        label = label_question(record, metric="recall_at_k", threshold=0.1)
        assert label["strategy"] == "split"
        assert label["improvement"] == pytest.approx(0.5)
        assert label["selected_plan"]["strategy"] == "split"

    def test_ties_prefer_once(self):
        label = label_question(make_record(), threshold=0.0)
        assert label["strategy"] == "once" and label["improvement"] == 0.0

    def test_improvement_equal_to_threshold_stays_once(self):
        record = make_record(once_recall=flat(0.5), split_recall=flat(0.6))
        assert label_question(record, metric="recall_at_k", threshold=0.1)["strategy"] == "once"

    def test_split_worse_than_once_is_once(self):
        record = make_record(once_recall=flat(1.0), split_recall=flat(0.0))
        label = label_question(record, metric="recall_at_k")
        assert label["strategy"] == "once" and label["improvement"] == -1.0

    def test_selected_plan_carries_the_original_question(self):
        label = label_question(make_record(qid="q9"))
        assert label["selected_plan"]["question"] == "Question q9?"
        assert label["question_id"] == "q9"

    def test_alpha_is_recorded_only_for_the_combined_metric(self):
        assert label_question(make_record(), metric="recall_at_k")["alpha"] is None
        assert label_question(make_record(), metric="combined", alpha=0.3)["alpha"] == 0.3

    @pytest.mark.parametrize("field", ["question_id", "question", "once_plan", "split_plan"])
    def test_requires_core_fields(self, field):
        record = make_record()
        del record[field]
        with pytest.raises(ValueError, match=field):
            label_question(record)

    def test_rejects_blank_identifiers_and_bad_threshold(self):
        with pytest.raises(ValueError):
            label_question(make_record(qid="  "))
        for threshold in (-0.1, 1.5, math.nan):
            with pytest.raises(ValueError, match="threshold"):
                label_question(make_record(), threshold=threshold)

    def test_rejects_invalid_teacher_plans(self):
        record = make_record()
        record["split_plan"] = once_plan()
        with pytest.raises(ValueError):
            label_question(record)


class TestSummaries:
    def labels(self):
        return [
            label_question(make_record(qid="a", source="hotpotqa", once_recall=flat(0.0), split_recall=flat(1.0)),
                           metric="recall_at_k"),
            label_question(make_record(qid="b", source="hotpotqa"), metric="recall_at_k"),
            label_question(make_record(qid="c", source="squad"), metric="recall_at_k"),
            label_question({**make_record(qid="d"), "source": None}, metric="recall_at_k"),
        ]

    def test_counts_by_strategy_and_source(self):
        summary = summarize_labels(self.labels())
        assert summary["total"] == 4 and summary["split"] == 1 and summary["once"] == 3
        assert summary["split_fraction"] == pytest.approx(0.25)
        assert summary["by_source"] == {"hotpotqa": {"once": 1, "split": 1},
                                        "squad": {"once": 1, "split": 0},
                                        "unknown": {"once": 1, "split": 0}}

    def test_empty_summary(self):
        assert summarize_labels([]) == {"total": 0, "once": 0, "split": 0,
                                        "split_fraction": 0.0, "by_source": {}}


class TestLabelRetrievalResults:
    def test_labels_every_saved_record(self, tmp_path):
        path = tmp_path / "retrieval.jsonl"
        save_teacher_retrieval(make_record(qid="a", once_recall=flat(0.0), split_recall=flat(1.0)), path)
        save_teacher_retrieval(make_record(qid="b"), path)
        labels = label_retrieval_results(path, metric="recall_at_k")
        assert [(l["question_id"], l["strategy"]) for l in labels] == [("a", "split"), ("b", "once")]

    def test_requires_saved_results(self, tmp_path):
        with pytest.raises(ValueError, match="No saved teacher retrieval records"):
            label_retrieval_results(tmp_path / "missing.jsonl")


class TestExports:
    def labels(self):
        return [
            label_question(make_record(qid="a", once_recall=flat(0.0), split_recall=flat(1.0)), metric="recall_at_k"),
            label_question(make_record(qid="b"), metric="recall_at_k"),
        ]

    def test_export_labeled_data_roundtrip_leaves_no_temp_files(self, tmp_path):
        path = export_labeled_data(self.labels(), tmp_path / "out" / "labels.jsonl")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert [r["question_id"] for r in rows] == ["a", "b"]
        assert rows[0]["selected_plan"]["strategy"] == "split"
        assert [p.name for p in path.parent.iterdir()] == ["labels.jsonl"]

    def test_export_rejects_non_finite_numbers(self, tmp_path):
        labels = self.labels()
        labels[0]["improvement"] = math.nan
        with pytest.raises(ValueError):
            export_labeled_data(labels, tmp_path / "labels.jsonl")
        assert not (tmp_path / "labels.jsonl").exists()

    def test_sft_export_loads_in_the_training_module(self, tmp_path):
        """Contract between data.training_data and planner.adaptive."""
        path = export_sft_examples(self.labels(), tmp_path / "sft.jsonl")
        examples = load_sft_examples(path)
        assert [e["question_id"] for e in examples] == ["a", "b"]
        assert [e["target"]["strategy"] for e in examples] == ["split", "once"]
        assert all(set(e["target"]) == {"strategy", "queries"} for e in examples)
        assert examples[0]["question"] == "Question a?"

    def test_sft_export_rejects_plan_label_mismatch(self, tmp_path):
        labels = self.labels()
        labels[0]["strategy"] = "once"  # selected_plan still says split
        with pytest.raises(ValueError, match="mismatch"):
            export_sft_examples(labels, tmp_path / "sft.jsonl")

    def test_summary_parquet_requires_labels(self, tmp_path):
        with pytest.raises(ValueError, match="No labels"):
            export_label_summary_parquet([], tmp_path / "s.parquet")

    def test_summary_parquet_drops_nested_plan(self, tmp_path):
        pd = pytest.importorskip("pandas")
        pytest.importorskip("pyarrow")
        path = export_label_summary_parquet(self.labels(), tmp_path / "s.parquet")
        frame = pd.read_parquet(path)
        assert "selected_plan" not in frame.columns
        assert list(frame["question_id"]) == ["a", "b"]
