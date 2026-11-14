"""Unit tests for adaptive_rag.evaluation.retrieval (passage Recall@K / NDCG@K)."""
from math import log2

import pytest

from adaptive_rag.evaluation.retrieval import avg_ndcg_at_k, avg_recall_at_k, ndcg_at_k, recall_at_k


def docs(*ids):
    return [{"uuid": i} for i in ids]


class TestRecallAtK:
    def test_fraction_of_gold_found_in_top_k(self):
        retrieved = docs("a", "x", "b")
        assert recall_at_k(["a", "b"], retrieved, 1) == 0.5
        assert recall_at_k(["a", "b"], retrieved, 2) == 0.5
        assert recall_at_k(["a", "b"], retrieved, 3) == 1.0

    def test_k_larger_than_retrieved_is_fine(self):
        assert recall_at_k(["a"], docs("a"), 10) == 1.0

    def test_no_gold_returns_zero(self):
        assert recall_at_k([], docs("a"), 3) == 0.0

    def test_nothing_retrieved(self):
        assert recall_at_k(["a"], [], 3) == 0.0

    def test_duplicate_gold_ids_counted_once(self):
        assert recall_at_k(["a", "a"], docs("a"), 1) == 1.0

    @pytest.mark.parametrize("k", [0, -1])
    def test_rejects_non_positive_k(self, k):
        with pytest.raises(ValueError):
            recall_at_k(["a"], docs("a"), k)


class TestNdcgAtK:
    def test_perfect_ranking(self):
        assert ndcg_at_k(["a", "b"], docs("a", "b"), 2) == pytest.approx(1.0)

    def test_penalises_late_hits(self):
        dcg = 1 / log2(3) + 1 / log2(4)
        idcg = 1 / log2(2) + 1 / log2(3)
        assert ndcg_at_k(["a", "b"], docs("x", "a", "b"), 3) == pytest.approx(dcg / idcg)

    def test_miss_in_top_k_scores_zero(self):
        assert ndcg_at_k(["a"], docs("x", "a"), 1) == 0.0

    def test_ideal_dcg_is_capped_by_k(self):
        # Two gold passages but k=1: a single hit is already ideal.
        assert ndcg_at_k(["a", "b"], docs("a"), 1) == pytest.approx(1.0)

    def test_repeated_gold_hit_not_double_counted(self):
        assert ndcg_at_k(["a"], docs("a", "a"), 2) == pytest.approx(1.0)

    def test_no_gold_returns_zero(self):
        assert ndcg_at_k([], docs("a"), 3) == 0.0

    @pytest.mark.parametrize("k", [0, -2])
    def test_rejects_non_positive_k(self, k):
        with pytest.raises(ValueError):
            ndcg_at_k(["a"], docs("a"), k)

    def test_bounded_between_zero_and_one(self):
        value = ndcg_at_k(["a", "b", "c"], docs("c", "x", "a", "y"), 4)
        assert 0.0 <= value <= 1.0


class TestAverages:
    def test_mean(self):
        assert avg_recall_at_k([1.0, 0.5, 0.0]) == pytest.approx(0.5)
        assert avg_ndcg_at_k([0.2, 0.4]) == pytest.approx(0.3)

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            avg_recall_at_k([])
        with pytest.raises(ValueError):
            avg_ndcg_at_k([])
