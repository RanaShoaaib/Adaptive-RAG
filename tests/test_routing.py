"""Tests for adaptive_rag.evaluation.routing (offline quality-cost curves)."""
import pandas as pd
import pytest

from adaptive_rag.evaluation.routing import choose_operating_point, routing_curve


def frames():
    ids = list("abcd")
    once = pd.DataFrame({"id": ids, "recall_at_7": [1.0, 0.0, 1.0, 0.5],
                         "reranked_pairs": [20] * 4, "search_calls": [1] * 4})
    split = pd.DataFrame({"id": ids, "recall_at_7": [1.0, 1.0, 0.5, 0.5],
                          "reranked_pairs": [80] * 4, "search_calls": [2] * 4})
    p = {"a": 0.1, "b": 0.9, "c": 0.2, "d": 0.8}
    return once, split, p


def test_endpoints_are_always_once_and_always_split():
    curve = routing_curve(*frames(), budgets=[0, 1])
    first, last = curve.iloc[0], curve.iloc[-1]
    assert first["learned_recall"] == pytest.approx(0.625) and first["learned_pairs"] == 20
    assert last["learned_recall"] == pytest.approx(0.75) and last["learned_pairs"] == 80
    assert first["n_split"] == 0 and last["n_split"] == 4 and pd.isna(first["tau"])


def test_routes_the_highest_scores_first_and_reports_tau():
    curve = routing_curve(*frames(), budgets=[0, 0.25, 0.5, 1]).set_index("budget")
    assert curve.loc[0.25, "n_split"] == 1 and curve.loc[0.25, "tau"] == 0.9
    assert curve.loc[0.25, "learned_recall"] == pytest.approx((1 + 1 + 1 + 0.5) / 4)  # b routed: 0 -> 1
    assert curve.loc[0.5, "tau"] == 0.8  # b and d routed
    assert curve.loc[0.5, "learned_pairs"] == pytest.approx((20 + 80 + 20 + 80) / 4)


def test_random_is_a_straight_line_and_oracle_dominates():
    curve = routing_curve(*frames(), budgets=[0, 0.25, 0.5, 1]).set_index("budget")
    assert curve.loc[0.5, "random_recall"] == pytest.approx((0.625 + 0.75) / 2)
    assert curve.loc[0.5, "random_pairs"] == pytest.approx(50)
    assert (curve["oracle_recall"] >= curve["learned_recall"] - 1e-12).all()
    # Only question b truly benefits (+1.0); the oracle never routes a question with no gain.
    assert curve.loc[0.5, "oracle_pairs"] == pytest.approx((20 * 3 + 80) / 4)
    assert curve.loc[0.5, "oracle_recall"] == pytest.approx(0.875)


def test_ties_are_broken_deterministically_by_id():
    once, split, _ = frames()
    flat = {k: 0.5 for k in "abcd"}
    first = routing_curve(once, split, flat, budgets=[0.5])
    second = routing_curve(once.iloc[::-1], split.iloc[::-1], flat, budgets=[0.5])
    assert first.equals(second)


def test_failed_rows_are_skipped_in_means():
    once, split, p = frames()
    once.loc[once["id"] == "a", "recall_at_7"] = float("nan")
    curve = routing_curve(once, split, p, budgets=[0])
    assert curve["learned_recall"].iloc[0] == pytest.approx((0 + 1 + 0.5) / 3)


@pytest.mark.parametrize("break_it", ["ids", "dupes", "column"])
def test_rejects_inconsistent_inputs(break_it):
    once, split, p = frames()
    if break_it == "ids":
        p = {**p, "z": 0.3}
    elif break_it == "dupes":
        once = pd.concat([once, once.iloc[:1]])
    else:
        split = split.drop(columns=["reranked_pairs"])
    with pytest.raises(ValueError):
        routing_curve(once, split, p)


def test_rejects_out_of_range_budget():
    with pytest.raises(ValueError):
        routing_curve(*frames(), budgets=[0, 1.5])


class TestChooseOperatingPoint:
    def curve(self):
        return pd.DataFrame({
            "budget": [0, 0.1, 0.5, 1.0], "n_split": [0, 1, 5, 10],
            "tau": [None, 0.9, 0.6, 0.1],
            "learned_recall": [0.70, 0.76, 0.79, 0.80], "learned_pairs": [20, 26, 50, 80],
        })

    def test_picks_the_smallest_budget_reaching_the_gain_fraction(self):
        point = choose_operating_point(self.curve(), 0.8)  # needs 0.08 of the 0.10 gain
        assert point["budget"] == 0.5 and point["tau"] == 0.6 and point["n_split"] == 5
        assert choose_operating_point(self.curve(), 0.5)["budget"] == 0.1

    def test_full_fraction_selects_a_budget_that_reaches_the_always_split_recall(self):
        assert choose_operating_point(self.curve(), 1.0)["budget"] == 1.0

    def test_raises_when_splitting_does_not_help(self):
        flat = self.curve().assign(learned_recall=[0.7, 0.69, 0.68, 0.7])
        with pytest.raises(ValueError, match="no threshold"):
            choose_operating_point(flat)

    def test_requires_endpoints_and_valid_fraction(self):
        with pytest.raises(ValueError, match="include budgets 0 and 1"):
            choose_operating_point(self.curve().iloc[1:])
        with pytest.raises(ValueError):
            choose_operating_point(self.curve(), 0)
