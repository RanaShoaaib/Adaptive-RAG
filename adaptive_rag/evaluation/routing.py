"""Offline quality-cost routing curves built from per-question forced-strategy results.

With the strategy prefilled, ``adaptive`` at threshold tau generates exactly what
``adaptive_once`` generates for questions with P(split) < tau and what
``adaptive_split`` generates for the rest. A routing curve therefore needs no new
generation: take each question's row from the SPLIT run if it is routed, else from
the ONCE run, and average.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import pandas as pd

DEFAULT_BUDGETS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0)
SERIES = ("learned", "random", "oracle")


def _aligned(once: pd.DataFrame, split: pd.DataFrame, p_split: Mapping[str, float],
             recall_col: str, cost_col: str, search_col: str) -> pd.DataFrame:
    frames = {}
    for name, df in (("once", once), ("split", split)):
        for col in ("id", recall_col, cost_col, search_col):
            if col not in df.columns:
                raise ValueError(f"{name} results are missing column {col!r}")
        if df["id"].astype(str).duplicated().any():
            raise ValueError(f"Duplicate ids in {name} results")
        frames[name] = df.assign(id=df["id"].astype(str)).set_index("id")[[recall_col, cost_col, search_col]]
    ids = set(frames["once"].index)
    if ids != set(frames["split"].index) or ids != {str(k) for k in p_split}:
        raise ValueError("ONCE results, SPLIT results and P(split) scores must cover the same question ids")
    p = pd.Series({str(k): float(v) for k, v in p_split.items()}, name="p")
    merged = frames["once"].add_suffix("_once").join(frames["split"].add_suffix("_split")).join(p)
    # Highest P(split) first; ties broken by id so the ranking is deterministic.
    merged = merged.reset_index().sort_values(["p", "id"], ascending=[False, True], kind="stable")
    return merged.reset_index(drop=True)


def routing_curve(once: pd.DataFrame, split: pd.DataFrame, p_split: Mapping[str, float],
                  budgets: Sequence[float] = DEFAULT_BUDGETS, *, recall_col: str = "recall_at_7",
                  cost_col: str = "reranked_pairs", search_col: str = "search_calls") -> pd.DataFrame:
    """Mean recall and cost when the top `budget` share (by P(split)) goes to SPLIT.

    Returns one row per budget with the learned router, the random-router expectation
    (a straight line between always-ONCE and always-SPLIT) and an oracle router that
    sends the questions with the largest true recall gain first (never a non-gainer).
    Means skip failed (NaN) planner rows.
    """
    if any(not 0 <= b <= 1 for b in budgets):
        raise ValueError("budgets must lie in [0, 1]")
    df = _aligned(once, split, p_split, recall_col, cost_col, search_col)
    n = len(df)
    r1, r2 = df[f"{recall_col}_once"], df[f"{recall_col}_split"]
    c1, c2 = df[f"{cost_col}_once"], df[f"{cost_col}_split"]
    s1, s2 = df[f"{search_col}_once"], df[f"{search_col}_split"]
    gain = (r2 - r1).fillna(0.0)
    oracle_order = gain.sort_values(ascending=False, kind="stable").index
    n_gainers = int((gain > 0).sum())
    ends = {"recall": (r1.mean(), r2.mean()), "pairs": (c1.mean(), c2.mean())}

    rows = []
    for budget in sorted(set(budgets)):
        k = int(round(budget * n))
        routed = pd.Series(False, index=df.index)
        routed.iloc[:k] = True
        pick = lambda a, b: a.where(~routed, b)  # noqa: E731 - SPLIT value if routed else ONCE value
        share = k / n
        oracle_routed = pd.Series(False, index=df.index)
        oracle_routed.loc[oracle_order[:min(k, n_gainers)]] = True
        row = {
            "budget": budget, "n_split": k,
            "tau": float(df["p"].iloc[k - 1]) if k else None,
            "learned_recall": pick(r1, r2).mean(), "learned_pairs": pick(c1, c2).mean(),
            "learned_searches": pick(s1, s2).mean(),
            "random_recall": ends["recall"][0] + share * (ends["recall"][1] - ends["recall"][0]),
            "random_pairs": ends["pairs"][0] + share * (ends["pairs"][1] - ends["pairs"][0]),
            "oracle_recall": r1.where(~oracle_routed, r2).mean(),
            "oracle_pairs": c1.where(~oracle_routed, c2).mean(),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def choose_operating_point(curve: pd.DataFrame, gain_fraction: float = 0.8) -> dict:
    """Smallest budget capturing `gain_fraction` of the recall gained by always-SPLIT.

    Fix the fraction before looking at the curve. Raises when always-SPLIT does not
    beat always-ONCE on this data, because no cost-aware threshold is then justified.
    """
    if not 0 < gain_fraction <= 1:
        raise ValueError("gain_fraction must be in (0, 1]")
    if curve["budget"].iloc[0] != 0 or curve["budget"].iloc[-1] != 1:
        raise ValueError("The curve must include budgets 0 and 1")
    base = curve["learned_recall"].iloc[0]
    total = curve["learned_recall"].iloc[-1] - base
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Always-SPLIT does not improve recall over always-ONCE here; no threshold is justified")
    reached = curve[(curve["learned_recall"] - base) >= gain_fraction * total - 1e-12]
    row = reached.iloc[0]
    return {
        "budget": float(row["budget"]), "n_split": int(row["n_split"]),
        "tau": None if row["tau"] is None or pd.isna(row["tau"]) else float(row["tau"]),
        "recall": float(row["learned_recall"]), "pairs": float(row["learned_pairs"]),
        "gain_fraction": gain_fraction, "total_gain": float(total),
    }
