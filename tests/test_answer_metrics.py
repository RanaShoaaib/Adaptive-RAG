"""Unit tests for adaptive_rag.evaluation.answer (SQuAD-style EM / token F1)."""
import pytest

from adaptive_rag.evaluation.answer import evaluate_answer, exact_match, normalize_answer, token_f1


class TestNormalizeAnswer:
    def test_lowercases_strips_punctuation_and_articles(self):
        assert normalize_answer("The  Quick, Brown-Fox!") == "quick brownfox"

    def test_articles_only_removed_as_whole_words(self):
        assert normalize_answer("an apple a day") == "apple day"
        assert normalize_answer("Theodore") == "theodore"

    def test_collapses_whitespace(self):
        assert normalize_answer("  a   b\tc\n") == "b c"

    def test_empty_string(self):
        assert normalize_answer("") == ""


class TestExactMatch:
    def test_ignores_case_punctuation_and_articles(self):
        assert exact_match("The Eiffel Tower.", "eiffel tower") == 1.0

    def test_different_answers(self):
        assert exact_match("Paris", "London") == 0.0

    def test_returns_float(self):
        assert isinstance(exact_match("a", "b"), float)


class TestTokenF1:
    def test_identical(self):
        assert token_f1("red apple", "red apple") == 1.0

    def test_partial_overlap_precision_recall(self):
        # pred has 3 tokens, ref 2, common 2 -> P=2/3, R=1 -> F1=0.8
        assert token_f1("red apple pie", "apple pie") == pytest.approx(0.8)

    def test_disjoint(self):
        assert token_f1("cat", "dog") == 0.0

    def test_duplicate_tokens_use_multiset_overlap(self):
        # common = min counts: x->1, y->1 => 2; P=R=2/3
        assert token_f1("x x y", "x y y") == pytest.approx(2 / 3)

    def test_both_empty_after_normalisation_is_perfect(self):
        assert token_f1("the", "a") == 1.0

    def test_one_empty_is_zero(self):
        assert token_f1("", "answer") == 0.0
        assert token_f1("answer", "") == 0.0

    def test_symmetric(self):
        assert token_f1("a b c", "b c d") == pytest.approx(token_f1("b c d", "a b c"))


class TestEvaluateAnswer:
    def test_takes_best_over_references(self):
        scores = evaluate_answer("Paris", ["London", "paris", "Berlin"])
        assert scores == {"exact_match": 1.0, "f1": 1.0}

    def test_em_and_f1_are_maximised_independently(self):
        scores = evaluate_answer("big red apple", ["red apple", "green"])
        assert scores["exact_match"] == 0.0
        assert scores["f1"] == pytest.approx(0.8)

    def test_requires_references(self):
        with pytest.raises(ValueError, match="references"):
            evaluate_answer("x", [])
