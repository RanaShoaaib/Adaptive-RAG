"""Unit tests for adaptive_rag.retrieval (reranker + retriever orchestration)."""
from types import SimpleNamespace

import pytest

from adaptive_rag.retrieval import reranker as rr
from adaptive_rag.retrieval import retriever as rt
from rag_fakes import FakeReranker, FakeWeaviateClient, doc, docs, once_plan, query, split_plan

ZERO_COSTS = {"search_calls": 0, "embedding_calls": 0, "reranker_calls": 0,
              "reranked_pairs": 0, "rewrite_calls": 0}


class TestRerank:
    def test_sorts_by_score_and_truncates(self):
        reranker = FakeReranker({"u1": 0.2, "u2": 0.9, "u3": 0.5})
        result = rr.rerank(reranker, "q", docs(1, 2, 3), 2)
        assert [(d["uuid"], s) for d, s in result] == [("u2", 0.9), ("u3", 0.5)]

    def test_pairs_use_title_newline_text(self):
        reranker = FakeReranker({"u1": 0.1})
        rr.rerank(reranker, "my query", docs(1), 1)
        assert reranker.calls == [[["my query", "T1\ndoc u1"]]]

    def test_empty_candidates_skip_the_model(self):
        reranker = FakeReranker({})
        assert rr.rerank(reranker, "q", [], 5) == []
        assert reranker.calls == []

    def test_invalid_top_k_only_raises_when_there_are_candidates(self):
        assert rr.rerank(FakeReranker({}), "q", [], 0) == []
        with pytest.raises(ValueError, match="positive"):
            rr.rerank(FakeReranker({"u1": 1.0}), "q", docs(1), 0)

    def test_scores_are_python_floats(self):
        import numpy as np
        class NpReranker:
            def predict(self, pairs):
                return np.array([0.25] * len(pairs), dtype="float32")
        (_, score), = rr.rerank(NpReranker(), "q", docs(1), 1)
        assert type(score) is float

    def test_ties_keep_original_order(self):
        reranker = FakeReranker(lambda q, d: 0.5)
        assert [d["uuid"] for d, _ in rr.rerank(reranker, "q", docs(3, 1, 2), 3)] == ["u3", "u1", "u2"]


class TestHybridSearch:
    def obj(self, i):
        return SimpleNamespace(uuid=f"uuid-{i}", properties={"title": f"T{i}", "text": f"x{i}"})

    @pytest.fixture
    def embed_spy(self, monkeypatch):
        seen = []
        monkeypatch.setattr(rt, "embed_queries", lambda model, qs: seen.append((model, qs)) or [[0.1, 0.2]])
        return seen

    def test_sends_lexical_and_semantic_parts_separately(self, embed_spy):
        client = FakeWeaviateClient(hybrid_objects=[self.obj(1), self.obj(2)])
        out = rt.hybrid_search(client, "M", query("lex words", "semantic text"), "COL", 0.4, 9)
        assert embed_spy == [("M", ["semantic text"])]
        (call,) = client.collection.hybrid_calls
        assert call == {"query": "lex words", "vector": [0.1, 0.2], "target_vector": "vector",
                        "alpha": 0.4, "limit": 9}
        assert out == [{"uuid": "uuid-1", "title": "T1", "text": "x1"},
                       {"uuid": "uuid-2", "title": "T2", "text": "x2"}]

    def test_uuid_is_stringified(self, embed_spy):
        class U:
            def __str__(self): return "abc"
        client = FakeWeaviateClient(hybrid_objects=[SimpleNamespace(uuid=U(), properties={"title": "t", "text": "x"})])
        assert rt.hybrid_search(client, "M", query("a", "b"))[0]["uuid"] == "abc"

    @pytest.mark.parametrize("alpha", [-0.01, 1.01])
    def test_validates_alpha_before_touching_the_client(self, alpha):
        with pytest.raises(ValueError, match="alpha"):
            rt.hybrid_search(None, None, query("a", "b"), alpha=alpha)

    def test_validates_top_k_before_touching_the_client(self):
        with pytest.raises(ValueError, match="hybrid_top_k"):
            rt.hybrid_search(None, None, query("a", "b"), hybrid_top_k=0)


class TestMergeAndRerank:
    def test_dedupes_by_uuid_and_reranks_against_original_query(self):
        reranker = FakeReranker({"u1": 0.1, "u2": 0.8, "u3": 0.6})
        first = [(doc(1), 0.9), (doc(2), 0.5)]
        second = [(doc(2), 0.7), (doc(3), 0.4)]
        out = rt.merge_and_rerank_candidates(first, second, reranker, query("l", "orig"))
        assert [d["uuid"] for d, _ in out] == ["u2", "u3", "u1"]
        assert all(call[0] == "orig" for call in reranker.calls[0])
        assert len(reranker.calls[0]) == 3

    def test_empty_inputs(self):
        assert rt.merge_and_rerank_candidates([], [], FakeReranker({}), query("l", "s")) == []


class Harness:
    """Patches hybrid_search/rewrite_query and records calls."""

    def __init__(self, monkeypatch, results, rewritten=None):
        self.results = results  # {lexical_query: [docs]}
        self.rewritten = rewritten or query("rewritten lex", "rewritten sem")
        self.searched, self.rewrites = [], []
        monkeypatch.setattr(rt, "hybrid_search", self.search)
        monkeypatch.setattr(rt, "rewrite_query", self.rewrite)

    def search(self, client, model, q, collection_name, alpha, top_k):
        self.searched.append(q["lexical_query"])
        return list(self.results.get(q["lexical_query"], []))

    def rewrite(self, slm, tok, q):
        self.rewrites.append(q)
        return self.rewritten


def args(reranker):
    return ("client", "embedder", reranker, "slm", "tok")


class TestRetrieveSingle:
    plan = {"question": "Original?", "strategy": "once", "queries": [query("lex", "sem")]}

    def test_good_first_pass_does_not_rewrite(self, monkeypatch):
        h = Harness(monkeypatch, {"lex": docs(1, 2, 3)})
        reranker = FakeReranker({"u1": 0.95, "u2": 0.5, "u3": 0.1})
        out, cost = rt.retrieve_single(*args(reranker), self.plan, final_top_k=2)
        assert [d["uuid"] for d in out] == ["u1", "u2"]
        assert cost == {**ZERO_COSTS, "search_calls": 1, "embedding_calls": 1,
                        "reranker_calls": 1, "reranked_pairs": 3}
        assert h.rewrites == []

    def test_ranking_uses_the_original_question_not_the_retrieval_query(self, monkeypatch):
        Harness(monkeypatch, {"lex": docs(1)})
        reranker = FakeReranker(lambda q, d: 0.99)
        rt.retrieve_single(*args(reranker), self.plan)
        assert reranker.calls[0][0][0] == "Original?"

    def test_threshold_boundary_is_inclusive(self, monkeypatch):
        h = Harness(monkeypatch, {"lex": docs(1)})
        rt.retrieve_single(*args(FakeReranker({"u1": 0.8})), self.plan, relevance_threshold=0.8)
        assert h.rewrites == []

    def test_weak_first_pass_triggers_one_rewrite_and_merges(self, monkeypatch):
        h = Harness(monkeypatch, {"lex": docs(1, 2), "rewritten lex": docs(2, 3)})
        reranker = FakeReranker({"u1": 0.2, "u2": 0.3, "u3": 0.9})
        out, cost = rt.retrieve_single(*args(reranker), self.plan, final_top_k=5)
        assert h.searched == ["lex", "rewritten lex"] and len(h.rewrites) == 1
        assert [d["uuid"] for d in out] == ["u3", "u2", "u1"]
        assert cost == {"search_calls": 2, "embedding_calls": 2, "reranker_calls": 3,
                        "reranked_pairs": 2 + 2 + 3, "rewrite_calls": 1}
        # final merge pass is scored against the original question
        assert reranker.calls[-1][0][0] == "Original?"

    def test_no_candidates_triggers_rewrite_without_reranker_calls(self, monkeypatch):
        h = Harness(monkeypatch, {})
        out, cost = rt.retrieve_single(*args(FakeReranker({})), self.plan)
        assert out == [] and len(h.rewrites) == 1
        assert cost == {**ZERO_COSTS, "search_calls": 2, "embedding_calls": 2, "rewrite_calls": 1}

    def test_final_top_k_must_be_positive(self, monkeypatch):
        Harness(monkeypatch, {})
        with pytest.raises(ValueError):
            rt.retrieve_single(*args(FakeReranker({})), self.plan, final_top_k=0)


class TestRetrieveDecomposed:
    plan = {"question": "Original?", "strategy": "split",
            "queries": [query("lexA", "semA"), query("lexB", "semB")]}

    def test_guaranteed_slots_per_subquery_then_global_fill(self, monkeypatch):
        h = Harness(monkeypatch, {"lexA": docs(1, 2, 3), "lexB": docs(4, 5, 6)})
        scores = {"u1": 0.9, "u2": 0.85, "u3": 0.4, "u4": 0.95, "u5": 0.9, "u6": 0.3}
        reranker = FakeReranker(scores)
        out, cost = rt.retrieve_decomposed(*args(reranker), self.plan,
                                           subquery_top_k=1, final_context_top_k=3)
        assert h.rewrites == []
        ids = [d["uuid"] for d in out]
        # guaranteed: u1 (A), u4 (B); one global slot goes to best remaining (u5, 0.9)
        assert set(ids) == {"u1", "u4", "u5"} and ids[0] == "u4"
        assert cost == {**ZERO_COSTS, "search_calls": 2, "embedding_calls": 2,
                        "reranker_calls": 4, "reranked_pairs": 3 + 3 + 4 + 3}

    def test_shared_passage_uses_one_slot(self, monkeypatch):
        Harness(monkeypatch, {"lexA": docs(1, 2), "lexB": docs(1, 3)})
        reranker = FakeReranker({"u1": 0.99, "u2": 0.9, "u3": 0.85})
        out, _ = rt.retrieve_decomposed(*args(reranker), self.plan,
                                        subquery_top_k=1, final_context_top_k=4)
        ids = [d["uuid"] for d in out]
        assert sorted(ids) == ["u1", "u2", "u3"] and len(ids) == len(set(ids))

    def test_final_list_never_exceeds_budget(self, monkeypatch):
        Harness(monkeypatch, {"lexA": docs(1, 2, 3, 4), "lexB": docs(5, 6, 7, 8)})
        reranker = FakeReranker(lambda q, d: 0.9)
        out, _ = rt.retrieve_decomposed(*args(reranker), self.plan,
                                        subquery_top_k=2, final_context_top_k=5)
        assert len(out) == 5

    def test_only_the_weak_subquery_is_rewritten(self, monkeypatch):
        h = Harness(monkeypatch, {"lexA": docs(1), "lexB": docs(2), "rewritten lex": docs(3)})
        reranker = FakeReranker({"u1": 0.95, "u2": 0.1, "u3": 0.7})
        out, cost = rt.retrieve_decomposed(*args(reranker), self.plan,
                                           subquery_top_k=2, final_context_top_k=6)
        assert [q["lexical_query"] for q in h.rewrites] == ["lexB"]
        assert h.searched == ["lexA", "lexB", "rewritten lex"]
        assert cost["rewrite_calls"] == 1 and cost["search_calls"] == 3
        assert {d["uuid"] for d in out} == {"u1", "u2", "u3"}

    def test_empty_subquery_results_are_tolerated(self, monkeypatch):
        Harness(monkeypatch, {})
        out, cost = rt.retrieve_decomposed(*args(FakeReranker({})), self.plan,
                                           subquery_top_k=1, final_context_top_k=3)
        assert out == []
        assert cost["rewrite_calls"] == 2 and cost["reranker_calls"] == 0

    def test_budget_and_positivity_guards(self, monkeypatch):
        Harness(monkeypatch, {})
        with pytest.raises(ValueError, match="positive"):
            rt.retrieve_decomposed(*args(FakeReranker({})), self.plan, subquery_top_k=0)
        with pytest.raises(ValueError, match="budget"):
            rt.retrieve_decomposed(*args(FakeReranker({})), self.plan,
                                   subquery_top_k=3, final_context_top_k=5)
