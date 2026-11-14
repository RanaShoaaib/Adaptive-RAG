"""Unit tests for adaptive_rag.embedding.embedder."""
import numpy as np

from adaptive_rag import config as cfg
from adaptive_rag.embedding import embedder as emb


class FakeEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return np.array([[float(len(t)), 1.0] for t in texts], dtype="float32")


def test_queries_get_the_query_prefix_and_normalization():
    enc = FakeEncoder()
    out = emb.embed_queries(enc, ["a", "bb"])
    (texts, kwargs), = enc.calls
    assert texts == [f"{cfg.QUERY_PREFIX}a", f"{cfg.QUERY_PREFIX}bb"]
    assert kwargs["normalize_embeddings"] is True
    assert kwargs["batch_size"] == cfg.EMBEDDING_BATCH_SIZE
    assert kwargs["show_progress_bar"] is False
    assert isinstance(out, list) and isinstance(out[0], list)


def test_passages_get_the_passage_prefix_only():
    enc = FakeEncoder()
    emb.embed_passages(enc, ["Title\nbody"])
    assert enc.calls[0][0] == [f"{cfg.PASSAGE_PREFIX}Title\nbody"]


def test_output_order_matches_input_order():
    out = emb.embed_passages(FakeEncoder(), ["x", "xxx", "xx"])
    assert [v[0] - len(cfg.PASSAGE_PREFIX) for v in out] == [1.0, 3.0, 2.0]


def test_empty_input_returns_empty_list():
    class Empty:
        def encode(self, texts, **kw):
            return np.zeros((0, 2))
    assert emb.embed_queries(Empty(), []) == []


def test_query_and_passage_paths_differ_only_by_prefix_constants():
    q, p = FakeEncoder(), FakeEncoder()
    emb.embed_queries(q, ["same"])
    emb.embed_passages(p, ["same"])
    assert q.calls[0][0][0].endswith("same") and p.calls[0][0][0].endswith("same")
    assert q.calls[0][1] == p.calls[0][1]


def test_get_embedding_model_forwards_the_name(monkeypatch):
    seen = []
    monkeypatch.setattr(emb, "SentenceTransformer", lambda name: seen.append(name) or "model")
    assert emb.get_embedding_model("custom") == "model"
    assert emb.get_embedding_model() == "model"
    assert seen == ["custom", cfg.EMBEDDING_MODEL]
