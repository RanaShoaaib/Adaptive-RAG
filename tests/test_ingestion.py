"""Unit tests for adaptive_rag.database.ingestion and weaviate_client."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from adaptive_rag.database import ingestion as ing
from adaptive_rag.database import weaviate_client as wc
from conftest import STUBBED
from rag_fakes import FakeWeaviateClient


def passages(n):
    return [{"passage_id": f"id{i}", "title": f"T{i}", "text": f"body {i}"} for i in range(n)]


class TestBatchUpload:
    def test_uploads_ids_properties_and_named_vector(self):
        client = FakeWeaviateClient()
        ing.batch_upload(client, passages(2), [[0.1], [0.2]], "COL")
        assert client.collection.added == [
            {"uuid": "id0", "properties": {"title": "T0", "text": "body 0"}, "vector": {"vector": [0.1]}},
            {"uuid": "id1", "properties": {"title": "T1", "text": "body 1"}, "vector": {"vector": [0.2]}},
        ]

    def test_empty_corpus_is_a_noop(self):
        client = FakeWeaviateClient()
        ing.batch_upload(client, [], [])
        assert client.collection.added == []

    def test_length_mismatch_is_rejected_before_any_upload(self):
        client = FakeWeaviateClient()
        with pytest.raises(ValueError, match="must match"):
            ing.batch_upload(client, passages(2), [[0.1]])
        assert client.collection.added == []

    def test_failed_objects_raise_with_count(self, capsys):
        client = FakeWeaviateClient()
        client.collection.batch.failed_objects = [MagicMock(message="boom")] * 7
        with pytest.raises(RuntimeError, match="7 objects failed"):
            ing.batch_upload(client, passages(1), [[0.1]])
        assert capsys.readouterr().out.count("boom") == 5


def test_create_collection_defines_title_text_and_self_provided_vector():
    client = MagicMock()
    ing.create_collection(client, "COL")
    kwargs = client.collections.create.call_args.kwargs
    assert kwargs["name"] == "COL"
    assert len(kwargs["properties"]) == 2


class TestIngestCorpus:
    @pytest.fixture
    def env(self, monkeypatch):
        state = {"embedded": [], "uploads": [], "created": []}
        monkeypatch.setattr(ing, "INGEST_CHUNK_SIZE", 2)
        monkeypatch.setattr(ing, "embed_passages",
                            lambda model, texts: state["embedded"].append(texts) or [[0.0]] * len(texts))

        def upload(client, docs, vectors, name):
            state["uploads"].append([d["passage_id"] for d in docs])
            client.count += len(docs)

        monkeypatch.setattr(ing, "batch_upload", upload)
        monkeypatch.setattr(ing, "create_collection",
                            lambda client, name: state["created"].append(name) or setattr(client, "exists_flag", True))
        return state

    def test_fresh_ingest_chunks_and_embeds_title_plus_text(self, env):
        client = FakeWeaviateClient()
        ing.ingest_corpus(client, passages(5), "M", "COL")
        assert env["created"] == ["COL"]
        assert env["uploads"] == [["id0", "id1"], ["id2", "id3"], ["id4"]]
        assert env["embedded"][0] == ["T0\nbody 0", "T1\nbody 1"]

    def test_complete_collection_is_skipped(self, env):
        client = FakeWeaviateClient(exists=True, count=3)
        ing.ingest_corpus(client, passages(3), "M", "COL")
        assert env["uploads"] == [] and client.deleted == []

    def test_incomplete_collection_is_rebuilt(self, env):
        client = FakeWeaviateClient(exists=True, count=1)
        ing.ingest_corpus(client, passages(3), "M", "COL")
        assert client.deleted == ["COL"] and client.count == 3

    def test_force_rebuilds_even_when_complete(self, env):
        client = FakeWeaviateClient(exists=True, count=3)
        ing.ingest_corpus(client, passages(3), "M", "COL", force_ingest=True)
        assert client.deleted == ["COL"] and len(env["uploads"]) == 2

    def test_empty_corpus_rejected(self, env):
        with pytest.raises(ValueError, match="empty"):
            ing.ingest_corpus(FakeWeaviateClient(), [], "M")

    def test_bad_chunk_size_rejected(self, env, monkeypatch):
        monkeypatch.setattr(ing, "INGEST_CHUNK_SIZE", 0)
        with pytest.raises(ValueError, match="INGEST_CHUNK_SIZE"):
            ing.ingest_corpus(FakeWeaviateClient(), passages(1), "M")

    def test_final_count_mismatch_is_detected(self, env, monkeypatch):
        monkeypatch.setattr(ing, "batch_upload", lambda client, docs, vectors, name: None)
        with pytest.raises(RuntimeError, match="incomplete"):
            ing.ingest_corpus(FakeWeaviateClient(), passages(2), "M")


needs_real_dotenv = pytest.mark.skipif(
    "dotenv" in STUBBED, reason="python-dotenv is not installed (stubbed)"
)


class TestWeaviateClient:
    @pytest.fixture(autouse=True)
    def isolated_env(self, monkeypatch, tmp_path):
        """Never read the developer's real .env; start with no WEAVIATE_* variables."""
        monkeypatch.setattr(wc, "ROOT_DIR", tmp_path)
        for name in ("WEAVIATE_API_KEY", "WEAVIATE_HTTP_HOST", "WEAVIATE_GRPC_HOST",
                     "WEAVIATE_HTTP_PORT", "WEAVIATE_GRPC_PORT",
                     "WEAVIATE_HTTP_SECURE", "WEAVIATE_GRPC_SECURE"):
            monkeypatch.delenv(name, raising=False)

    def write_env(self, tmp_path, text):
        (tmp_path / ".env").write_text(text, encoding="utf-8")

    @needs_real_dotenv
    def test_dotenv_is_loaded_when_the_module_is_used_alone(self, monkeypatch, tmp_path):
        self.write_env(tmp_path, "WEAVIATE_HTTP_HOST=http.env.test\nWEAVIATE_GRPC_HOST=grpc.env.test\n")
        settings = wc._connection_settings()
        assert settings["http_host"] == "http.env.test" and settings["grpc_host"] == "grpc.env.test"
        assert settings["http_port"] == 443 and settings["grpc_secure"] is True
        monkeypatch.delenv("WEAVIATE_HTTP_HOST")
        monkeypatch.delenv("WEAVIATE_GRPC_HOST")

    @needs_real_dotenv
    def test_api_key_comes_from_dotenv_too(self, monkeypatch, tmp_path):
        self.write_env(tmp_path, "WEAVIATE_API_KEY=from-file\nWEAVIATE_HTTP_HOST=h\nWEAVIATE_GRPC_HOST=g\n")
        seen = {}
        monkeypatch.setattr(wc.weaviate, "connect_to_custom", lambda **kw: seen.update(kw) or "client", raising=False)
        monkeypatch.setattr(wc, "Auth", SimpleNamespace(api_key=lambda key: key))
        assert wc.get_weaviate_client() == "client"
        assert seen["auth_credentials"] == "from-file"
        for name in ("WEAVIATE_API_KEY", "WEAVIATE_HTTP_HOST", "WEAVIATE_GRPC_HOST"):
            monkeypatch.delenv(name)

    @needs_real_dotenv
    def test_existing_environment_wins_over_dotenv(self, monkeypatch, tmp_path):
        self.write_env(tmp_path, "WEAVIATE_HTTP_HOST=from-file\nWEAVIATE_GRPC_HOST=g\n")
        monkeypatch.setenv("WEAVIATE_HTTP_HOST", "from-shell")
        assert wc._connection_settings()["http_host"] == "from-shell"
        monkeypatch.delenv("WEAVIATE_GRPC_HOST")

    def test_missing_host_is_a_clear_error(self):
        with pytest.raises(ValueError, match="WEAVIATE_HTTP_HOST is not set"):
            wc._connection_settings()

    @pytest.mark.parametrize("name,value", [("WEAVIATE_HTTP_PORT", "abc"), ("WEAVIATE_GRPC_PORT", "70000"),
                                            ("WEAVIATE_HTTP_PORT", "0")])
    def test_invalid_ports_are_rejected(self, monkeypatch, name, value):
        monkeypatch.setenv("WEAVIATE_HTTP_HOST", "h")
        monkeypatch.setenv("WEAVIATE_GRPC_HOST", "g")
        monkeypatch.setenv(name, value)
        with pytest.raises(ValueError, match="between 1 and 65535"):
            wc._connection_settings()

    def test_invalid_tls_flag_is_rejected(self, monkeypatch):
        monkeypatch.setenv("WEAVIATE_HTTP_HOST", "h")
        monkeypatch.setenv("WEAVIATE_GRPC_HOST", "g")
        monkeypatch.setenv("WEAVIATE_HTTP_SECURE", "maybe")
        with pytest.raises(ValueError, match="true or false"):
            wc._connection_settings()

    def test_endpoint_fingerprint_tracks_endpoint_without_exposing_it(self, monkeypatch):
        monkeypatch.setenv("WEAVIATE_HTTP_HOST", "http.one.test")
        monkeypatch.setenv("WEAVIATE_GRPC_HOST", "grpc.one.test")
        first = wc.endpoint_fingerprint()
        assert first == wc.endpoint_fingerprint()
        assert "one.test" not in first and len(first) == 64
        monkeypatch.setenv("WEAVIATE_HTTP_HOST", "http.two.test")
        assert wc.endpoint_fingerprint() != first

    def test_missing_api_key_is_a_clear_error(self, monkeypatch):
        monkeypatch.delenv("WEAVIATE_API_KEY", raising=False)
        monkeypatch.setattr(wc.weaviate, "connect_to_custom", lambda **kw: pytest.fail("must not connect"), raising=False)
        with pytest.raises(ValueError, match="WEAVIATE_API_KEY"):
            wc.get_weaviate_client()

    def test_connects_with_configured_endpoints(self, monkeypatch):
        monkeypatch.setenv("WEAVIATE_API_KEY", "secret")
        monkeypatch.setenv("WEAVIATE_HTTP_HOST", "http.example.test")
        monkeypatch.setenv("WEAVIATE_GRPC_HOST", "grpc.example.test")
        monkeypatch.setenv("WEAVIATE_GRPC_PORT", "50051")
        monkeypatch.setenv("WEAVIATE_HTTP_SECURE", "false")
        seen = {}
        monkeypatch.setattr(wc.weaviate, "connect_to_custom", lambda **kw: seen.update(kw) or "client", raising=False)
        assert wc.get_weaviate_client() == "client"
        assert seen["http_host"] == "http.example.test"
        assert seen["grpc_port"] == 50051
        assert seen["http_secure"] is False

    def test_session_closes_client_on_success_and_on_error(self, monkeypatch):
        clients = []

        def make():
            c = FakeWeaviateClient()
            clients.append(c)
            return c

        monkeypatch.setattr(wc, "get_weaviate_client", make)
        with wc.weaviate_session() as c:
            assert c is clients[0] and not c.closed
        assert clients[0].closed
        with pytest.raises(RuntimeError):
            with wc.weaviate_session():
                raise RuntimeError("fail")
        assert clients[1].closed
