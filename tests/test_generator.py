"""Unit tests for adaptive_rag.generation.generator."""
from adaptive_rag import config as cfg
from adaptive_rag.generation import generator as gen
from rag_fakes import docs


def test_prompt_structure_and_numbered_context():
    msgs = gen.generate_augmented_prompt("Who?", docs(1, 2))
    assert [m["role"] for m in msgs] == ["system", "user"]
    user = msgs[1]["content"]
    assert user.startswith("Question: Who?\n\nContext:\n")
    assert "[1] Title: T1\ndoc u1\n\n[2] Title: T2\ndoc u2" in user
    assert "Unknown" in msgs[0]["content"]


def test_empty_context_still_produces_a_prompt():
    assert gen.generate_augmented_prompt("Q?", [])[1]["content"].endswith("Context:\n")


def test_generate_answer_strips_and_uses_answer_token_budget(monkeypatch):
    seen = {}

    def fake_generate(model, tok, messages, max_new_tokens):
        seen.update(model=model, tok=tok, messages=messages, n=max_new_tokens)
        return "  Paris \n"

    monkeypatch.setattr(gen, "generate", fake_generate)
    assert gen.generate_answer("M", "T", "Capital?", docs(1)) == "Paris"
    assert seen["n"] == cfg.ANSWERING_MAX_NEW_TOKENS
    assert seen["model"] == "M" and seen["tok"] == "T"
    assert "Capital?" in seen["messages"][1]["content"]
