"""Unit tests for adaptive_rag.data.prepare_data (corpus building, normalization, sampling)."""
import uuid
from collections import Counter

import pytest

from adaptive_rag.data import prepare_data as pdm
from rag_fakes import FakeDataset


class TestPassageIds:
    def test_deterministic_and_content_addressed(self):
        a = pdm.make_passage_id("T", "text")
        assert a == pdm.make_passage_id("T", "text")
        assert a != pdm.make_passage_id("T", "other")
        assert a != pdm.make_passage_id("U", "text")
        assert str(uuid.UUID(a)) == a


    def test_chunk_joins_and_strips_sentences(self):
        assert pdm.chunk_from_sentence_list([" A. ", "B.\n", "C."]) == "A. B. C."
        assert pdm.chunk_from_sentence_list([]) == ""


class TestHotpotChunks:
    def test_pairs_titles_with_sentences(self):
        chunks = pdm.make_hotpotqa_chunks(["T1", "T2"], [["a", "b"], ["c"]])
        assert [(c["title"], c["text"]) for c in chunks] == [("T1", "a b"), ("T2", "c")]
        assert chunks[0]["passage_id"] == pdm.make_passage_id("T1", "a b")

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="must match"):
            pdm.make_hotpotqa_chunks(["T1"], [])


def hotpot_row(qid, titles, sentences, support):
    return {"id": qid, "question": f"{qid}?", "answer": "ans",
            "context": {"title": titles, "sentences": sentences},
            "supporting_facts": {"title": support, "sent_id": [0] * len(support)}}


class TestBuildHotpotCorpus:
    def dataset(self):
        return FakeDataset([
            hotpot_row("q1", ["A", "B", "C"], [["a."], ["b."], ["c."]], ["A", "C", "A"]),
            hotpot_row("q2", ["B", "D"], [["b."], ["d."]], ["D"]),
        ])

    def test_gold_ids_follow_supporting_titles_in_context_order_without_duplicates(self):
        updated, chunks = pdm.build_hotpotqa_corpus(self.dataset())
        by_title = {c["title"]: c["passage_id"] for c in chunks}
        assert updated[0]["gold_passage_ids"] == [by_title["A"], by_title["C"]]
        assert updated[1]["gold_passage_ids"] == [by_title["D"]]

    def test_chunks_cover_every_context_passage(self):
        _, chunks = pdm.build_hotpotqa_corpus(self.dataset())
        assert len(chunks) == 5  # shared passage B appears twice before dedup

    def test_missing_supporting_title_raises(self):
        bad = FakeDataset([hotpot_row("q1", ["A"], [["a."]], ["Z"])])
        with pytest.raises(ValueError, match="missing from context"):
            pdm.build_hotpotqa_corpus(bad)

    def test_requires_columns(self):
        with pytest.raises(ValueError, match="HotpotQA"):
            pdm.build_hotpotqa_corpus(FakeDataset([{"id": 1}]))


class TestBuildSquadCorpus:
    def test_gold_id_matches_chunk_id(self):
        ds = FakeDataset([{"title": "T", "context": "ctx 1", "id": "1"},
                          {"title": "T", "context": "ctx 2", "id": "2"}])
        updated, chunks = pdm.build_squad_corpus(ds)
        assert [c["passage_id"] for c in chunks] == [r["gold_passage_ids"][0] for r in updated]
        assert chunks[0]["passage_id"] == pdm.make_passage_id("T", "ctx 1")
        assert chunks[1]["text"] == "ctx 2"

    def test_requires_columns(self):
        with pytest.raises(ValueError, match="SQuAD"):
            pdm.build_squad_corpus(FakeDataset([{"title": "T"}]))


class TestCombineCorpus:
    def test_first_occurrence_wins_and_order_is_stable(self):
        a = {"passage_id": "1", "title": "first", "text": "x"}
        dup = {"passage_id": "1", "title": "second", "text": "x"}
        b = {"passage_id": "2", "title": "b", "text": "y"}
        assert pdm.combine_and_deduplicate_corpus([[a, b], [dup]]) == [a, b]

    def test_empty(self):
        assert pdm.combine_and_deduplicate_corpus([]) == []


class TestNormalizeData:
    def test_hotpotqa_wraps_single_answer(self):
        out = pdm.normalize_data({"id": ["1", "2"], "question": ["a", "b"], "answer": ["x", "y"],
                                  "gold_passage_ids": [["p"], ["q"]]}, " HotpotQA ")
        assert out["answers"] == [["x"], ["y"]]
        assert out["source"] == ["hotpotqa", "hotpotqa"]
        assert set(out) == {"id", "source", "question", "answers", "gold_passage_ids"}

    def test_squad_keeps_reference_list(self):
        out = pdm.normalize_data({"id": ["1"], "question": ["a"],
                                  "answers": [{"text": ["x", "y"], "answer_start": [0, 1]}],
                                  "gold_passage_ids": [["p"]]}, "squad")
        assert out["answers"] == [["x", "y"]] and out["source"] == ["squad"]

    def test_unknown_type(self):
        with pytest.raises(ValueError, match="Unsupported"):
            pdm.normalize_data({"id": []}, "nq")


class TestSampleDataset:
    def dataset(self, hotpot=60, squad=20):
        rows = [{"id": f"h{i}", "source": "hotpotqa"} for i in range(hotpot)]
        rows += [{"id": f"s{i}", "source": "squad"} for i in range(squad)]
        return FakeDataset(rows)

    def counts(self, ds):
        return Counter(r["source"] for r in ds)

    def test_stratified_preserves_proportions_and_exact_size(self):
        out = pdm.sample_dataset(self.dataset(), 20, strategy="stratified")
        assert len(out) == 20 and self.counts(out) == {"hotpotqa": 15, "squad": 5}

    def test_balanced_allocates_evenly(self):
        out = pdm.sample_dataset(self.dataset(), 20, strategy="balanced")
        assert self.counts(out) == {"hotpotqa": 10, "squad": 10}

    def test_largest_remainder_sums_to_n(self):
        out = pdm.sample_dataset(self.dataset(50, 50), 7, strategy="balanced")
        assert len(out) == 7

    def test_deterministic_for_a_seed_and_distinct_rows(self):
        ds = self.dataset()
        a = [r["id"] for r in pdm.sample_dataset(ds, 30, seed=1)]
        assert a == [r["id"] for r in pdm.sample_dataset(ds, 30, seed=1)]
        assert a != [r["id"] for r in pdm.sample_dataset(ds, 30, seed=2)]
        assert len(set(a)) == 30

    def test_balanced_with_too_small_source_raises(self):
        with pytest.raises(ValueError, match="Insufficient"):
            pdm.sample_dataset(self.dataset(60, 2), 20, strategy="balanced")

    @pytest.mark.parametrize("n", [0, -1, 81])
    def test_n_bounds(self, n):
        with pytest.raises(ValueError, match="between"):
            pdm.sample_dataset(self.dataset(), n)

    def test_strategy_and_source_column_validated(self):
        with pytest.raises(ValueError, match="strategy"):
            pdm.sample_dataset(self.dataset(), 5, strategy="random")
        with pytest.raises(ValueError, match="source"):
            pdm.sample_dataset(FakeDataset([{"id": 1}]), 1)


class TestSubsetIO:
    def test_load_missing_subset(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            pdm.load_subset(tmp_path / "nope.parquet")

    def test_save_creates_parent_and_delegates(self, tmp_path):
        calls = []

        class DS:
            def to_parquet(self, path):
                calls.append(path)
                open(path, "wb").close()

        target = tmp_path / "a" / "b" / "s.parquet"
        pdm.save_subset(DS(), target)
        assert calls == [str(target)] and target.exists()


class TestPrepareSubsets:
    def test_rejects_empty_and_unknown(self):
        with pytest.raises(ValueError, match="empty"):
            pdm.prepare_subsets({})
        with pytest.raises(ValueError, match="Unknown splits"):
            pdm.prepare_subsets({"dev": {"n": 1, "strategy": "balanced"}})

    def test_only_requested_splits_are_written(self, monkeypatch, tmp_path):
        ds = FakeDataset([{"id": str(i), "source": "squad"} for i in range(10)])
        monkeypatch.setattr(pdm, "get_data_and_corpus", lambda: (ds, ds, ds, []))
        monkeypatch.setattr(pdm, "SUBSET_PATHS", {k: tmp_path / f"{k}.parquet" for k in ("train", "val", "test")})
        saved = []
        monkeypatch.setattr(pdm, "save_subset", lambda subset, path: saved.append((len(subset), path.name)))
        pdm.prepare_subsets({"val": {"n": 4, "strategy": "stratified"}})
        assert saved == [(4, "val.parquet")]
