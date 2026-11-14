"""Unit tests for adaptive_rag.planner.adaptive (SFT data, masking, inference)."""
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import adaptive_rag.models.lm as lm_module
from adaptive_rag.planner import adaptive
from adaptive_rag.planner.adaptive import (
    CompletionCollator,
    generate_adaptive_query,
    load_sft_examples,
    planner_messages,
    split_sft_examples,
    tokenize_sft_examples,
    train_planner,
)
from rag_fakes import FakeChatTokenizer, once_plan, plan_json, split_plan


def test_training_resume_verifies_data_and_settings(tmp_path, monkeypatch):
    import transformers

    checkpoint = tmp_path / "checkpoints" / "checkpoint-1"
    trainer = Mock()
    monkeypatch.setattr(transformers, "Trainer", Mock(return_value=trainer), raising=False)
    monkeypatch.setattr(transformers, "TrainingArguments", Mock(), raising=False)
    monkeypatch.setattr(transformers, "set_seed", Mock(), raising=False)
    monkeypatch.setitem(sys.modules, "transformers.trainer_utils", SimpleNamespace(
        get_last_checkpoint=lambda directory: str(checkpoint),
    ))
    model = Mock()
    tokenizer = SimpleNamespace(pad_token_id=0, save_pretrained=Mock())
    load_model = Mock(return_value=(model, tokenizer))
    monkeypatch.setattr(adaptive, "load_base_model", load_model)
    monkeypatch.setattr(adaptive, "configure_lora", lambda model, **kwargs: model)
    monkeypatch.setattr(adaptive, "tokenize_sft_examples", lambda *args, **kwargs: [])

    train, val = [example("train")], [example("val")]
    train_planner(train, val, tmp_path)
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")
    train_planner(train, val, tmp_path, resume_from_checkpoint=True)
    trainer.train.assert_called_with(resume_from_checkpoint=str(checkpoint.resolve()))

    with pytest.raises(ValueError, match="cannot be verified"):
        train_planner([example("train", "Changed?")], val, tmp_path, resume_from_checkpoint=True)
    with pytest.raises(ValueError, match="cannot be verified"):
        train_planner(train, val, tmp_path, resume_from_checkpoint=True, rank=8)
    assert load_model.call_count == 2


# ----------------------------------------------------------------------------- helpers
def sft_line(qid="q1", question="Who?", plan=None, strategy=None, **overrides):
    plan = plan or once_plan()
    record = {"question_id": qid, "question": question,
              "strategy": strategy or plan["strategy"], "target": json.dumps(plan)}
    record.update(overrides)
    return json.dumps(record)


def write_jsonl(path, *lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def example(qid="q1", question="Who wrote it?", plan=None):
    return {"question_id": qid, "question": question, "target": plan or once_plan()}


# ----------------------------------------------------------------------------- messages
class TestPlannerMessages:
    def test_structure(self):
        messages = planner_messages("Who painted it?")
        assert [m["role"] for m in messages] == ["system", "user"]
        assert messages[0]["content"] == adaptive.SYSTEM_PROMPT
        assert "Who painted it?" in messages[1]["content"]

    @pytest.mark.parametrize("bad", ["", "   ", None, 3])
    def test_rejects_blank_questions(self, bad):
        with pytest.raises(ValueError):
            planner_messages(bad)


# ----------------------------------------------------------------------------- loading
class TestLoadSftExamples:
    def test_loads_and_strips_extra_target_keys(self, tmp_path):
        plan = {**once_plan(), "question": "should be dropped"}
        path = write_jsonl(tmp_path / "t.jsonl", sft_line("q1", plan=plan))
        (record,) = load_sft_examples(path)
        assert record["question_id"] == "q1"
        assert set(record["target"]) == {"strategy", "queries"}

    def test_ignores_blank_lines(self, tmp_path):
        path = tmp_path / "t.jsonl"
        path.write_text(sft_line("q1") + "\n\n   \n" + sft_line("q2") + "\n", encoding="utf-8")
        assert [r["question_id"] for r in load_sft_examples(path)] == ["q1", "q2"]

    def test_question_ids_are_normalised_to_strings(self, tmp_path):
        path = write_jsonl(tmp_path / "t.jsonl", sft_line(qid=7))
        assert load_sft_examples(path)[0]["question_id"] == "7"

    def test_rejects_empty_file(self, tmp_path):
        path = tmp_path / "t.jsonl"
        path.write_text("\n", encoding="utf-8")
        with pytest.raises(ValueError, match="No SFT examples"):
            load_sft_examples(path)

    def test_rejects_duplicate_ids(self, tmp_path):
        path = write_jsonl(tmp_path / "t.jsonl", sft_line("q1"), sft_line("q1"))
        with pytest.raises(ValueError, match="line 2.*duplicate"):
            load_sft_examples(path)

    def test_rejects_strategy_label_mismatch(self, tmp_path):
        path = write_jsonl(tmp_path / "t.jsonl", sft_line("q1", plan=once_plan(), strategy="split"))
        with pytest.raises(ValueError, match="strategy does not match"):
            load_sft_examples(path)

    def test_rejects_invalid_plan_in_target(self, tmp_path):
        bad = {"strategy": "once", "queries": [once_plan()["queries"][0]] * 2}
        path = write_jsonl(tmp_path / "t.jsonl", sft_line("q1", plan=bad))
        with pytest.raises(ValueError, match="Invalid SFT record at line 1"):
            load_sft_examples(path)

    def test_rejects_malformed_json_and_missing_keys(self, tmp_path):
        with pytest.raises(ValueError, match="line 1"):
            load_sft_examples(write_jsonl(tmp_path / "a.jsonl", "{not json"))
        with pytest.raises(ValueError, match="line 1"):
            load_sft_examples(write_jsonl(tmp_path / "b.jsonl", json.dumps({"question": "x"})))

    def test_rejects_blank_question(self, tmp_path):
        path = write_jsonl(tmp_path / "t.jsonl", sft_line("q1", question="  "))
        with pytest.raises(ValueError, match="empty question"):
            load_sft_examples(path)


# ----------------------------------------------------------------------------- split
class TestSplitSftExamples:
    def records(self, n=10):
        return [example(f"q{i}") for i in range(n)]

    def test_sizes_and_disjointness(self):
        train, val = split_sft_examples(self.records(10), validation_fraction=0.2, seed=1)
        assert (len(train), len(val)) == (8, 2)
        ids = lambda rs: {r["question_id"] for r in rs}  # noqa: E731
        assert not ids(train) & ids(val)
        assert ids(train) | ids(val) == {f"q{i}" for i in range(10)}

    def test_deterministic_for_a_seed_and_input_unchanged(self):
        records = self.records(20)
        before = [r["question_id"] for r in records]
        a = split_sft_examples(records, seed=3)
        b = split_sft_examples(records, seed=3)
        assert a == b
        assert [r["question_id"] for r in records] == before

    def test_seed_changes_the_split(self):
        records = self.records(40)
        a = {r["question_id"] for r in split_sft_examples(records, seed=1)[1]}
        b = {r["question_id"] for r in split_sft_examples(records, seed=2)[1]}
        assert a != b

    def test_always_keeps_one_example_on_each_side(self):
        train, val = split_sft_examples(self.records(2), validation_fraction=0.01)
        assert len(train) == 1 and len(val) == 1
        train, val = split_sft_examples(self.records(2), validation_fraction=0.99)
        assert len(train) == 1 and len(val) == 1

    @pytest.mark.parametrize("fraction", [0, 1, -0.1, 1.5])
    def test_rejects_bad_fraction(self, fraction):
        with pytest.raises(ValueError):
            split_sft_examples(self.records(5), validation_fraction=fraction)

    def test_needs_two_records(self):
        with pytest.raises(ValueError):
            split_sft_examples(self.records(1))


# ----------------------------------------------------------------------------- tokenisation
class TestTokenizeSftExamples:
    def test_masks_prompt_and_supervises_completion(self):
        tokenizer = FakeChatTokenizer()
        (row,) = tokenize_sft_examples([example(question="Who?")], tokenizer)
        prefix = tokenizer.apply_chat_template(planner_messages("Who?"), tokenize=True,
                                               add_generation_prompt=True, return_dict=False)
        n = len(prefix)
        assert row["input_ids"][:n] == prefix
        assert row["labels"][:n] == [-100] * n
        assert row["labels"][n:] == row["input_ids"][n:]
        assert len(row["input_ids"]) == len(row["labels"]) == len(row["attention_mask"])
        assert set(row["attention_mask"]) == {1}

    def test_completion_is_the_compact_target_json_plus_end_of_turn(self):
        tokenizer = FakeChatTokenizer()
        plan = split_plan()
        (row,) = tokenize_sft_examples([example(plan=plan)], tokenizer)
        completion = "".join(chr(i) for i, lab in zip(row["input_ids"], row["labels"]) if lab != -100)
        assert completion == json.dumps(plan, ensure_ascii=False, separators=(",", ":")) + "<|im_end|>\n"

    def test_requests_plain_token_lists_from_the_chat_template(self):
        """Regression: transformers>=5 defaults apply_chat_template to return_dict=True
        (a BatchEncoding). Slicing/concatenating that crashes unless return_dict=False."""
        tokenizer = FakeChatTokenizer()  # defaults to return_dict=True like transformers 5
        rows = tokenize_sft_examples([example("q1"), example("q2")], tokenizer)
        assert len(rows) == 2
        assert tokenizer.calls and all(call["return_dict"] is False for call in tokenizer.calls)
        assert all(isinstance(i, int) for i in rows[0]["input_ids"])

    def test_rejects_examples_that_would_be_truncated(self):
        with pytest.raises(ValueError, match="truncating"):
            tokenize_sft_examples([example()], FakeChatTokenizer(), max_length=40)

    def test_rejects_tiny_max_length(self):
        with pytest.raises(ValueError, match="too small"):
            tokenize_sft_examples([example()], FakeChatTokenizer(), max_length=8)

    def test_detects_chat_template_prefix_mismatch(self):
        with pytest.raises(ValueError, match="prefix mismatch"):
            tokenize_sft_examples([example()], FakeChatTokenizer(mutate_prefix=True))

    def test_rejects_empty_completion(self):
        with pytest.raises(ValueError, match="Empty assistant completion"):
            tokenize_sft_examples([example()], FakeChatTokenizer(drop_assistant_content=True))


class TestCompletionCollator:
    def test_requires_a_pad_token(self):
        tokenizer = FakeChatTokenizer()
        tokenizer.pad_token_id = None
        with pytest.raises(ValueError, match="pad token"):
            CompletionCollator(tokenizer)

    def test_pads_inputs_masks_and_labels(self):
        torch = pytest.importorskip("torch")
        if not hasattr(torch, "tensor") or type(torch).__name__ == "MagicMock":
            pytest.skip("real torch required")
        collator = CompletionCollator(FakeChatTokenizer())
        batch = collator([
            {"input_ids": [5, 6, 7], "attention_mask": [1, 1, 1], "labels": [-100, 6, 7]},
            {"input_ids": [8], "attention_mask": [1], "labels": [8]},
        ])
        assert batch["input_ids"].tolist() == [[5, 6, 7], [8, 0, 0]]
        assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
        assert batch["labels"].tolist() == [[-100, 6, 7], [8, -100, -100]]


# ----------------------------------------------------------------------------- training guards
class TestTrainPlannerValidation:
    def test_requires_non_empty_train_and_validation_sets(self, tmp_path):
        with pytest.raises(ValueError, match="Nonempty"):
            train_planner([], [example()], tmp_path)
        with pytest.raises(ValueError, match="Nonempty"):
            train_planner([example()], [], tmp_path)

    def test_rejects_overlapping_ids_before_loading_any_model(self, tmp_path):
        with pytest.raises(ValueError, match="overlap"):
            train_planner([example("q1")], [example("q1")], tmp_path)
        assert not (tmp_path / "adapter").exists()


# ----------------------------------------------------------------------------- inference
class Scripted:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, model, tokenizer, messages, max_new_tokens=None, **kwargs):
        self.calls.append({"messages": messages, "max_new_tokens": max_new_tokens})
        return self.responses.pop(0)


@pytest.fixture
def script(monkeypatch):
    def install(*responses):
        fake = Scripted(*responses)
        monkeypatch.setattr(lm_module, "generate", fake, raising=False)
        return fake
    return install


class TestGenerateAdaptiveQuery:
    def test_returns_plan_with_question_and_attempt_count(self, script):
        fake = script(plan_json(once_plan()))
        plan = generate_adaptive_query(None, None, "Who wrote it?")
        assert plan["strategy"] == "once" and plan["question"] == "Who wrote it?"
        assert plan["planner_attempts"] == 1
        assert len(fake.calls) == 1
        assert fake.calls[0]["messages"] == planner_messages("Who wrote it?")

    def test_handles_fenced_json(self, script):
        script("```json\n" + plan_json(split_plan()) + "\n```")
        assert generate_adaptive_query(None, None, "q?")["strategy"] == "split"

    def test_retries_invalid_output_once_then_succeeds(self, script):
        fake = script("garbage", plan_json(once_plan()))
        plan = generate_adaptive_query(None, None, "q?")
        assert plan["planner_attempts"] == 2
        assert "previous response was invalid" in fake.calls[1]["messages"][-1]["content"]

    def test_raises_after_retries_are_exhausted(self, script):
        script("garbage", "more garbage")
        with pytest.raises(ValueError, match="invalid after 2 attempts"):
            generate_adaptive_query(None, None, "q?")

    def test_zero_retries_means_single_attempt(self, script):
        fake = script("garbage")
        with pytest.raises(ValueError, match="after 1 attempts"):
            generate_adaptive_query(None, None, "q?", max_retries=0)
        assert len(fake.calls) == 1

    def test_forced_strategy_adds_instruction_and_enforces_result(self, script):
        fake = script(plan_json(split_plan()))
        plan = generate_adaptive_query(None, None, "q?", forced_strategy="split")
        assert plan["strategy"] == "split"
        instruction = fake.calls[0]["messages"][-1]["content"]
        assert "strategy 'split'" in instruction and "2 retrieval" in instruction

    def test_forced_once_asks_for_a_single_query(self, script):
        fake = script(plan_json(once_plan()))
        generate_adaptive_query(None, None, "q?", forced_strategy="once")
        assert "1 retrieval" in fake.calls[0]["messages"][-1]["content"]

    def test_forced_strategy_mismatch_is_rejected(self, script):
        script(plan_json(once_plan()), plan_json(once_plan()))
        with pytest.raises(ValueError, match="Expected split"):
            generate_adaptive_query(None, None, "q?", forced_strategy="split")

    def test_argument_validation(self, script):
        script()
        with pytest.raises(ValueError, match="forced_strategy"):
            generate_adaptive_query(None, None, "q?", forced_strategy="both")
        with pytest.raises(ValueError, match="max_retries"):
            generate_adaptive_query(None, None, "q?", max_retries=-1)

    def test_blank_question_is_rejected(self, script):
        script()
        with pytest.raises(ValueError):
            generate_adaptive_query(None, None, "  ")
