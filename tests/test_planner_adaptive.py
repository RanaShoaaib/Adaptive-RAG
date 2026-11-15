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
    DECISION_PREFIX,
    decision_prefix_ids,
    decision_prior_baseline,
    decision_token_ids,
    generate_adaptive_query,
    load_sft_examples,
    planner_loss_terms,
    planner_messages,
    roc_auc,
    score_split_probabilities,
    split_sft_examples,
    threshold_for_budget,
    tokenize_sft_examples,
    train_planner,
)
from rag_fakes import FakeChatTokenizer, once_plan, plan_json, split_plan


def test_training_resume_verifies_data_and_settings(tmp_path, monkeypatch):
    import transformers

    checkpoint = tmp_path / "checkpoints" / "checkpoint-1"
    trainer = Mock()
    build_trainer = Mock(return_value=trainer)
    monkeypatch.setattr(adaptive, "build_planner_trainer", build_trainer)
    monkeypatch.setattr(adaptive, "decision_token_ids", lambda tokenizer: {"once": 1, "split": 2})
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
    with pytest.raises(ValueError, match="cannot be verified"):
        train_planner(train, val, tmp_path, resume_from_checkpoint=True, decision_weight=0.5)
    assert load_model.call_count == 2
    kwargs = build_trainer.call_args.kwargs
    assert kwargs["decision_weight"] == 0.25 and kwargs["label_smoothing"] == 0.05


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


class TestDecisionPosition:
    def test_decision_token_is_the_first_char_of_the_strategy_value(self):
        tokenizer = FakeChatTokenizer()
        rows = tokenize_sft_examples([example("a", plan=once_plan()), example("b", plan=split_plan())], tokenizer)
        once_row, split_row = rows
        assert chr(once_row["input_ids"][once_row["decision_pos"]]) == "o" and once_row["decision_label"] == 0
        assert chr(split_row["input_ids"][split_row["decision_pos"]]) == "s" and split_row["decision_label"] == 1
        for row in rows:
            assert row["labels"][row["decision_pos"]] == row["input_ids"][row["decision_pos"]]

    def test_tokens_before_the_decision_are_the_json_prefix(self):
        tokenizer = FakeChatTokenizer()
        assert "".join(map(chr, decision_prefix_ids(tokenizer))) == DECISION_PREFIX
        assert decision_token_ids(tokenizer) == {"once": ord("o"), "split": ord("s")}

    def test_rejects_tokenizers_that_merge_the_decision_with_the_prefix(self):
        class Merging(FakeChatTokenizer):
            def decode(self, ids, skip_special_tokens=False, **kwargs):
                return super().decode(ids)

            def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False,
                                    return_dict=True, **kwargs):
                ids = super().apply_chat_template(messages, tokenize, add_generation_prompt, False)
                if not tokenize:
                    return ids
                text = "".join(map(chr, ids)).replace('":"once', '\x01').replace('":"split', '\x02')
                return [ord(c) for c in text]
        with pytest.raises(ValueError):
            tokenize_sft_examples([example()], Merging())


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
        assert "decision_pos" not in batch

    def test_passes_decision_fields_through(self):
        torch = pytest.importorskip("torch")
        collator = CompletionCollator(FakeChatTokenizer())
        batch = collator([
            {"input_ids": [5, 6], "attention_mask": [1, 1], "labels": [-100, 6], "decision_pos": 1, "decision_label": 1},
            {"input_ids": [8], "attention_mask": [1], "labels": [8], "decision_pos": 0, "decision_label": 0},
        ])
        assert batch["decision_pos"].tolist() == [1, 0]
        assert batch["decision_label"].tolist() == [1.0, 0.0]


class TestLoss:
    @staticmethod
    def setup():
        torch = pytest.importorskip("torch")
        if type(torch).__name__ == "MagicMock" or not hasattr(torch, "randn"):
            pytest.skip("real torch required")
        gen = torch.Generator().manual_seed(0)
        logits = torch.randn(2, 6, 8, generator=gen)
        labels = torch.tensor([[-100, -100, 3, 4, 5, 6], [-100, 2, 3, 4, 5, -100]])
        decision_pos = torch.tensor([2, 2])
        decision_label = torch.tensor([1.0, 0.0])
        return torch, logits, labels, decision_pos, decision_label, {"once": 3, "split": 4}

    def test_matches_manual_computation_and_excludes_decision_from_plan(self):
        torch, logits, labels, pos, y, ids = self.setup()
        decision, plan, p = planner_loss_terms(logits, labels, pos, y, ids)
        margin = torch.stack([logits[b, 1, ids["split"]] - logits[b, 1, ids["once"]] for b in range(2)])
        expected = torch.nn.functional.binary_cross_entropy_with_logits(margin, y)
        assert torch.allclose(decision, expected)
        assert torch.allclose(p, torch.sigmoid(margin))
        losses = []
        for b in range(2):
            for t in range(5):  # logits[t] predicts labels[t + 1]
                target = labels[b, t + 1]
                if target != -100 and t + 1 != 2:
                    losses.append(torch.nn.functional.cross_entropy(logits[b, t:t + 1], target.view(1)))
        assert torch.allclose(plan, torch.stack(losses).mean())

    def test_plan_loss_ignores_the_decision_step_logits(self):
        torch, logits, labels, pos, y, ids = self.setup()
        _, plan_a, _ = planner_loss_terms(logits, labels, pos, y, ids)
        changed = logits.clone()
        changed[:, 1] += 5.0
        _, plan_b, _ = planner_loss_terms(changed, labels, pos, y, ids)
        assert torch.allclose(plan_a, plan_b)

    def test_label_smoothing_changes_only_the_decision_term(self):
        torch, logits, labels, pos, y, ids = self.setup()
        d0, p0, _ = planner_loss_terms(logits, labels, pos, y, ids)
        d1, p1, _ = planner_loss_terms(logits, labels, pos, y, ids, label_smoothing=0.05)
        assert torch.allclose(p0, p1) and not torch.allclose(d0, d1)
        margin = torch.stack([logits[b, 1, 4] - logits[b, 1, 3] for b in range(2)])
        smoothed = y * 0.95 + 0.025
        assert torch.allclose(d1, torch.nn.functional.binary_cross_entropy_with_logits(margin, smoothed))

    def test_gradient_reaches_the_decision_logits(self):
        torch, logits, labels, pos, y, ids = self.setup()
        logits.requires_grad_(True)
        decision, _, _ = planner_loss_terms(logits, labels, pos, y, ids)
        decision.backward()
        assert logits.grad[:, 1, [3, 4]].abs().sum() > 0 and logits.grad[:, 0].abs().sum() == 0


class TestDecisionHelpers:
    def test_roc_auc(self):
        assert roc_auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
        assert roc_auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
        assert roc_auc([0.5, 0.5], [1, 0]) == 0.5
        assert roc_auc([0.5, 0.6], [1, 1]) is None

    def test_prior_baseline_is_the_entropy_of_the_train_rate(self):
        import math
        train = [example(f"t{i}", plan=split_plan() if i == 0 else once_plan()) for i in range(4)]
        val = [example("v1", plan=once_plan()), example("v2", plan=split_plan())]
        result = decision_prior_baseline(train, val)
        assert result["train_split_rate"] == 0.25
        assert result["prior_decision_loss"] == pytest.approx((-math.log(0.75) - math.log(0.25)) / 2)

    def test_threshold_routes_at_least_the_budget_and_is_label_free(self):
        probs = [i / 100 for i in range(100)]
        tau = threshold_for_budget(probs, 0.10)
        assert tau == 0.90 and sum(p >= tau for p in probs) == 10
        assert threshold_for_budget([0.2, 0.2, 0.2], 0.10) == 0.2
        assert threshold_for_budget(probs, 1.0) == 0.0

    @pytest.mark.parametrize("budget", [0, -0.1, 1.5])
    def test_threshold_rejects_bad_budget(self, budget):
        with pytest.raises(ValueError):
            threshold_for_budget([0.1, 0.2], budget)

    def test_threshold_rejects_empty(self):
        with pytest.raises(ValueError):
            threshold_for_budget([], 0.1)

    def test_scores_probability_from_the_last_position_after_the_json_prefix(self):
        torch = pytest.importorskip("torch")
        if type(torch).__name__ == "MagicMock" or not hasattr(torch, "randn"):
            pytest.skip("real torch required")
        tokenizer = FakeChatTokenizer()
        seen = []

        class Model:
            device = "cpu"

            def __call__(self, input_ids):
                seen.append(input_ids[0].tolist())
                logits = torch.zeros(1, input_ids.shape[1], 256)
                logits[0, -1, ord("s")] = 2.0
                logits[0, -1, ord("o")] = 0.0
                return SimpleNamespace(logits=logits)

        (p,) = score_split_probabilities(Model(), tokenizer, ["Who?"])
        assert p == pytest.approx(torch.sigmoid(torch.tensor(2.0)).item())
        assert "".join(map(chr, seen[0])).endswith("<|im_start|>assistant\n" + DECISION_PREFIX)


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

    @pytest.fixture
    def prefilled(self, monkeypatch):
        calls = []

        def install(*responses):
            queue = list(responses)

            def fake(model, tokenizer, messages, strategy, max_new_tokens):
                calls.append({"messages": messages, "strategy": strategy})
                return queue.pop(0)
            monkeypatch.setattr(adaptive, "_generate_prefilled", fake)
            return calls
        return install

    def test_forced_strategy_prefills_without_an_instruction_message(self, prefilled):
        calls = prefilled(plan_json(split_plan()))
        plan = generate_adaptive_query(None, None, "q?", forced_strategy="split")
        assert plan["strategy"] == "split" and "p_split" not in plan
        assert calls[0]["strategy"] == "split"
        assert calls[0]["messages"] == planner_messages("q?")

    def test_forced_strategy_mismatch_is_rejected(self, prefilled):
        prefilled(plan_json(once_plan()), plan_json(once_plan()))
        with pytest.raises(ValueError, match="Expected split"):
            generate_adaptive_query(None, None, "q?", forced_strategy="split")

    @pytest.mark.parametrize("p_split, threshold, expected", [
        (0.40, 0.30, "split"), (0.30, 0.30, "split"), (0.29, 0.30, "once"),
    ])
    def test_threshold_decides_the_prefilled_strategy(self, prefilled, monkeypatch, p_split, threshold, expected):
        monkeypatch.setattr(adaptive, "score_split_probabilities", lambda m, t, qs: [p_split])
        plan_for = {"split": split_plan(), "once": once_plan()}[expected]
        calls = prefilled(plan_json(plan_for))
        plan = generate_adaptive_query(None, None, "q?", decision_threshold=threshold)
        assert calls[0]["strategy"] == expected and plan["strategy"] == expected
        assert plan["p_split"] == p_split

    def test_forced_strategy_overrides_the_threshold(self, prefilled, monkeypatch):
        monkeypatch.setattr(adaptive, "score_split_probabilities",
                            lambda *a: pytest.fail("must not score when forced"))
        calls = prefilled(plan_json(once_plan()))
        generate_adaptive_query(None, None, "q?", forced_strategy="once", decision_threshold=0.0)
        assert calls[0]["strategy"] == "once"

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
