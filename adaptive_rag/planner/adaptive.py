"""Supervised LoRA/QLoRA training and inference for the adaptive RAG planner.

Inputs are JSONL records exported by data.training_data.export_sft_examples:
{question_id, question, strategy, target}, where target is planner JSON.
Training masks prompt tokens and supervises only assistant completion tokens.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

from adaptive_rag.config import SLM_MODEL, SLM_MAX_NEW_TOKENS, SEED
from adaptive_rag.evaluation.checkpoint import fingerprint
from adaptive_rag.evaluation.experiment import _write_json_atomic
from adaptive_rag.planner.prompted import validate_query_plan

SYSTEM_PROMPT = (
    "You are a retrieval planner for a RAG system. Given a question, "
    "choose 'once' or 'split' and write lexical and semantic retrieval queries. "
    "Use exactly one query for 'once' or exactly two for 'split'. "
    "Return only a valid JSON object with keys 'strategy' and 'queries'. "
    "Each query must have nonempty 'lexical_query' and 'semantic_query' fields."
)


DECISION_PREFIX = '{"strategy":"'  # The next token starts the "once"/"split" value.
DEFAULT_DECISION_WEIGHT = 0.25
DEFAULT_LABEL_SMOOTHING = 0.05


def planner_messages(question: str) -> list[dict[str, str]]:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a nonempty string")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}\nReturn only the retrieval plan JSON."},
    ]


def load_sft_examples(path: str | Path) -> list[dict[str, Any]]:
    """Read and validate the SFT export; reject duplicate question IDs."""
    records = []
    seen = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                question = record["question"]
                question_id = str(record["question_id"])
                target = json.loads(record["target"])
                validate_query_plan(target)
                if target["strategy"] != record["strategy"]:
                    raise ValueError("strategy does not match target")
                if not isinstance(question, str) or not question.strip() or not question_id.strip():
                    raise ValueError("empty question or question_id")
                if question_id in seen:
                    raise ValueError("duplicate question_id")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid SFT record at line {line_number}: {exc}") from exc
            seen.add(question_id)
            # 'question' is deliberately excluded from the assistant's target.
            target = {"strategy": target["strategy"], "queries": target["queries"]}
            records.append({"question_id": question_id, "question": question, "target": target})
    if not records:
        raise ValueError("No SFT examples found")
    return records


def split_sft_examples(records: list[dict[str, Any]], *, validation_fraction: float = 0.2,
                       seed: int = SEED) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministic question-level train/validation split of labeled training data."""
    if not 0 < validation_fraction < 1 or len(records) < 2:
        raise ValueError("Need at least two records and validation_fraction in (0, 1)")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, min(len(shuffled) - 1, round(len(shuffled) * validation_fraction)))
    return shuffled[n_val:], shuffled[:n_val]


def tokenize_sft_examples(records: list[dict[str, Any]], tokenizer: Any, *, max_length: int = 768):
    """Create completion-only labels; reject examples that would be truncated."""
    if max_length < 32:
        raise ValueError("max_length is too small")
    encoded = []
    for record in records:
        messages = planner_messages(record["question"])
        completion = json.dumps(record["target"], ensure_ascii=False, separators=(",", ":"))

        prefix = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )

        full = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": completion}],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
        )

        if full[:len(prefix)] != prefix:
            raise ValueError("Chat template prefix mismatch; inspect tokenizer template")
        completion_ids = full[len(prefix):]
        if not completion_ids:
            raise ValueError("Empty assistant completion")
        ids = full
        if len(ids) > max_length:
            raise ValueError(
                f"Example {record['question_id']} has {len(ids)} tokens; "
                f"increase max_length ({max_length}) to avoid truncating JSON."
            )
        strategy = record["target"]["strategy"]
        if not completion.startswith(DECISION_PREFIX + strategy):
            raise ValueError("Target JSON must start with the strategy field")
        decision_pos = len(prefix) + _decision_index(tokenizer, completion_ids, strategy)
        encoded.append({
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "labels": [-100] * len(prefix) + completion_ids,
            "decision_pos": decision_pos,
            "decision_label": int(strategy == "split"),
        })
    return encoded


def _decision_index(tokenizer: Any, completion_ids: list[int], strategy: str) -> int:
    """Index (within the completion) of the first token of the strategy value."""
    consumed = 0
    for index, token_id in enumerate(completion_ids):
        if consumed == len(DECISION_PREFIX):
            piece = tokenizer.decode([token_id], skip_special_tokens=False).strip()
            if piece and strategy.startswith(piece):
                return index
            break
        if consumed > len(DECISION_PREFIX):
            break
        consumed += len(tokenizer.decode([token_id], skip_special_tokens=False))
    raise ValueError(
        "Cannot isolate the strategy token: the tokenizer merges it with the JSON prefix"
    )


_DUMMY_PLANS = {
    "once": {"strategy": "once", "queries": [{"lexical_query": "a", "semantic_query": "b"}]},
    "split": {"strategy": "split", "queries": [
        {"lexical_query": "a", "semantic_query": "b"},
        {"lexical_query": "c", "semantic_query": "d"},
    ]},
}


def _dummy_rows(tokenizer: Any) -> dict[str, dict[str, Any]]:
    records = [{"question_id": name, "question": "question", "target": plan}
               for name, plan in _DUMMY_PLANS.items()]
    rows = tokenize_sft_examples(records, tokenizer, max_length=100_000)
    return dict(zip(_DUMMY_PLANS, rows))


def decision_token_ids(tokenizer: Any) -> dict[str, int]:
    """Token ids that start the "once" / "split" value after the JSON prefix."""
    ids = {name: row["input_ids"][row["decision_pos"]] for name, row in _dummy_rows(tokenizer).items()}
    if ids["once"] == ids["split"]:
        raise ValueError("'once' and 'split' start with the same token; cannot train a 2-way decision")
    return ids


def decision_prefix_ids(tokenizer: Any) -> list[int]:
    """Assistant tokens that precede the decision token (identical for every question)."""
    prefixes = []
    for row in _dummy_rows(tokenizer).values():
        start = next(i for i, label in enumerate(row["labels"]) if label != -100)
        prefixes.append(row["input_ids"][start:row["decision_pos"]])
    if prefixes[0] != prefixes[1]:
        raise ValueError("Assistant prefix tokens differ between strategies")
    return prefixes[0]


class CompletionCollator:
    """Pad input/attention/label sequences with labels=-100."""

    def __init__(self, tokenizer: Any):
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            raise ValueError("Tokenizer must have a pad token")

    def __call__(self, examples):
        import torch
        longest = max(len(row["input_ids"]) for row in examples)
        batch = {
            "input_ids": torch.tensor([row["input_ids"] + [self.pad_id] * (longest - len(row["input_ids"])) for row in examples], dtype=torch.long),
            "attention_mask": torch.tensor([row["attention_mask"] + [0] * (longest - len(row["attention_mask"])) for row in examples], dtype=torch.long),
            "labels": torch.tensor([row["labels"] + [-100] * (longest - len(row["labels"])) for row in examples], dtype=torch.long),
        }
        if all("decision_pos" in row for row in examples):
            batch["decision_pos"] = torch.tensor([row["decision_pos"] for row in examples], dtype=torch.long)
            batch["decision_label"] = torch.tensor([row["decision_label"] for row in examples], dtype=torch.float)
        return batch


def roc_auc(scores: list[float], labels: list[int]) -> float | None:
    """Rank-based ROC AUC (ties count half); None when only one class is present."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def decision_prior_baseline(train_records: list[dict[str, Any]],
                            val_records: list[dict[str, Any]]) -> dict[str, float]:
    """Unsmoothed decision loss of always predicting the training split rate."""
    rate = sum(r["target"]["strategy"] == "split" for r in train_records) / len(train_records)
    rate = min(max(rate, 1e-6), 1 - 1e-6)
    losses = [-math.log(rate if r["target"]["strategy"] == "split" else 1 - rate)
              for r in val_records]
    return {"train_split_rate": rate, "prior_decision_loss": sum(losses) / len(losses)}


def planner_loss_terms(logits, labels, decision_pos, decision_label, token_ids, *,
                       label_smoothing: float = 0.0):
    """Return (decision_loss, plan_loss, p_split).

    decision_loss: 2-way BCE between the "split" and "once" first-token logits at the
    strategy position. plan_loss: mean cross-entropy over the other completion tokens.
    """
    import torch
    import torch.nn.functional as F
    batch = torch.arange(logits.shape[0], device=logits.device)
    decision_pos = decision_pos.to(logits.device)
    at = decision_pos - 1  # logits at t predict the token at t + 1
    step_logits = logits[batch, at].float()
    margin = step_logits[:, token_ids["split"]] - step_logits[:, token_ids["once"]]
    target = decision_label.to(logits.device).float()
    target = target * (1 - label_smoothing) + 0.5 * label_smoothing
    decision_loss = F.binary_cross_entropy_with_logits(margin, target)

    shift_logits = logits[:, :-1]
    shift_labels = labels[:, 1:].to(logits.device).clone()
    shift_labels[batch, at] = -100  # the decision token belongs to the decision term only
    mask = shift_labels != -100
    plan_loss = F.cross_entropy(shift_logits[mask].float(), shift_labels[mask])
    return decision_loss, plan_loss, torch.sigmoid(margin.detach())


def build_planner_trainer(*, model, args, train_dataset, eval_dataset, data_collator,
                          token_ids: dict[str, int], decision_weight: float,
                          label_smoothing: float):
    """Trainer using L = w * decision BCE + (1 - w) * plan CE, with component logging."""
    from transformers import Trainer

    class PlannerTrainer(Trainer):
        def _bucket(self, name):
            if not hasattr(self, "_loss_buckets"):
                self._loss_buckets = {}
            return self._loss_buckets.setdefault(
                name, {"decision": 0.0, "plan": 0.0, "n": 0, "p": [], "y": []})

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            inputs = dict(inputs)
            decision_pos = inputs.pop("decision_pos")
            decision_label = inputs.pop("decision_label")
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            training = model.training
            decision, plan, p_split = planner_loss_terms(
                outputs.logits, labels, decision_pos, decision_label, token_ids,
                label_smoothing=label_smoothing if training else 0.0)
            loss = decision_weight * decision + (1 - decision_weight) * plan
            bucket = self._bucket("train" if training else "eval")
            bucket["decision"] += float(decision.detach())
            bucket["plan"] += float(plan.detach())
            bucket["n"] += 1
            if not training:
                bucket["p"].extend(p_split.cpu().tolist())
                bucket["y"].extend(int(v) for v in decision_label.cpu().tolist())
            return (loss, outputs) if return_outputs else loss

        def log(self, logs, *args, **kwargs):
            logs = dict(logs)
            name = "eval" if "eval_loss" in logs else "train" if "loss" in logs else None
            if name:
                bucket = self._bucket(name)
                if bucket["n"]:
                    prefix = "eval_" if name == "eval" else ""
                    logs[prefix + "decision_loss"] = bucket["decision"] / bucket["n"]
                    logs[prefix + "plan_loss"] = bucket["plan"] / bucket["n"]
                    if name == "eval":
                        auc = roc_auc(bucket["p"], bucket["y"])
                        if auc is not None:
                            logs["eval_decision_auc"] = auc
                self._loss_buckets[name] = {"decision": 0.0, "plan": 0.0, "n": 0, "p": [], "y": []}
            return super().log(logs, *args, **kwargs)

    trainer = PlannerTrainer(model=model, args=args, train_dataset=train_dataset,
                             eval_dataset=eval_dataset, data_collator=data_collator)
    # The loss is a per-batch mean; make Trainer divide by gradient accumulation steps.
    trainer.model_accepts_loss_kwargs = False
    return trainer


def load_base_model(model_name: str = SLM_MODEL, *, use_4bit: bool = True):
    """Load tokenizer and model for training. 4-bit QLoRA requires CUDA/bitsandbytes."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {"device_map": "auto", "dtype": "auto"}
    if use_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit QLoRA requires a CUDA GPU; use use_4bit=False")
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.config.use_cache = False
    return model, tokenizer


def configure_lora(model: Any, *, rank: int = 16, alpha: int = 32, dropout: float = 0.05,
                   use_4bit: bool = True):
    """Attach trainable LoRA adapters to Qwen attention projections."""
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, config)


def train_planner(
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    output_dir: str | Path,
    *, model_name: str = SLM_MODEL, use_4bit: bool = True,
    max_length: int = 768, rank: int = 16, alpha: int = 32,
    dropout: float = 0.05, epochs: float = 3.0,
    learning_rate: float = 2e-4, batch_size: int = 2,
    gradient_accumulation_steps: int = 8, seed: int = SEED,
    resume_from_checkpoint: str | Path | bool = False,
    decision_weight: float = DEFAULT_DECISION_WEIGHT,
    label_smoothing: float = DEFAULT_LABEL_SMOOTHING,
):
    """Fine-tune on selected plans; save adapter/tokenizer to output_dir.

    Returns (trainer, model, tokenizer). The caller controls where artifacts persist.
    """
    if not train_records or not val_records:
        raise ValueError("Nonempty train and validation records are required")
    if {r["question_id"] for r in train_records} & {r["question_id"] for r in val_records}:
        raise ValueError("Training and validation question IDs overlap")
    if not 0 < decision_weight < 1:
        raise ValueError("decision_weight must be between 0 and 1")
    if not 0 <= label_smoothing < 0.5:
        raise ValueError("label_smoothing must be in [0, 0.5)")
    import torch
    from transformers import TrainingArguments, set_seed
    from transformers.trainer_utils import get_last_checkpoint
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    config_path = output_dir / "training_config.json"
    config = {
        "model_name": model_name, "use_4bit": use_4bit, "max_length": max_length,
        "rank": rank, "alpha": alpha, "dropout": dropout, "epochs": epochs,
        "learning_rate": learning_rate, "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps, "seed": seed,
        "decision_weight": decision_weight, "label_smoothing": label_smoothing,
        "train_examples": len(train_records), "val_examples": len(val_records),
        "train_sha256": fingerprint(train_records), "val_sha256": fingerprint(val_records),
    }
    latest = get_last_checkpoint(str(checkpoints_dir)) if checkpoints_dir.is_dir() else None
    checkpoint = None
    if resume_from_checkpoint:
        if resume_from_checkpoint is True and latest is None:
            raise ValueError(f"No training checkpoints found in {checkpoints_dir}")
        checkpoint = Path(latest if resume_from_checkpoint is True else resume_from_checkpoint).resolve()
        if checkpoint.parent != checkpoints_dir.resolve() or not (checkpoint / "trainer_state.json").is_file():
            raise ValueError(f"Expected a Trainer checkpoint in {checkpoints_dir}: {checkpoint}")
        if not config_path.is_file() or json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Training data or configuration cannot be verified. Use a new output directory.")
    elif latest is not None:
        raise ValueError(f"Checkpoints already exist in {checkpoints_dir}. Resume or use a new output directory.")

    set_seed(seed)
    model, tokenizer = load_base_model(model_name, use_4bit=use_4bit)
    model = configure_lora(model, rank=rank, alpha=alpha, dropout=dropout, use_4bit=use_4bit)
    model.config.use_cache = False
    train_data = tokenize_sft_examples(train_records, tokenizer, max_length=max_length)
    val_data = tokenize_sft_examples(val_records, tokenizer, max_length=max_length)
    args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        seed=seed,
    )
    trainer = build_planner_trainer(
        model=model, args=args, train_dataset=train_data, eval_dataset=val_data,
        data_collator=CompletionCollator(tokenizer),
        token_ids=decision_token_ids(tokenizer),
        decision_weight=decision_weight, label_smoothing=label_smoothing,
    )
    _write_json_atomic(config_path, config)
    trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    model.save_pretrained(output_dir / "adapter")
    tokenizer.save_pretrained(output_dir / "adapter")
    return trainer, model, tokenizer


def load_adaptive_planner(adapter_dir: str | Path, *, base_model_name: str | None = None,
                          use_4bit: bool = False):
    """Load a saved PEFT adapter for inference, returning (model, tokenizer)."""
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    adapter_dir = str(adapter_dir)
    peft_config = PeftConfig.from_pretrained(adapter_dir)
    base_name = base_model_name or peft_config.base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {"device_map": "auto", "dtype": "auto"}
    if use_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit adapter loading requires CUDA")
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
    base = AutoModelForCausalLM.from_pretrained(base_name, **kwargs)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    model.config.use_cache = True
    return model, tokenizer


def score_split_probabilities(model: Any, tokenizer: Any, questions: list[str]) -> list[float]:
    """P(split) from the first strategy-token logits, one forward pass per question."""
    import torch
    token_ids = decision_token_ids(tokenizer)
    assistant_prefix = decision_prefix_ids(tokenizer)
    probabilities = []
    for question in questions:
        ids = tokenizer.apply_chat_template(
            planner_messages(question), tokenize=True, add_generation_prompt=True,
            return_dict=False) + assistant_prefix
        input_ids = torch.tensor([ids], dtype=torch.long, device=model.device)
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits[0, -1].float()
        margin = logits[token_ids["split"]] - logits[token_ids["once"]]
        probabilities.append(float(torch.sigmoid(margin)))
    return probabilities


def threshold_for_budget(probabilities: list[float], budget: float = 0.10) -> float:
    """Label-free threshold: the smallest tau routing at least `budget` of questions to SPLIT."""
    if not probabilities:
        raise ValueError("probabilities must be nonempty")
    if not 0 < budget <= 1:
        raise ValueError("budget must be in (0, 1]")
    ranked = sorted(probabilities, reverse=True)
    return ranked[max(math.ceil(budget * len(ranked)), 1) - 1]


def _generate_prefilled(model: Any, tokenizer: Any, messages: list[dict[str, str]],
                        strategy: str, max_new_tokens: int) -> str:
    """Prefill the JSON prefix plus the first strategy token, then let the model continue."""
    import torch
    prefill = decision_prefix_ids(tokenizer) + [decision_token_ids(tokenizer)[strategy]]
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=False) + prefill
    input_ids = torch.tensor([ids], dtype=torch.long, device=model.device)
    output = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                            max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(prefill + output[0][input_ids.shape[1]:].tolist(),
                            skip_special_tokens=True)


def generate_adaptive_query(model: Any, tokenizer: Any, question: str, *,
                            max_new_tokens: int = SLM_MAX_NEW_TOKENS,
                            forced_strategy: str | None = None,
                            decision_threshold: float | None = None,
                            max_retries: int = 1) -> dict[str, Any]:
    """Generate a valid plan; optionally force ONCE/SPLIT for controlled ablations.

    Retrying invalid output does not change the original question. Raises ValueError
    after all attempts; the evaluation runner records planner failures.
    """
    from adaptive_rag.models.lm import generate
    if forced_strategy not in (None, "once", "split"):
        raise ValueError("forced_strategy must be None, 'once', or 'split'")
    if max_retries < 0:
        raise ValueError("max_retries must be nonnegative")
    messages = planner_messages(question)
    # Forced or thresholded strategies are prefilled in the training format, so the
    # model writes the queries for a decision it was actually trained to follow.
    strategy, p_split = forced_strategy, None
    if strategy is None and decision_threshold is not None:
        p_split = score_split_probabilities(model, tokenizer, [question])[0]
        strategy = "split" if p_split >= decision_threshold else "once"
    for attempt in range(max_retries + 1):
        if strategy is not None:
            response = _generate_prefilled(model, tokenizer, messages, strategy, max_new_tokens)
        else:
            response = generate(model, tokenizer, messages, max_new_tokens=max_new_tokens)
        cleaned = response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if len(lines) >= 2 and lines[-1].strip() == "```":
                cleaned = "\n".join(lines[1:-1]).strip()
        try:
            plan = json.loads(cleaned)
            validate_query_plan(plan)
            if strategy is not None and plan["strategy"] != strategy:
                raise ValueError(f"Expected {strategy}, got {plan['strategy']}")
            result = {**plan, "question": question, "planner_attempts": attempt + 1}
            if p_split is not None:
                result["p_split"] = p_split
            return result
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            if attempt == max_retries:
                raise ValueError(
                    f"Adaptive planner invalid after {attempt + 1} attempts: "
                    f"{exc}; last output={response!r}"
                ) from exc
            messages = [*messages, {
                "role": "user",
                "content": (
                    f"Your previous response was invalid: {exc}. "
                    "Start over. Return only valid JSON with keys strategy and queries; "
                    "each query needs lexical_query and semantic_query. "
                    "ONCE requires exactly one query and SPLIT exactly two. "
                    + (f"You MUST use strategy '{strategy}'. " if strategy else "")
                ),
            }]
    raise RuntimeError("Unexpected planner generation failure")
