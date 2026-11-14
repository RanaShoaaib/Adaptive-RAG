"""Supervised LoRA/QLoRA training and inference for the adaptive RAG planner.

Inputs are JSONL records exported by data.training_data.export_sft_examples:
{question_id, question, strategy, target}, where target is planner JSON.
Training masks prompt tokens and supervises only assistant completion tokens.
"""
from __future__ import annotations

import json
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
        encoded.append({
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "labels": [-100] * len(prefix) + completion_ids,
        })
    return encoded


class CompletionCollator:
    """Pad input/attention/label sequences with labels=-100."""

    def __init__(self, tokenizer: Any):
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            raise ValueError("Tokenizer must have a pad token")

    def __call__(self, examples):
        import torch
        longest = max(len(row["input_ids"]) for row in examples)
        return {
            "input_ids": torch.tensor([row["input_ids"] + [self.pad_id] * (longest - len(row["input_ids"])) for row in examples], dtype=torch.long),
            "attention_mask": torch.tensor([row["attention_mask"] + [0] * (longest - len(row["attention_mask"])) for row in examples], dtype=torch.long),
            "labels": torch.tensor([row["labels"] + [-100] * (longest - len(row["labels"])) for row in examples], dtype=torch.long),
        }


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
):
    """Fine-tune on selected plans; save adapter/tokenizer to output_dir.

    Returns (trainer, model, tokenizer). The caller controls where artifacts persist.
    """
    if not train_records or not val_records:
        raise ValueError("Nonempty train and validation records are required")
    if {r["question_id"] for r in train_records} & {r["question_id"] for r in val_records}:
        raise ValueError("Training and validation question IDs overlap")
    import torch
    from transformers import Trainer, TrainingArguments, set_seed
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
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=CompletionCollator(tokenizer),
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


def generate_adaptive_query(model: Any, tokenizer: Any, question: str, *,
                            max_new_tokens: int = SLM_MAX_NEW_TOKENS,
                            forced_strategy: str | None = None,
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
    if forced_strategy is not None:
        messages = [*messages, {
            "role": "user",
            "content": (
                f"For this evaluation, use strategy '{forced_strategy}' exactly. "
                f"Return exactly {1 if forced_strategy == 'once' else 2} retrieval "
                "query object(s) in the same JSON schema, with no explanation."
            ),
        }]
    for attempt in range(max_retries + 1):
        response = generate(model, tokenizer, messages, max_new_tokens=max_new_tokens)
        cleaned = response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if len(lines) >= 2 and lines[-1].strip() == "```":
                cleaned = "\n".join(lines[1:-1]).strip()
        try:
            plan = json.loads(cleaned)
            validate_query_plan(plan)
            if forced_strategy is not None and plan["strategy"] != forced_strategy:
                raise ValueError(f"Expected {forced_strategy}, got {plan['strategy']}")
            return {**plan, "question": question, "planner_attempts": attempt + 1}
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
                    + (f"You MUST use strategy '{forced_strategy}'. " if forced_strategy else "")
                ),
            }]
    raise RuntimeError("Unexpected planner generation failure")
