#!/usr/bin/env python3
"""Train the real repository QLoRA adapter on the frozen training split only."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    args = parser.parse_args()

    import torch
    import yaml
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required; refusing CPU or placeholder training")
    cfg = yaml.safe_load(Path(args.config).read_text())
    data_files = {"train": args.train_jsonl, "validation": args.val_jsonl}
    ds = load_dataset("json", data_files=data_files)
    if not len(ds["train"]):
        raise SystemExit("training split is empty")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def expected_response(ex):
        gt = ex["ground_truth"]
        response = {
            "predicted_files": [], "predicted_tests": [], "primary_files": [],
            "related_files": [], "suggested_checks": [], "referenced_apis": [],
            "patch": None, "analysis": "",
        }
        task_type = ex["task_type"]
        if task_type in ("change_impact_prediction", "code_review", "patch_generation"):
            response["predicted_files"] = gt.get("impacted", gt.get("files", []))
        if task_type == "test_impact_prediction":
            response["predicted_tests"] = gt.get("tests_to_run", [])
        if task_type == "bug_localization":
            response["primary_files"] = gt.get("primary_files", [])
        if task_type == "cross_file_dependency_reasoning":
            response["related_files"] = gt.get("related_files", [])
        if task_type == "code_review":
            response["suggested_checks"] = gt.get("suggested_checks", [])
        if task_type == "patch_generation":
            response["patch"] = gt.get("patch")
        return response

    def render(ex):
        messages = [
            {"role": "system", "content": "Use repository evidence and return one JSON object."},
            {"role": "user", "content": ex["instruction"] + "\nContext files: " + json.dumps(ex["context_files"])},
            {"role": "assistant", "content": json.dumps(expected_response(ex), ensure_ascii=False)},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        else:
            text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        return {"text": text}

    ds = ds.map(render)

    def tokenize(batch):
        return tokenizer(
            batch["text"], truncation=True, max_length=cfg["max_seq_length"],
            padding=False,
        )

    tokenized = ds.map(tokenize, batched=True, remove_columns=ds["train"].column_names)
    compute_dtype = getattr(torch, cfg["bnb_4bit_compute_dtype"])
    quant = BitsAndBytesConfig(
        load_in_4bit=cfg["load_in_4bit"],
        bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg["bnb_4bit_use_double_quant"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name_or_path"], quantization_config=quant, device_map="auto", trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"], bias=cfg["bias"], task_type=cfg["task_type"],
    ))

    # Honor the saved run configuration, even on hardware supporting both types.
    dtype_name = cfg["bnb_4bit_compute_dtype"]
    if dtype_name not in ("float16", "bfloat16"):
        raise SystemExit(f"unsupported compute dtype: {dtype_name}")
    bf16 = dtype_name == "bfloat16"
    if bf16 and not torch.cuda.is_bf16_supported():
        raise SystemExit("configured BF16 is unsupported by this GPU")
    training_kwargs = dict(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_train_epochs"],
        warmup_ratio=cfg["warmup_ratio"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        eval_steps=cfg["eval_steps"],
        save_strategy="steps",
        save_total_limit=cfg["save_total_limit"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        optim=cfg["optim"],
        max_grad_norm=cfg["max_grad_norm"],
        report_to=cfg["report_to"],
        seed=cfg["seed"],
        bf16=bf16,
        fp16=not bf16,
    )
    strategy_key = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters
        else "evaluation_strategy"
    )
    training_kwargs[strategy_key] = "steps" if len(tokenized["validation"]) else "no"
    training_args = TrainingArguments(**training_kwargs)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"] if len(tokenized["validation"]) else None,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    result = trainer.train()
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    (output / "training_metrics.json").write_text(json.dumps(result.metrics, indent=2))
    if not (output / "adapter_config.json").is_file():
        raise SystemExit("training completed without adapter_config.json")


if __name__ == "__main__":
    main()
