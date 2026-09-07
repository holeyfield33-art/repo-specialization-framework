"""
6. QLoRA configuration
Configurable for Qwen2.5-Coder-1.5B-Instruct and SmolLM3-3B (or any 0.5B–3B HF model).
Adapter saved separately; current facts stay in packs/graph, not weights.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import yaml


@dataclass
class QLoRAHyperParams:
    model_name_or_path: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    max_seq_length: int = 2048
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    num_train_epochs: float = 2.0
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 50
    eval_steps: int = 50
    save_total_limit: int = 2
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_8bit"
    max_grad_norm: float = 0.3
    seed: int = 42

    output_dir: str = "results/adapters/repo-qlora"
    report_to: str = "none"

    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    @classmethod
    def for_model(cls, model_key: str) -> "QLoRAHyperParams":
        key = model_key.lower()
        if "smol" in key:
            return cls(
                model_name_or_path="HuggingFaceTB/SmolLM3-3B",
                lora_r=8,
                lora_alpha=16,
                max_seq_length=2048,
                learning_rate=1e-4,
            )
        return cls(
            model_name_or_path="Qwen/Qwen2.5-Coder-1.5B-Instruct",
            lora_r=16,
            lora_alpha=32,
        )


TRAIN_SCRIPT = r'''
#!/usr/bin/env python3
"""Repo-specific QLoRA SFT. Adapter only; facts remain in packs/graph."""
import argparse, json, os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.dry_run:
        print("DRY RUN — would train with:")
        print(json.dumps(cfg, indent=2))
        print(f"train examples: {sum(1 for _ in open(args.train_jsonl))}")
        return

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    bnb = BitsAndBytesConfig(
        load_in_4bit=cfg["load_in_4bit"],
        bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(__import__("torch"), cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=cfg["bnb_4bit_use_double_quant"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name_or_path"], quantization_config=bnb, device_map="auto", trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"], bias=cfg["bias"], task_type=cfg["task_type"],
    )
    model = get_peft_model(model, lora)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def format_example(ex):
        return {
            "text": f"<|im_start|>user\n{ex['instruction']}\nContext files: {ex['context_files']}<|im_end|>\n"
                    f"<|im_start|>assistant\n{json.dumps(ex['ground_truth'])}<|im_end|>"
        }

    ds = load_dataset("json", data_files={"train": args.train_jsonl}, split="train")
    ds = ds.map(format_example)

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_train_epochs"],
        warmup_ratio=cfg["warmup_ratio"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        optim=cfg["optim"],
        max_grad_norm=cfg["max_grad_norm"],
        report_to=cfg["report_to"],
        seed=cfg["seed"],
        bf16=True,
    )
    from transformers import Trainer
    trainer = Trainer(model=model, args=training_args, train_dataset=ds, tokenizer=tokenizer)
    trainer.train()
    model.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"Adapter saved to {cfg['output_dir']}")

if __name__ == "__main__":
    main()
'''
