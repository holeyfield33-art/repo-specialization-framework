"""Frozen QLoRA hyperparameters for the two RSEF model runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

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
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
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

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(asdict(self), sort_keys=False), encoding="utf-8")

    @classmethod
    def for_model(cls, model_key: str) -> "QLoRAHyperParams":
        if "smol" in model_key.lower():
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
