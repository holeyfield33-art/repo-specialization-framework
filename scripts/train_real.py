#!/usr/bin/env python3
"""
Real repository-specific LoRA training.

This actually loads a base model, attaches LoRA adapters, runs optimizer steps
on the historical training split, and writes real adapter weights
(`adapter_model.safetensors`) plus a machine-readable training trace.

It replaces the previous placeholder path, which wrote ADAPTER_PLACEHOLDER.txt
and trained nothing. Only conventions/patterns are learned here; volatile facts
stay in the file packs and dependency graph.

Usage:
  python scripts/train_real.py \
      --train_jsonl results/data/splits/train.jsonl \
      --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
      --out results/adapters/qwen-repo-qlora \
      --steps 200

Runs on CPU (slowly, fp32) or GPU. Pass --load_in_4bit for QLoRA on CUDA.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prompting import SYSTEM_PROMPT, build_prompt, build_target


def load_tasks(path: Path) -> List[SimpleNamespace]:
    """Read a split JSONL written by history_tasks.write_splits."""
    tasks: List[SimpleNamespace] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            tasks.append(SimpleNamespace(**json.loads(line)))
    if not tasks:
        raise ValueError(f"no training examples found in {path}")
    return tasks


def build_examples(
    tasks: List[SimpleNamespace],
    packs_dir: Path | None,
    condition: str,
) -> List[Dict[str, str]]:
    """Training uses the condition-C context shape (packs, no graph): the
    adapter learns the repository's conventions, and the graph stays an
    inference-time input so condition D remains a separable variable."""
    examples = []
    for t in tasks:
        target = build_target(t)
        if target == '{"impacted": []}':
            # No supervisable path list — training on an empty answer teaches
            # the model to answer nothing.
            continue
        examples.append(
            {
                "task_id": t.task_id,
                "prompt": build_prompt(t, condition, packs_dir, graph=None),
                "target": target,
            }
        )
    if not examples:
        raise ValueError(
            "every training task resolved to an empty answer; nothing to train on"
        )
    return examples


def encode(tokenizer, prompt: str, target: str, max_seq_len: int):
    """Tokenize one example, masking the prompt so loss is computed on the
    completion only."""
    import torch

    if tokenizer.chat_template:
        prompt_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt_text = f"{SYSTEM_PROMPT}\n\n{prompt}\n\nAnswer: "

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target + tokenizer.eos_token, add_special_tokens=False)["input_ids"]

    # Truncate the prompt from the left so the answer always survives.
    budget = max_seq_len - len(target_ids)
    if budget < 1:
        target_ids = target_ids[: max_seq_len - 1]
        budget = 1
    prompt_ids = prompt_ids[-budget:]

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + list(target_ids)
    return (
        torch.tensor([input_ids], dtype=torch.long),
        torch.tensor([labels], dtype=torch.long),
    )


def per_layer_grad_norms(model) -> Dict[str, float]:
    """Real per-parameter gradient norms for every trainable (LoRA) tensor."""
    norms = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            norms[name] = float(param.grad.detach().norm().item())
    return norms


def main() -> None:
    parser = argparse.ArgumentParser(description="Real repo-specific LoRA training")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--packs_dir", default=None,
                        help="File-pack directory; training prompts match condition C.")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="QLoRA 4-bit base weights (requires CUDA + bitsandbytes).")
    parser.add_argument("--trace_every", type=int, default=1,
                        help="Record per-layer gradient norms every N steps.")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    packs_dir = Path(args.packs_dir) if args.packs_dir else None

    tasks = load_tasks(Path(args.train_jsonl))
    examples = build_examples(tasks, packs_dir, condition="C")
    print(f"[train_real] {len(tasks)} tasks -> {len(examples)} supervisable examples")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if args.load_in_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--load_in_4bit requires CUDA; run without it for CPU training"
            )
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"
    elif torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs["device_map"] = "auto"

    print(f"[train_real] loading base model {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    if args.load_in_4bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.train()

    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    losses: List[Dict[str, Any]] = []
    grad_traces: List[Dict[str, Any]] = []
    started = time.time()

    for step in range(args.steps):
        example = examples[step % len(examples)]
        input_ids, labels = encode(
            tokenizer, example["prompt"], example["target"], args.max_seq_len
        )
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()

        if step % args.trace_every == 0:
            norms = per_layer_grad_norms(model)
            grad_traces.append(
                {"step": step, "task_id": example["task_id"], "grad_norms": norms}
            )

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 0.3
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        loss_value = float(loss.detach().item())
        losses.append({"step": step, "loss": loss_value, "task_id": example["task_id"]})
        if step % max(1, args.steps // 20) == 0 or step == args.steps - 1:
            print(f"[train_real] step {step}/{args.steps} loss={loss_value:.4f}")

    elapsed = time.time() - started
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    weights = out_dir / "adapter_model.safetensors"
    if not weights.exists():
        raise RuntimeError(
            f"training finished but no adapter weights at {weights}; refusing to "
            "report a trained adapter that does not exist"
        )

    trace = {
        "status": "TRAINED",
        "base_model": args.model,
        "train_jsonl": str(args.train_jsonl),
        "example_count": len(examples),
        "steps": args.steps,
        "learning_rate": args.lr,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout},
        "load_in_4bit": args.load_in_4bit,
        "device": str(device),
        "elapsed_seconds": round(elapsed, 2),
        "first_loss": losses[0]["loss"] if losses else None,
        "last_loss": losses[-1]["loss"] if losses else None,
        "loss_curve": losses,
        "grad_trace_count": sum(len(t["grad_norms"]) for t in grad_traces),
        "gradient_traces": grad_traces,
    }
    (out_dir / "training_trace.json").write_text(
        json.dumps(trace, indent=2), encoding="utf-8"
    )

    # A stale NOT_TRAINED marker next to real weights would be a lie.
    stale = out_dir / "NOT_TRAINED.txt"
    if stale.exists():
        stale.unlink()

    print(
        f"[train_real] adapter saved to {out_dir} "
        f"(loss {trace['first_loss']:.4f} -> {trace['last_loss']:.4f}, "
        f"{trace['grad_trace_count']} gradient traces)"
    )


if __name__ == "__main__":
    main()
