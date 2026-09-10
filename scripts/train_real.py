#!/usr/bin/env python3
"""
Real repository-specific LoRA training — replaces the ADAPTER_PLACEHOLDER.txt
step that run_experiment.py used to perform.

Auto-selects the backend:
  * CUDA + bitsandbytes  -> real 4-bit QLoRA (Colab T4 / A100)
  * CUDA, no bitsandbytes-> real bf16 LoRA, no quantization
  * CPU only             -> real fp32 LoRA (same adapter math, slower, more RAM)

Every path is real gradient descent on real data. No simulated scores, no
placeholder files. If --train_jsonl is missing it fails loudly rather than
substituting synthetic data: real training on fake data is the same
"simulated but labeled as evidence" problem this script exists to remove.

TrainingTelemetryHook logs, per step, which layers are actually being pushed
to change — block-level dLoss/d(output) and dLoss/dW norms for every adapted
projection. That is the evidence needed to answer whether tuning (conditions
C/D) does anything condition B does not.

Usage:
    python scripts/train_real.py \
        --train_jsonl results/data/splits/train.jsonl \
        --packs_dir results/file_packs \
        --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
        --out results/adapters/qwen-repo-qlora \
        --telemetry results/adapters/qwen-repo-qlora/train_telemetry.jsonl \
        --steps 200
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prompting import SYSTEM_PROMPT, build_prompt, build_target

#: Every projection LoRA adapts. Telemetry covers attention as well as MLP:
#: "did tuning change anything" is not answerable from the MLP alone.
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


# --------------------------------------------------------------------------
# Training telemetry
# --------------------------------------------------------------------------

@dataclass
class StepTelemetry:
    step: int
    loss: float
    compute_ms: float
    task_ids: List[str] = field(default_factory=list)
    grad_norms: Dict[str, float] = field(default_factory=dict)
    weight_grad_norms: Dict[str, float] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "loss": round(self.loss, 6),
            "compute_ms": self.compute_ms,
            "task_ids": self.task_ids,
            "grad_norms": {k: round(v, 5) for k, v in self.grad_norms.items()},
            "weight_grad_norms": {k: round(v, 5) for k, v in self.weight_grad_norms.items()},
        }


class TrainingTelemetryHook:
    """Backward hooks on each LoRA-wrapped projection: real
    dLoss/d(block_output) and dLoss/dW norms, one record per training step.

    Keys are `<layer_index>.<projection>` parsed from the module path rather
    than a running counter, so a norm can be traced back to the actual
    transformer layer it came from. Family-agnostic via module-name search,
    since PEFT wraps the base model and attribute paths shift by architecture.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._block_grad_norms: Dict[str, float] = {}
        self._weight_grad_norms: Dict[str, float] = {}
        self._handles: List[Any] = []
        self.tracked: List[str] = []

    @staticmethod
    def _key(name: str) -> str:
        layer = _LAYER_RE.search(name)
        proj = name.rsplit(".", 1)[-1]
        return f"{layer.group(1) if layer else '?'}.{proj}"

    def attach(self) -> None:
        for name, module in self.model.named_modules():
            if not name.endswith(tuple("." + t for t in TARGET_MODULES)):
                continue
            if not hasattr(module, "lora_B"):
                continue
            key = self._key(name)
            self.tracked.append(key)
            self._handles.append(
                module.register_full_backward_hook(self._make_block_hook(key))
            )
            for _adapter_name, lora_b in module.lora_B.items():
                if lora_b.weight.requires_grad:
                    self._handles.append(
                        lora_b.weight.register_hook(self._make_weight_hook(key))
                    )
        if not self.tracked:
            raise RuntimeError(
                "telemetry attached to zero LoRA projections — the adapter is not "
                "wired to the modules it claims to train. Refusing to report "
                "gradient evidence that does not exist."
            )

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def reset(self) -> None:
        self._block_grad_norms.clear()
        self._weight_grad_norms.clear()

    def _make_block_hook(self, key: str):
        def hook(_module, _grad_input, grad_output):
            g = grad_output[0]
            if g is not None:
                self._block_grad_norms[key] = float(g.detach().float().norm().item())
        return hook

    def _make_weight_hook(self, key: str):
        def hook(grad: torch.Tensor):
            self._weight_grad_norms[key] = float(grad.detach().float().norm().item())
        return hook

    def collect(
        self, step: int, loss: float, compute_ms: float, task_ids: List[str]
    ) -> StepTelemetry:
        return StepTelemetry(
            step=step,
            loss=loss,
            compute_ms=compute_ms,
            task_ids=task_ids,
            grad_norms=dict(self._block_grad_norms),
            weight_grad_norms=dict(self._weight_grad_norms),
        )


def summarize_telemetry(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Which layers actually moved, aggregated across the run.

    A near-zero weight-gradient norm everywhere means tuning changed nothing,
    and conditions C/D should not be expected to differ from A/B. That is a
    real finding, so it gets computed rather than left for someone to eyeball.
    """
    if not records:
        return {"steps": 0, "layers_moved": 0, "tracked_layers": 0}
    totals: Dict[str, float] = {}
    for rec in records:
        for key, value in rec["weight_grad_norms"].items():
            totals[key] = totals.get(key, 0.0) + value
    means = {k: v / len(records) for k, v in totals.items()}
    moved = {k: v for k, v in means.items() if v > 1e-8}
    ranked = sorted(means.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "steps": len(records),
        "tracked_layers": len(means),
        "layers_moved": len(moved),
        "all_layers_moved": len(moved) == len(means) and bool(means),
        "mean_weight_grad_norm": sum(means.values()) / len(means) if means else 0.0,
        "top_layers_by_mean_weight_grad": ranked[:10],
        "quiet_layers": [k for k, v in ranked if v <= 1e-8],
    }


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_train_examples(path: Path) -> List[SimpleNamespace]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run scripts/run_experiment.py first to "
            "generate real task splits — this script refuses to substitute "
            "synthetic data for a real training run."
        )
    examples: List[SimpleNamespace] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(SimpleNamespace(**json.loads(line)))
    if not examples:
        raise ValueError(f"{path} is empty — no real training examples to train on.")
    return examples


def build_examples(
    tasks: List[SimpleNamespace],
    packs_dir: Optional[Path],
    condition: str = "C",
) -> List[Dict[str, str]]:
    """Training uses the condition-C context shape (packs, no graph): the
    adapter learns the repository's conventions, and the graph stays an
    inference-time input so condition D remains a separable variable.

    Prompts are built with the same src.prompting helpers the evaluator uses,
    so C/D are not handicapped by a train/eval format mismatch.
    """
    out: List[Dict[str, str]] = []
    for t in tasks:
        target = build_target(t)
        if target == '{"impacted": []}':
            # Training on an empty answer teaches the model to answer nothing.
            continue
        out.append(
            {
                "task_id": t.task_id,
                "prompt": build_prompt(t, condition, packs_dir, graph=None),
                "target": target,
            }
        )
    if not out:
        raise ValueError(
            "every training task resolved to an empty answer; nothing to train on"
        )
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def check_temporal_integrity(
    packs_dir: Optional[Path], tasks: List[SimpleNamespace]
) -> Dict[str, Any]:
    """Detect future state reaching historical training prompts.

    File packs are generated from one repository snapshot. When that snapshot
    is HEAD and the training tasks come from earlier commits, each prompt
    carries source, dependency and test metadata that did not exist at the
    task's commit — including changes from the very commits held out for
    evaluation. The temporal family split does not prevent this, and the
    contamination audit cannot see it, because it compares task metadata and
    not pack contents.

    Fixing it properly means generating packs at each task's commit, which is
    follow-up work. Detecting and recording it is not: a run must not be able
    to claim temporal integrity it does not have.
    """
    if packs_dir is None:
        return {
            "status": "NO_PACKS",
            "note": "training prompts carry no pack contents, so no pack-state leakage",
        }
    index_path = packs_dir / "index.json"
    if not index_path.exists():
        return {"status": "UNKNOWN", "note": f"no pack index at {index_path}"}
    packs_head = json.loads(index_path.read_text(encoding="utf-8")).get("head_sha")
    train_commits = sorted({str(getattr(t, "commit_sha", "")) for t in tasks})
    historical = [c for c in train_commits if c and c != packs_head]
    if not packs_head:
        return {"status": "UNKNOWN", "note": "pack index records no head_sha"}
    if not historical:
        return {"status": "OK", "packs_head_sha": packs_head}
    return {
        "status": "HEAD_STATE_PACKS",
        "packs_head_sha": packs_head,
        "training_commits_predating_packs": historical[:20],
        "note": (
            "Training prompts contain repository state from the pack snapshot, "
            "not from each task's own commit. Post-cutoff content — including "
            "state from evaluation commits — can therefore reach the adapter. "
            "Treat conditions C/D from this adapter as temporally contaminated "
            "until packs are generated per-commit."
        ),
    }


def encode(tokenizer, prompt: str, target: str, max_seq_length: int):
    """Tokenize one example, masking the prompt so loss is computed on the
    completion only.

    Training on the prompt too would let loss fall by memorising repository
    context that is supplied at inference anyway — the adapter is supposed to
    learn conventions, not facts that live in the packs.
    """
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
    budget = max_seq_length - len(target_ids)
    if budget < 1:
        target_ids = target_ids[: max_seq_length - 1]
        budget = 1
    prompt_ids = prompt_ids[-budget:]

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + list(target_ids)
    return (
        torch.tensor([input_ids], dtype=torch.long),
        torch.tensor([labels], dtype=torch.long),
    )


# --------------------------------------------------------------------------
# Device / precision selection
# --------------------------------------------------------------------------

def select_backend(allow_4bit: bool = True) -> Dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        return {"device": "cpu", "quantized": False, "gpu_name": None}
    try:
        import bitsandbytes  # noqa: F401

        quantized = allow_4bit
    except ImportError:
        quantized = False
    return {
        "device": "cuda",
        "quantized": quantized,
        "gpu_name": torch.cuda.get_device_name(0),
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Real repo-specific LoRA training")
    p.add_argument("--train_jsonl", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    p.add_argument("--out", type=Path, default=Path("results/adapters/repo-qlora"))
    p.add_argument("--telemetry", type=Path, default=None)
    p.add_argument("--packs_dir", type=Path, default=None,
                   help="File-pack directory; training prompts then match condition C.")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max_seq_length", "--max_seq_len", dest="max_seq_length",
                   type=int, default=2048)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repo_head", default=None,
                   help="Repository HEAD sha this training run belongs to; recorded "
                        "in adapter_provenance.json so evaluation can verify reuse.")
    p.add_argument("--no_4bit", action="store_true",
                   help="Force full-precision LoRA even when bitsandbytes is available.")
    args = p.parse_args()

    import torch

    torch.manual_seed(args.seed)

    backend = select_backend(allow_4bit=not args.no_4bit)
    print(f"=== backend: {backend} ===")

    tasks = load_train_examples(args.train_jsonl)
    examples = build_examples(tasks, args.packs_dir)

    temporal = check_temporal_integrity(args.packs_dir, tasks)
    if temporal["status"] == "HEAD_STATE_PACKS":
        print("=== WARNING: temporal integrity — " + temporal["note"] + " ===")
    else:
        print(f"=== temporal integrity: {temporal['status']} ===")
    print(f"=== {len(tasks)} real tasks -> {len(examples)} supervisable examples "
          f"from {args.train_jsonl} ===")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if backend["device"] == "cuda" and backend["quantized"]:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["device_map"] = "auto"
        print("=== real 4-bit QLoRA path (GPU + bitsandbytes) ===")
    elif backend["device"] == "cuda":
        load_kwargs["dtype"] = torch.bfloat16
        load_kwargs["device_map"] = "auto"
        print("=== GPU present, bitsandbytes missing — real bf16 LoRA, no 4-bit ===")
    else:
        load_kwargs["dtype"] = torch.float32
        print("=== CPU fallback — real fp32 LoRA, no quantization (slower, more RAM) ===")

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    if backend["device"] == "cuda" and backend["quantized"]:
        model = prepare_model_for_kbit_training(model)
    if backend["device"] == "cpu":
        model = model.to("cpu")

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.train()

    hook = TrainingTelemetryHook(model)
    hook.attach()
    print(f"=== telemetry tracking {len(hook.tracked)} LoRA projections ===")

    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        [q for q in model.parameters() if q.requires_grad], lr=args.lr
    )

    telemetry_records: List[Dict[str, Any]] = []
    losses: List[Dict[str, Any]] = []
    n = len(examples)
    started = time.time()

    for step in range(args.steps):
        t0 = time.time()
        hook.reset()
        optimizer.zero_grad(set_to_none=True)

        accum_loss = 0.0
        step_task_ids: List[str] = []
        for micro in range(args.grad_accum):
            example = examples[(step * args.grad_accum + micro) % n]
            step_task_ids.append(example["task_id"])
            input_ids, labels = encode(
                tokenizer, example["prompt"], example["target"], args.max_seq_length
            )
            out = model(input_ids=input_ids.to(device), labels=labels.to(device))
            loss = out.loss / args.grad_accum
            loss.backward()
            accum_loss += float(loss.item())

        torch.nn.utils.clip_grad_norm_(
            [q for q in model.parameters() if q.requires_grad], max_norm=0.3
        )
        optimizer.step()

        compute_ms = round((time.time() - t0) * 1000.0, 1)
        telemetry_records.append(
            hook.collect(step, accum_loss, compute_ms, step_task_ids).to_json()
        )
        losses.append({"step": step, "loss": accum_loss})

        if step % 10 == 0 or step == args.steps - 1:
            print(f"step {step:4d}  loss={accum_loss:.4f}  {compute_ms:.0f}ms")

    hook.detach()
    elapsed = time.time() - started

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out))
    tokenizer.save_pretrained(str(args.out))

    weights = args.out / "adapter_model.safetensors"
    if not weights.exists():
        raise RuntimeError(
            f"training finished but no adapter weights at {weights}; refusing to "
            "report a trained adapter that does not exist"
        )
    print(f"=== real adapter saved to {args.out} ===")

    provenance = {
        "base_model": args.model,
        "repo_head_sha": args.repo_head,
        "train_split_sha256": sha256_file(args.train_jsonl),
        "train_jsonl": str(args.train_jsonl),
        "packs_head_sha": temporal.get("packs_head_sha"),
        "steps": args.steps,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "temporal_integrity": temporal,
    }
    (args.out / "adapter_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )

    telemetry_path = args.telemetry or (args.out / "train_telemetry.jsonl")
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    with telemetry_path.open("w", encoding="utf-8") as fh:
        for rec in telemetry_records:
            fh.write(json.dumps(rec) + "\n")
    print(f"=== telemetry written to {telemetry_path} ({len(telemetry_records)} steps) ===")

    tele_summary = summarize_telemetry(telemetry_records)
    trace = {
        "status": "TRAINED",
        "base_model": args.model,
        "train_jsonl": str(args.train_jsonl),
        "backend": backend,
        "example_count": len(examples),
        "steps": args.steps,
        "grad_accum": args.grad_accum,
        "learning_rate": args.lr,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout},
        "device": str(device),
        "elapsed_seconds": round(elapsed, 2),
        "first_loss": losses[0]["loss"] if losses else None,
        "last_loss": losses[-1]["loss"] if losses else None,
        "loss_curve": losses,
        "telemetry_path": str(telemetry_path),
        "telemetry_summary": tele_summary,
        "temporal_integrity": temporal,
    }
    (args.out / "training_trace.json").write_text(
        json.dumps(trace, indent=2), encoding="utf-8"
    )

    # A stale NOT_TRAINED marker sitting next to real weights would be a lie.
    stale = args.out / "NOT_TRAINED.txt"
    if stale.exists():
        stale.unlink()

    print(
        f"=== loss {trace['first_loss']:.4f} -> {trace['last_loss']:.4f} | "
        f"{tele_summary['layers_moved']}/{tele_summary['tracked_layers']} "
        f"tracked projections received gradient ==="
    )


if __name__ == "__main__":
    main()
