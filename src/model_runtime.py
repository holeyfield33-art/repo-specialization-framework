"""Real Hugging Face inference for RSEF conditions.

Heavy ML imports are intentionally lazy so ingestion and unit tests remain usable
on CPU-only machines.  This module never substitutes simulated output.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


OUTPUT_SCHEMA = {
    "predicted_files": ["path/from/repository"],
    "predicted_tests": ["test/path"],
    "primary_files": ["bug/source/path"],
    "related_files": ["dependency/path"],
    "suggested_checks": ["short invariant/check"],
    "referenced_apis": ["symbolName"],
    "patch": "unified diff or null",
    "analysis": "brief evidence-grounded explanation",
}


class InferenceError(RuntimeError):
    pass


@dataclass
class Generation:
    parsed: Dict[str, Any]
    raw_text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    parse_error: Optional[str] = None


def require_cuda() -> Dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise InferenceError("PyTorch is not installed") from exc
    if not torch.cuda.is_available():
        raise InferenceError("CUDA GPU is required; refusing simulated or CPU experiment output")
    props = torch.cuda.get_device_properties(0)
    return {
        "gpu_model": torch.cuda.get_device_name(0),
        "vram_bytes": int(props.total_memory),
        "vram_gib": round(props.total_memory / 1024**3, 2),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
    }


def _extract_json(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    start, end = stripped.find("{"), stripped.rfind("}")
    if start >= 0 and end > start:
        candidates.append(stripped[start:end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    raise InferenceError("model output did not contain one valid JSON object")


class HFGenerator:
    def __init__(
        self,
        model_name: str,
        adapter_path: Optional[Path] = None,
        max_input_tokens: int = 2048,
        max_new_tokens: int = 512,
    ):
        require_cuda()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant,
            device_map="auto",
            trust_remote_code=True,
        )
        if adapter_path is not None:
            from peft import PeftModel
            config = adapter_path / "adapter_config.json"
            if not config.is_file():
                raise InferenceError(f"missing trained adapter config: {config}")
            model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
        self.model = model.eval()

    def generate(self, instruction: str, context: str) -> Generation:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are evaluating a software repository. Use only supplied evidence. "
                    "Return exactly one JSON object matching this schema: " + json.dumps(OUTPUT_SCHEMA)
                ),
            },
            {"role": "user", "content": instruction + "\n\nREPOSITORY CONTEXT:\n" + context},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        device = next(self.model.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        started = time.perf_counter()
        with self.torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        latency = (time.perf_counter() - started) * 1000
        input_count = int(encoded["input_ids"].shape[-1])
        new_ids = generated[0, input_count:]
        raw = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        try:
            parsed = _extract_json(raw)
            parse_error = None
        except InferenceError as exc:
            parsed = {}
            parse_error = str(exc)
        return Generation(parsed, raw, input_count, int(new_ids.shape[-1]), latency, parse_error)
