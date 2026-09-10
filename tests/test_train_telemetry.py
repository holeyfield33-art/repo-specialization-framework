"""
Tests for real training telemetry.

The telemetry answers "did tuning actually push any layer to change?" — which
is what separates conditions C/D from B. If the hook silently attached to
nothing, the run would report success with an empty evidence file, so the
failure modes get tested rather than assumed.

Model-dependent tests build a tiny Qwen2 from config (no download).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from scripts.train_real import (  # noqa: E402
    TARGET_MODULES,
    StepTelemetry,
    TrainingTelemetryHook,
    build_examples,
    encode,
    load_train_examples,
    select_backend,
    summarize_telemetry,
)


@pytest.fixture(scope="module")
def tiny_peft_model():
    """A real (tiny) Qwen2 wrapped in real LoRA — no weights downloaded."""
    from transformers import AutoConfig, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    cfg = AutoConfig.for_model(
        "qwen2",
        vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model = AutoModelForCausalLM.from_config(cfg)
    lora = LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0,
        target_modules=TARGET_MODULES, bias="none", task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora)


# --- hook wiring -------------------------------------------------------------

def test_hook_attaches_to_every_adapted_projection(tiny_peft_model):
    hook = TrainingTelemetryHook(tiny_peft_model)
    hook.attach()
    try:
        # 2 layers x 7 target projections
        assert len(hook.tracked) == 2 * len(TARGET_MODULES)
        assert "0.q_proj" in hook.tracked
        assert "1.down_proj" in hook.tracked
    finally:
        hook.detach()


def test_hook_keys_identify_the_real_layer_index(tiny_peft_model):
    """Keys must be traceable to an actual transformer layer, not a counter."""
    hook = TrainingTelemetryHook(tiny_peft_model)
    hook.attach()
    try:
        layers = {k.split(".", 1)[0] for k in hook.tracked}
        assert layers == {"0", "1"}
    finally:
        hook.detach()


def test_hook_covers_attention_not_just_mlp(tiny_peft_model):
    hook = TrainingTelemetryHook(tiny_peft_model)
    hook.attach()
    try:
        projections = {k.split(".", 1)[1] for k in hook.tracked}
        assert {"q_proj", "k_proj", "v_proj", "o_proj"} <= projections
        assert {"gate_proj", "up_proj", "down_proj"} <= projections
    finally:
        hook.detach()


def test_hook_refuses_to_attach_to_an_unadapted_model():
    """Reporting gradient evidence for zero tracked layers would be a lie."""
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.for_model(
        "qwen2", vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=32,
    )
    bare = AutoModelForCausalLM.from_config(cfg)  # no LoRA
    with pytest.raises(RuntimeError, match="zero LoRA projections"):
        TrainingTelemetryHook(bare).attach()


def test_hook_records_real_gradients_on_a_backward_pass(tiny_peft_model):
    hook = TrainingTelemetryHook(tiny_peft_model)
    hook.attach()
    try:
        hook.reset()
        ids = torch.randint(0, 128, (1, 16))
        out = tiny_peft_model(input_ids=ids, labels=ids)
        out.loss.backward()
        tele = hook.collect(step=0, loss=float(out.loss), compute_ms=1.0, task_ids=["t"])
        assert tele.weight_grad_norms, "no weight gradients captured"
        assert tele.grad_norms, "no block gradients captured"
        assert all(v >= 0 for v in tele.weight_grad_norms.values())
        assert any(v > 0 for v in tele.weight_grad_norms.values()), (
            "every gradient was exactly zero — nothing was learned"
        )
    finally:
        hook.detach()
        tiny_peft_model.zero_grad(set_to_none=True)


def test_detach_removes_every_handle(tiny_peft_model):
    hook = TrainingTelemetryHook(tiny_peft_model)
    hook.attach()
    hook.detach()
    assert hook._handles == []


# --- serialization -----------------------------------------------------------

def test_step_telemetry_is_json_serializable():
    tele = StepTelemetry(
        step=3, loss=1.23456789, compute_ms=42.0, task_ids=["a"],
        grad_norms={"0.q_proj": 1.0}, weight_grad_norms={"0.q_proj": 2.0},
    )
    payload = json.loads(json.dumps(tele.to_json()))
    assert payload["step"] == 3
    assert payload["task_ids"] == ["a"]
    assert payload["loss"] == pytest.approx(1.234568, abs=1e-6)


# --- summary -----------------------------------------------------------------

def test_summary_reports_which_layers_moved():
    records = [
        {"weight_grad_norms": {"0.q_proj": 1.0, "1.q_proj": 0.0}},
        {"weight_grad_norms": {"0.q_proj": 3.0, "1.q_proj": 0.0}},
    ]
    s = summarize_telemetry(records)
    assert s["steps"] == 2
    assert s["tracked_layers"] == 2
    assert s["layers_moved"] == 1
    assert s["all_layers_moved"] is False
    assert s["quiet_layers"] == ["1.q_proj"]
    assert s["top_layers_by_mean_weight_grad"][0][0] == "0.q_proj"


def test_summary_handles_an_empty_run():
    assert summarize_telemetry([]) == {
        "steps": 0, "layers_moved": 0, "tracked_layers": 0
    }


# --- data path ---------------------------------------------------------------

def test_missing_split_fails_loudly_instead_of_faking_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="refuses to substitute"):
        load_train_examples(tmp_path / "nope.jsonl")


def test_empty_split_fails_loudly(tmp_path):
    empty = tmp_path / "train.jsonl"
    empty.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_train_examples(empty)


def test_build_examples_skips_unsupervisable_tasks():
    from types import SimpleNamespace

    tasks = [
        SimpleNamespace(task_id="good", instruction="i", context_files=["a.js"],
                        ground_truth={"impacted": ["b.js"]}),
        SimpleNamespace(task_id="empty", instruction="i", context_files=["a.js"],
                        ground_truth={"impacted": []}),
    ]
    built = build_examples(tasks, packs_dir=None)
    assert [e["task_id"] for e in built] == ["good"]


def test_build_examples_raises_when_nothing_is_supervisable():
    from types import SimpleNamespace

    tasks = [SimpleNamespace(task_id="e", instruction="i", context_files=[],
                             ground_truth={"impacted": []})]
    with pytest.raises(ValueError, match="nothing to train on"):
        build_examples(tasks, packs_dir=None)


def test_encode_masks_the_prompt_so_loss_is_on_the_answer_only():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    ids, labels = encode(tok, "some long prompt text", '{"impacted": ["a.js"]}', 128)
    assert ids.shape == labels.shape
    masked = (labels == -100).sum().item()
    assert masked > 0, "prompt tokens must be masked out of the loss"
    assert masked < labels.numel(), "the answer must remain supervised"
    # the unmasked tail must be exactly the target tokens
    assert (labels[labels != -100] == ids[labels != -100]).all()


def test_encode_keeps_the_answer_when_the_prompt_overflows():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    ids, labels = encode(tok, "word " * 5000, '{"impacted": ["a.js"]}', 64)
    assert ids.shape[-1] <= 64
    assert (labels != -100).sum().item() > 0, "answer was truncated away"


# --- backend selection -------------------------------------------------------

def test_select_backend_reports_a_usable_shape():
    b = select_backend()
    assert b["device"] in ("cpu", "cuda")
    assert isinstance(b["quantized"], bool)
    if b["device"] == "cpu":
        assert b["quantized"] is False


def test_no_4bit_flag_disables_quantization():
    assert select_backend(allow_4bit=False)["quantized"] is False
