"""
Tests for the real/simulated split introduced alongside real training.

The central guarantee under test: a run that claims to be real must have
actually loaded adapter weights, and a run that did not must be unmistakably
labelled as simulated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import (
    aggregate_metrics,
    condition_provenance,
    evaluate_condition,
    require_adapter,
    run_provenance,
)
from src.history_tasks import TaskExample
from src.prompting import build_prompt, build_target, parse_answer


def make_task(task_id: str = "fam-1-impact") -> TaskExample:
    return TaskExample(
        task_id=task_id,
        task_type="change_impact_prediction",
        family_id="fam-1",
        commit_sha="abc123",
        instruction="List the files impacted by changes to src/detector.js.",
        context_files=["src/detector.js"],
        ground_truth={"impacted": ["src/policy.js", "src/hash.js"], "seeds": ["src/detector.js"]},
        evidence=["graph:v1"],
        split="eval",
    )


# --- the guarantee: real mode never silently simulates -----------------------

def test_real_mode_without_adapter_raises(tmp_path):
    """evaluate_condition(mode="real") for a tuned condition with no adapter
    must raise, not fall back to _simulate_model_response."""
    empty_adapter = tmp_path / "adapters" / "qwen-repo-qlora"
    empty_adapter.mkdir(parents=True)
    (empty_adapter / "NOT_TRAINED.txt").write_text("no adapter here\n")

    with pytest.raises(RuntimeError) as exc:
        evaluate_condition(
            [make_task()],
            "C",
            packs_dir=None,
            graph=None,
            adapter_path=empty_adapter,
            mode="real",
        )
    assert "adapter_model.safetensors" in str(exc.value)
    assert "UNTESTED" in str(exc.value)


def test_real_mode_with_adapter_path_none_raises():
    with pytest.raises(RuntimeError):
        evaluate_condition(
            [make_task()], "D", packs_dir=None, graph=None, adapter_path=None, mode="real"
        )


def test_require_adapter_accepts_real_weights(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"\x00")
    assert require_adapter(adapter, "C") == adapter


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        evaluate_condition([make_task()], "A", None, None, mode="pretend")


# --- simulated results must be labelled as such ------------------------------

def test_simulate_mode_labels_every_result():
    rows = evaluate_condition([make_task()], "D", packs_dir=None, graph=None, mode="simulate")
    assert rows, "eval-split task should produce a result"
    assert all(r.data_provenance == "simulated" for r in rows)


def test_simulate_mode_is_the_default():
    rows = evaluate_condition([make_task()], "A", packs_dir=None, graph=None)
    assert all(r.data_provenance == "simulated" for r in rows)


def test_simulated_notes_carry_loud_prefix():
    rows = evaluate_condition([make_task()], "B", packs_dir=None, graph=None, mode="simulate")
    assert rows[0].notes.startswith("[SIMULATED-NOT-EVIDENCE-B]")


def test_aggregate_metrics_propagates_provenance():
    by_cond = {
        c: evaluate_condition([make_task()], c, packs_dir=None, graph=None, mode="simulate")
        for c in ("A", "B", "C", "D")
    }
    summary = aggregate_metrics(by_cond)
    assert all(s["data_provenance"] == "simulated" for s in summary.values())
    assert run_provenance(summary) == "simulated"


def test_empty_condition_is_labelled_none():
    summary = aggregate_metrics({"A": []})
    assert summary["A"] == {"n": 0, "data_provenance": "none"}


def test_run_provenance_flags_mixed_runs():
    assert run_provenance({"A": {"data_provenance": "real"}, "B": {"data_provenance": "simulated"}}) == "mixed"
    assert run_provenance({"A": {"data_provenance": "real"}}) == "real"
    assert run_provenance({}) == "none"


def test_condition_provenance_of_empty_rows():
    assert condition_provenance([]) == "none"


# --- context assembly is the independent variable ----------------------------

def test_prompt_differs_per_condition(tmp_path):
    packs = tmp_path / "packs"
    packs.mkdir()
    (packs / "src__detector.js.pack.json").write_text(
        json.dumps(
            {
                "path": "src/detector.js",
                "source_sha": "deadbeefcafe",
                "exact_source": "function detect() {}",
                "imports": ["src/policy.js"],
                "exports": ["detect"],
                "symbols": ["detect"],
                "callers": [],
                "callees": ["src/policy.js"],
                "related_tests": ["test/detector-unit-test.js"],
                "dependency_edges": [
                    {"src": "src/detector.js", "dst": "src/policy.js", "type": "IMPORTS"}
                ],
            }
        )
    )
    task = make_task()
    prompt_a = build_prompt(task, "A", packs, graph=None)
    prompt_b = build_prompt(task, "B", packs, graph=None)

    assert "Repository context:" in prompt_a
    assert "Structured file packs:" not in prompt_a
    assert "Structured file packs:" in prompt_b
    assert "related_tests" in prompt_b
    assert prompt_a != prompt_b


def test_graph_block_only_for_condition_d():
    class StubGraph:
        version = "v1"

        def impact_neighborhood(self, seeds, radius=1, max_nodes=25):
            return {
                "seeds": seeds,
                "nodes": ["src/policy.js"],
                "edges": [{"src": "src/detector.js", "dst": "src/policy.js", "type": "IMPORTS"}],
                "uncertainty": [],
                "radius": radius,
                "graph_version": "v1",
            }

    task = make_task()
    assert "impact neighborhood" not in build_prompt(task, "C", None, StubGraph())
    assert "impact neighborhood" in build_prompt(task, "D", None, StubGraph())


def test_build_target_matches_evaluator_schema():
    assert json.loads(build_target(make_task())) == {
        "impacted": ["src/policy.js", "src/hash.js"]
    }


# --- unparseable model output is a failure, never a fallback score -----------

def test_parse_answer_extracts_json():
    assert parse_answer('sure!\n{"impacted": ["a.js"]}\n') == ["a.js"]
    assert parse_answer('{"note": {"x": 1}, "impacted": ["a.js", "b.js"]}') == ["a.js", "b.js"]


def test_parse_answer_returns_none_on_garbage():
    assert parse_answer("I cannot answer that.") is None
    assert parse_answer('{"impacted": "not-a-list"}') is None
    assert parse_answer("{broken json") is None


# --- the known task-design leak stays visible --------------------------------

def test_answer_leak_is_annotated():
    leaky = make_task()
    leaky.context_files = ["src/detector.js", "src/policy.js"]  # contains the answer
    rows = evaluate_condition([leaky], "A", packs_dir=None, graph=None, mode="simulate")
    assert "ground-truth paths present in task.context_files" in rows[0].notes
