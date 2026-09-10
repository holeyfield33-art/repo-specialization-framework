"""
End-to-end regression tests for the experiment runner.

Guards the two behaviours the real/simulated split exists to protect:
  * a default run stays backward compatible AND labels itself simulated
  * a run that claims to be real cannot proceed without adapter weights
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

CONDITIONS = ("A", "B", "C", "D")


def run_experiment(repo: Path, out: Path, *extra: str):
    return subprocess.run(
        [sys.executable, "-m", "scripts.run_experiment",
         "--repo", str(repo), "--out", str(out), *extra],
        cwd=ROOT, capture_output=True, text=True,
    )


@pytest.fixture(scope="module")
def default_run(sample_repo, tmp_path_factory):
    out = tmp_path_factory.mktemp("default-run")
    proc = run_experiment(sample_repo, out)
    assert proc.returncode == 0, f"default run failed:\n{proc.stdout}\n{proc.stderr}"
    return {"out": out, "proc": proc}


# --- backward compatibility --------------------------------------------------

def test_default_run_produces_every_artifact(default_run):
    out = default_run["out"]
    for name in (
        "summary.json", "metrics_raw.json", "dashboard.html", "streamlit_app.py",
        "experiment_index.json", "contamination_report.json",
        "verification_gate.json", "qlora_config.yaml",
    ):
        assert (out / name).exists(), f"missing artifact: {name}"


def test_default_run_evaluates_all_four_conditions(default_run):
    summary = json.loads((default_run["out"] / "summary.json").read_text())
    assert set(summary) == set(CONDITIONS)
    assert all(summary[c]["n"] > 0 for c in CONDITIONS)


# --- provenance labelling ----------------------------------------------------

def test_default_run_labels_every_condition_simulated(default_run):
    summary = json.loads((default_run["out"] / "summary.json").read_text())
    assert all(summary[c]["data_provenance"] == "simulated" for c in CONDITIONS)


def test_every_raw_metric_row_is_labelled(default_run):
    raw = json.loads((default_run["out"] / "metrics_raw.json").read_text())
    rows = [r for rows in raw.values() for r in rows]
    assert rows
    assert all(r["data_provenance"] == "simulated" for r in rows)
    assert all(r["notes"].startswith("[SIMULATED-NOT-EVIDENCE-") for r in rows)


def test_dashboard_shows_the_warning_banner(default_run):
    html = (default_run["out"] / "dashboard.html").read_text()
    assert '"provenance": "simulated"' in html
    assert "SIMULATED RESULTS" in html
    assert "not experimental evidence" in html


def test_index_records_how_the_run_was_produced(default_run):
    idx = json.loads((default_run["out"] / "experiment_index.json").read_text())
    assert idx["data_provenance"] == "simulated"
    assert idx["eval_mode"] == "simulate"
    assert idx["real_train"] is False
    assert idx["adapter_trained"] is False


def test_stdout_warns_the_operator(default_run):
    assert "not experimental evidence" in default_run["proc"].stdout.lower()


# --- the adapter marker ------------------------------------------------------

def test_not_trained_marker_replaces_the_old_placeholder(default_run):
    adapter = default_run["out"] / "adapters" / "qwen-repo-qlora"
    assert not (adapter / "ADAPTER_PLACEHOLDER.txt").exists()
    marker = adapter / "NOT_TRAINED.txt"
    assert marker.exists()
    text = marker.read_text()
    assert "UNTESTED" in text
    assert "not merely untuned" in text
    assert not (adapter / "adapter_model.safetensors").exists()


def test_runner_no_longer_emits_the_unused_train_script(default_run):
    """results/train_qlora.py was written but never executed — emitting it was
    part of how the pipeline looked trained without training."""
    assert not (default_run["out"] / "train_qlora.py").exists()


# --- real mode refuses to fake it --------------------------------------------

def test_eval_mode_real_without_adapter_fails_the_run(sample_repo, tmp_path):
    proc = run_experiment(sample_repo, tmp_path / "real-fail", "--eval-mode", "real")
    assert proc.returncode != 0, "real eval without an adapter must not succeed"
    combined = proc.stdout + proc.stderr
    assert "adapter_model.safetensors" in combined
    assert "UNTESTED" in combined
    assert not (tmp_path / "real-fail" / "summary.json").exists(), (
        "a failed real run must not leave a summary anyone could quote"
    )


def test_contamination_audit_still_passes(default_run):
    contam = json.loads((default_run["out"] / "contamination_report.json").read_text())
    assert contam["status"] == "PASS"
    assert contam["violations"] == []


def test_verification_gate_still_passes(default_run):
    gate = json.loads((default_run["out"] / "verification_gate.json").read_text())
    assert gate["status"] == "PASS"
