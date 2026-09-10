"""
Regression tests for the five findings raised in code review on PR #1.

Each one was verified against the code before being fixed, and each gets a test
here so it cannot come back quietly. Ordered by the finding's severity.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import (
    aggregate_metrics,
    evaluate_condition,
    scoreability_report,
    verify_adapter_provenance,
)
from src.history_tasks import TaskExample
from src.prompting import expected_answer_paths, is_scoreable


def task(task_id, task_type, ground_truth, context_files=("src/a.js",), split="eval"):
    return TaskExample(
        task_id=task_id, task_type=task_type, family_id="fam", commit_sha="sha",
        instruction="i", context_files=list(context_files),
        ground_truth=ground_truth, evidence=[], split=split,
    )


SCOREABLE = task("t-impact", "change_impact_prediction",
                 {"impacted": ["src/b.js"], "seeds": ["src/a.js"]})
REVIEW = task("t-review", "code_review",
              {"commit_message": "m", "files": ["src/a.js"], "suggested_checks": ["x"]})
XFILE = task("t-xfile", "cross_file_dependency_reasoning",
             {"edges": [{"src": "src/a.js", "dst": "src/b.js", "type": "IMPORTS"}]})


# --- P1: reasoning tasks were scored against an empty answer -----------------

def test_tasks_without_a_comparable_answer_are_not_scoreable():
    assert is_scoreable(SCOREABLE)
    assert not is_scoreable(REVIEW), "code_review stores files/suggested_checks"
    assert not is_scoreable(XFILE), "cross_file_dependency_reasoning stores edges"


def test_unscoreable_tasks_are_excluded_from_results():
    rows = evaluate_condition([SCOREABLE, REVIEW, XFILE], "A", None, None, mode="simulate")
    assert [r.task_id for r in rows] == ["t-impact"], (
        "an unscoreable task produced a metric row; any non-empty answer would "
        "have counted as a success"
    )


def test_scoreability_report_counts_what_it_dropped():
    report = scoreability_report([SCOREABLE, REVIEW, XFILE])
    assert report["eval_tasks_total"] == 3
    assert report["scoreable"] == 1
    assert report["excluded"] == 2
    assert report["excluded_by_task_type"] == {
        "code_review": 1, "cross_file_dependency_reasoning": 1
    }


def test_scoreability_report_ignores_non_eval_splits():
    train = task("t-train", "change_impact_prediction", {"impacted": ["b.js"]}, split="train")
    assert scoreability_report([train, SCOREABLE])["eval_tasks_total"] == 1


def test_summary_surfaces_the_exclusion_count():
    rows = {"A": evaluate_condition([SCOREABLE], "A", None, None, mode="simulate")}
    report = scoreability_report([SCOREABLE, REVIEW, XFILE])
    assert aggregate_metrics(rows, report)["A"]["unscoreable_excluded"] == 2


# --- P1: evaluation truncated the prompt from the wrong side -----------------

class _StubTokenizer:
    chat_template = None
    pad_token_id = 0
    truncation_side = "right"

    def __init__(self):
        self.seen_kwargs = None

    def __call__(self, text, **kwargs):
        import torch

        self.seen_kwargs = kwargs
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    def decode(self, ids, skip_special_tokens=True):
        return '{"impacted": ["src/b.js"]}'


class _StubModel:
    def __init__(self):
        import torch

        self.device = torch.device("cpu")

    def generate(self, **kwargs):
        import torch

        return torch.tensor([[1, 2, 3, 4, 5]])


def test_evaluation_truncates_from_the_left_like_training(monkeypatch):
    """Right truncation drops the tail of the prompt, which holds the answer
    schema and, for condition D, the whole graph neighborhood."""
    pytest.importorskip("torch")
    import src.evaluation as ev

    tok, model = _StubTokenizer(), _StubModel()
    monkeypatch.setattr(ev, "_load_model", lambda *a, **k: (model, tok))

    result = ev.real_model_response(SCOREABLE, "A", None, None, None, "stub-model")

    assert tok.truncation_side == "left", (
        "evaluation must truncate from the left, matching train_real.encode"
    )
    assert tok.seen_kwargs["truncation"] is True
    assert result["predicted_impacted"] == ["src/b.js"]
    assert result["success"] is True


def test_scoring_an_unscoreable_task_raises_rather_than_guessing(monkeypatch):
    """Defence in depth: evaluate_condition filters these out, but if that
    filter ever regresses the scorer itself must refuse rather than award a
    free success to any non-empty prediction."""
    pytest.importorskip("torch")
    import src.evaluation as ev

    monkeypatch.setattr(ev, "_load_model", lambda *a, **k: (_StubModel(), _StubTokenizer()))
    with pytest.raises(RuntimeError, match="no comparable ground-truth"):
        ev.real_model_response(REVIEW, "A", None, None, None, "stub-model")


# --- P1: a foreign adapter was accepted on filename alone --------------------

def _adapter(tmp_path, **provenance):
    d = tmp_path / "adapter"
    d.mkdir(exist_ok=True)
    (d / "adapter_model.safetensors").write_bytes(b"\x00")
    if provenance:
        (d / "adapter_provenance.json").write_text(json.dumps(provenance))
    return d


EXPECTED = {"base_model": "Qwen/Q", "repo_head_sha": "abc123", "train_split_sha256": "ff"}


def test_adapter_without_provenance_is_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="adapter_provenance.json"):
        verify_adapter_provenance(_adapter(tmp_path), EXPECTED)


def test_adapter_from_another_repo_revision_is_rejected(tmp_path):
    stale = dict(EXPECTED, repo_head_sha="deadbeef")
    with pytest.raises(RuntimeError, match="not trained for this run"):
        verify_adapter_provenance(_adapter(tmp_path, **stale), EXPECTED)


def test_adapter_from_another_base_model_is_rejected(tmp_path):
    other = dict(EXPECTED, base_model="some/other-model")
    with pytest.raises(RuntimeError, match="not trained for this run"):
        verify_adapter_provenance(_adapter(tmp_path, **other), EXPECTED)


def test_adapter_trained_on_a_different_split_is_rejected(tmp_path):
    other = dict(EXPECTED, train_split_sha256="00")
    with pytest.raises(RuntimeError, match="not trained for this run"):
        verify_adapter_provenance(_adapter(tmp_path, **other), EXPECTED)


def test_matching_adapter_verifies(tmp_path):
    out = verify_adapter_provenance(_adapter(tmp_path, **EXPECTED), EXPECTED)
    assert out["status"] == "VERIFIED"
    assert out["mismatches"] == {}


def test_override_is_explicit_and_labelled(tmp_path):
    stale = dict(EXPECTED, repo_head_sha="deadbeef")
    out = verify_adapter_provenance(
        _adapter(tmp_path, **stale), EXPECTED, allow_unverified=True
    )
    assert out["status"] == "MISMATCH_ACCEPTED"
    assert "repo_head_sha" in out["mismatches"]


# --- P1: HEAD-state packs leak future state into historical training ---------

def test_temporal_check_flags_head_state_packs(tmp_path):
    from scripts.train_real import check_temporal_integrity

    packs = tmp_path / "packs"
    packs.mkdir()
    (packs / "index.json").write_text(json.dumps({"head_sha": "head999"}))
    tasks = [SimpleNamespace(commit_sha="early111"), SimpleNamespace(commit_sha="early222")]
    out = check_temporal_integrity(packs, tasks)
    assert out["status"] == "HEAD_STATE_PACKS"
    assert out["packs_head_sha"] == "head999"
    assert "early111" in out["training_commits_predating_packs"]


def test_temporal_check_is_ok_when_packs_match_the_task_commit(tmp_path):
    from scripts.train_real import check_temporal_integrity

    packs = tmp_path / "packs"
    packs.mkdir()
    (packs / "index.json").write_text(json.dumps({"head_sha": "same"}))
    out = check_temporal_integrity(packs, [SimpleNamespace(commit_sha="same")])
    assert out["status"] == "OK"


def test_temporal_check_reports_no_packs():
    from scripts.train_real import check_temporal_integrity

    assert check_temporal_integrity(None, [])["status"] == "NO_PACKS"


def test_pack_index_records_the_commit_it_was_built_from(pipeline):
    index = json.loads((pipeline["packs_dir"] / "index.json").read_text())
    assert index["head_sha"], "pack index must record its source commit"
    assert index["head_sha"] == pipeline["manifest"].head_sha


# --- P2: model output reached the dashboard as markup ------------------------

HOSTILE = "</script><script>alert('xss')</script>"


def test_dashboard_payload_cannot_close_the_script_tag(tmp_path):
    from src.dashboard import build_dashboard

    out = tmp_path / "d.html"
    build_dashboard(
        {"A": {"n": 1, "data_provenance": "real"}}, {"status": "PASS"}, {"status": "PASS"},
        [{"task_id": "t", "A": HOSTILE, "D": "ok"}], out,
    )
    html = out.read_text()
    assert "</script><script>alert" not in html
    assert "\\u003c/script\\u003e" in html


def test_dashboard_renders_model_text_as_text_not_markup(tmp_path):
    from src.dashboard import build_dashboard

    out = tmp_path / "d.html"
    build_dashboard({}, {}, {}, [{"task_id": "t", "A": "x", "D": "y"}], out)
    html = out.read_text()
    assert "div.innerHTML" not in html, "examples must not be built with innerHTML"
    assert "textContent" in html


def test_hostile_task_id_is_also_escaped(tmp_path):
    from src.dashboard import build_dashboard

    out = tmp_path / "d.html"
    build_dashboard({}, {}, {}, [{"task_id": HOSTILE, "A": "a", "D": "b"}], out)
    assert "</script><script>" not in out.read_text()
