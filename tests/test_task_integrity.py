"""
Regression tests for task construction integrity.

These exist because an earlier version of history_tasks built
`context_files = changed + related[:8]` while the ground truth was
`related[:10]` — every condition was handed part of its own answer, so the
A/B/C/D comparison measured nothing. That class of bug is invisible in the
output (scores just look good), so it gets a test rather than a comment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import _answer_leaks_into_context
from src.history_tasks import CANDIDATE_SET_TASK_TYPES, enforce_context_answer_disjoint
from src.prompting import expected_answer_paths


def _leaking(tasks):
    return [
        t
        for t in tasks
        if t.task_type not in CANDIDATE_SET_TASK_TYPES and _answer_leaks_into_context(t)
    ]


def test_no_generated_task_leaks_its_answer(pipeline):
    """THE regression test: no task may carry ground-truth paths in context."""
    leaks = _leaking(pipeline["tasks"])
    assert not leaks, "tasks leaking their own answer into context_files: " + ", ".join(
        f"{t.task_id} ({t.task_type}): "
        f"{sorted(set(expected_answer_paths(t)) & set(t.context_files))}"
        for t in leaks
    )


def test_eval_split_specifically_is_clean(pipeline):
    """The eval split is what produces the reported numbers, so assert it
    separately — a leak here inflates every published figure."""
    eval_tasks = [t for t in pipeline["tasks"] if t.split == "eval"]
    assert eval_tasks, "fixture should produce eval-split tasks"
    assert not _leaking(eval_tasks)


def test_change_impact_context_is_seeds_only(pipeline):
    impact = [t for t in pipeline["tasks"] if t.task_type == "change_impact_prediction"]
    assert impact, "fixture should produce change-impact tasks"
    for t in impact:
        seeds = set(t.ground_truth.get("seeds", []))
        assert set(t.context_files) <= seeds, (
            f"{t.task_id} context_files must be the seed/changed files only, "
            f"got extras: {sorted(set(t.context_files) - seeds)}"
        )


def test_test_impact_context_excludes_the_tests(pipeline):
    for t in pipeline["tasks"]:
        if t.task_type != "test_impact_prediction":
            continue
        tests = set(t.ground_truth.get("tests_to_run", []))
        assert not (tests & set(t.context_files)), (
            f"{t.task_id} hands the model the tests it is asked to predict"
        )


def test_leak_detector_actually_detects(pipeline):
    """Guard the guard: if _answer_leaks_into_context silently stopped
    working, every test above would pass vacuously."""
    task = next(t for t in pipeline["tasks"] if expected_answer_paths(t))
    assert not _answer_leaks_into_context(task)
    task.context_files = list(task.context_files) + [expected_answer_paths(task)[0]]
    assert _answer_leaks_into_context(task)


def test_family_split_integrity(pipeline):
    """A change family must live wholly in one split — the framework's
    contamination claim depends on it."""
    by_family = {}
    for t in pipeline["tasks"]:
        by_family.setdefault(t.family_id, set()).add(t.split)
    straddling = {f: s for f, s in by_family.items() if len(s) > 1}
    # Synthetic tasks are intentionally spread across splits by index.
    straddling = {f: s for f, s in straddling.items() if not f.startswith("synthetic")}
    assert not straddling, f"families spanning splits: {straddling}"


def test_ground_truth_paths_exist_in_repo(pipeline):
    """An answer that names a file the repo does not contain is unscoreable."""
    real_paths = {f.path for f in pipeline["manifest"].files}
    for t in pipeline["tasks"]:
        for path in expected_answer_paths(t):
            assert path in real_paths, f"{t.task_id} expects nonexistent {path}"


# --- the enforcement helper itself -------------------------------------------

def _task(task_id, task_type, context_files, ground_truth):
    from src.history_tasks import TaskExample

    return TaskExample(
        task_id=task_id,
        task_type=task_type,
        family_id="fam",
        commit_sha="sha",
        instruction="x",
        context_files=context_files,
        ground_truth=ground_truth,
        evidence=[],
        split="eval",
    )


def test_enforcement_strips_answer_paths():
    t = _task("t1", "change_impact_prediction", ["a.js", "b.js"], {"impacted": ["b.js"]})
    kept, report = enforce_context_answer_disjoint([t])
    assert kept[0].context_files == ["a.js"]
    assert report["stripped"] == [{"task_id": "t1", "removed": ["b.js"]}]


def test_enforcement_drops_tasks_left_without_evidence():
    t = _task("t2", "change_impact_prediction", ["b.js"], {"impacted": ["b.js"]})
    kept, report = enforce_context_answer_disjoint([t])
    assert kept == []
    assert report["dropped"] == [{"task_id": "t2", "reason": "no evidence left"}]


def test_enforcement_leaves_candidate_set_tasks_alone():
    t = _task("t3", "bug_localization", ["a.js", "b.js"], {"primary_files": ["b.js"]})
    kept, report = enforce_context_answer_disjoint([t])
    assert kept[0].context_files == ["a.js", "b.js"]
    assert report["stripped"] == []


def test_enforcement_is_idempotent(pipeline):
    once, _ = enforce_context_answer_disjoint(list(pipeline["tasks"]))
    twice, report = enforce_context_answer_disjoint(once)
    assert len(once) == len(twice)
    assert report["stripped"] == [] and report["dropped"] == []


def test_bug_localization_still_exists_after_enforcement(pipeline):
    """The candidate-set carve-out must not be a loophole that silently
    deletes a whole task type."""
    types = {t.task_type for t in pipeline["tasks"]}
    assert "change_impact_prediction" in types
    assert "test_impact_prediction" in types


# --- determinism -------------------------------------------------------------

def test_task_generation_is_deterministic_across_processes(sample_repo):
    """The framework's premise is deterministic, SHA-bound artifacts.

    `list(some_set_of_strings)` iterates in an order that varies per process
    with PYTHONHASHSEED, so identical inputs produced different ground truth,
    different file packs and different SHAs on every run. This runs generation
    in separate subprocesses precisely because a same-process loop would share
    one hash seed and pass regardless.
    """
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        f"""
        import sys, json, hashlib
        sys.path.insert(0, {str(ROOT)!r})
        from pathlib import Path
        from dataclasses import asdict
        from src.ingestion import ingest_repository
        from src.graph import build_dependency_graph
        from src.history_tasks import generate_tasks_from_history

        m = ingest_repository(Path({str(sample_repo)!r}), repo_name="r", include_content=True)
        g = build_dependency_graph(m)
        tasks = generate_tasks_from_history(m, g)
        blob = "".join(json.dumps(asdict(t), sort_keys=True) for t in tasks)
        print(hashlib.sha256(blob.encode()).hexdigest())
        """
    )
    digests = set()
    for seed in ("0", "1", "12345"):
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
        )
        assert proc.returncode == 0, proc.stderr
        digests.add(proc.stdout.strip())
    assert len(digests) == 1, (
        f"task generation is not reproducible across hash seeds: {digests}"
    )


def test_impact_neighborhood_returns_sorted_collections(pipeline):
    graph = pipeline["graph"]
    seeds = [f.path for f in pipeline["manifest"].files][:2]
    neigh = graph.impact_neighborhood(seeds, radius=1, max_nodes=25)
    assert neigh["nodes"] == sorted(neigh["nodes"])
    assert neigh["seeds"] == sorted(neigh["seeds"])
    edge_keys = [(e["src"], e["dst"], e["type"]) for e in neigh["edges"]]
    assert edge_keys == sorted(edge_keys)
