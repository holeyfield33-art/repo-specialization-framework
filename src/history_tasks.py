"""
4–5. Historical training-data extraction & evidence-grounded task generation.

Primary split unit = change family. Temporal preference: train on earlier,
eval on later. No leakage of future state.

Task context must not contain the task's own answer. `context_files` carries
only the evidence a solver is given (the changed/seed files); the files it must
predict live in `ground_truth` alone. Conditions reach the answer through their
own context ladder (packs for B/C, the dependency graph for D) or not at all.
tests/test_task_integrity.py enforces this on every generated task.
Tasks prioritized:
  bug localization, root-cause, change-impact, patch generation,
  test-impact, cross-file dependency reasoning, code review,
  contract/invariant verification.
Synthetic only when mechanically validated from known-good history.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .ingestion import ChangeFamily, RepoManifest
from .graph import DependencyGraph
from .prompting import expected_answer_paths


@dataclass
class TaskExample:
    task_id: str
    task_type: str
    family_id: str
    commit_sha: str
    instruction: str
    context_files: List[str]  # paths only; packs/graph supplied at inference
    ground_truth: Dict[str, Any]
    evidence: List[str]  # provenance strings
    split: str = "train"  # train | val | eval
    difficulty: str = "medium"


#: Task types where supplying the answer set IS the task: bug localization
#: asks which of these candidates is the culprit, not which files exist. Every
#: other type must keep its answer out of the prompt.
CANDIDATE_SET_TASK_TYPES = {"bug_localization"}


def enforce_context_answer_disjoint(
    tasks: List["TaskExample"],
) -> Tuple[List["TaskExample"], Dict[str, Any]]:
    """Strip ground-truth paths out of every task's context_files.

    Applied to all tasks at generation time so the invariant holds by
    construction rather than per task type. A task whose context is entirely
    consumed by this had no evidence beyond its own answer and is dropped: an
    unanswerable task scored across four conditions is noise in every one.

    Returns the surviving tasks plus a machine-readable report.
    """
    kept: List["TaskExample"] = []
    report: Dict[str, Any] = {"stripped": [], "dropped": []}
    for t in tasks:
        if t.task_type in CANDIDATE_SET_TASK_TYPES:
            kept.append(t)
            continue
        answer = set(expected_answer_paths(t))
        if not answer:
            kept.append(t)
            continue
        removed = [f for f in t.context_files if f in answer]
        if removed:
            t.context_files = [f for f in t.context_files if f not in answer]
            report["stripped"].append({"task_id": t.task_id, "removed": removed})
        if not t.context_files:
            report["dropped"].append({"task_id": t.task_id, "reason": "no evidence left"})
            continue
        kept.append(t)
    return kept, report


TASK_TYPES = [
    "bug_localization",
    "root_cause_explanation",
    "change_impact_prediction",
    "patch_generation",
    "test_impact_prediction",
    "cross_file_dependency_reasoning",
    "code_review",
    "contract_invariant_verification",
]


def _family_temporal_split(
    families: List[ChangeFamily],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Dict[str, str]:
    """Assign each family wholly to one split. Temporal: earlier -> train.
    When few families exist (e.g. shallow sample), still reserve at least one
    family for eval by demoting the most recent train family.
    """
    sorted_f = sorted(families, key=lambda f: f.temporal_rank)
    n = len(sorted_f)
    if n == 0:
        return {}
    n_train = max(1, int(n * train_ratio))
    n_val = max(0, int(n * val_ratio))
    if n >= 2 and n_train + n_val >= n:
        n_train = max(1, n - 1 - max(0, n_val))
    assignment = {}
    for i, f in enumerate(sorted_f):
        if i < n_train:
            assignment[f.family_id] = "train"
        elif i < n_train + n_val:
            assignment[f.family_id] = "val"
        else:
            assignment[f.family_id] = "eval"
    if n == 1:
        assignment[sorted_f[0].family_id] = "train"
    return assignment


def generate_tasks_from_history(
    manifest: RepoManifest,
    graph: DependencyGraph,
    seed: int = 42,
) -> List[TaskExample]:
    random.seed(seed)
    assignment = _family_temporal_split(manifest.change_families)
    tasks: List[TaskExample] = []
    path_set = {f.path for f in manifest.files}
    family_by_id = {f.family_id: f for f in manifest.change_families}

    for fam in manifest.change_families:
        split = assignment.get(fam.family_id, "train")
        changed = [p for p in fam.files if p in path_set]
        if not changed:
            continue
        neigh = graph.impact_neighborhood(changed, radius=1, max_nodes=15)
        # sorted, not list(set(...)): this ordering becomes ground truth
        related = sorted(set(neigh["nodes"]) - set(changed))

        tasks.append(
            TaskExample(
                task_id=f"{fam.family_id}-impact",
                task_type="change_impact_prediction",
                family_id=fam.family_id,
                commit_sha=fam.root_commit,
                instruction=(
                    f"Given the following changed files in commit {fam.root_commit[:12]}, "
                    f"list the files most likely to be impacted (callers, callees, tests). "
                    f"Changed: {changed[:5]}. Use only repository structure evidence."
                ),
                # Only the CHANGED files. `related` is the answer: including it
                # here would hand every condition the ground truth and make the
                # A/B/C/D comparison meaningless. Condition D reaches `related`
                # through the dependency graph — that is D's whole advantage.
                context_files=changed,
                ground_truth={
                    "impacted": related[:10],
                    "seeds": changed,
                    "uncertainty": neigh.get("uncertainty", []),
                },
                evidence=[f"graph:{graph.version}", f"commit:{fam.root_commit}"],
                split=split,
            )
        )

        tests = []
        for p in changed:
            for fr in manifest.files:
                if fr.path == p:
                    tests.extend(fr.related_tests)
        tests = list(dict.fromkeys(tests))
        tasks.append(
            TaskExample(
                task_id=f"{fam.family_id}-test-impact",
                task_type="test_impact_prediction",
                family_id=fam.family_id,
                commit_sha=fam.root_commit,
                instruction=(
                    f"Which tests should be re-run after changes to {changed[:3]}? "
                    "Answer with paths only; justify from TEST edges."
                ),
                # Only the changed files; `tests` is the answer.
                context_files=changed,
                ground_truth={"tests_to_run": tests},
                evidence=[f"test-heuristic:{t}" for t in tests[:5]],
                split=split,
            )
        )

        if related:
            tasks.append(
                TaskExample(
                    task_id=f"{fam.family_id}-xfile",
                    task_type="cross_file_dependency_reasoning",
                    family_id=fam.family_id,
                    commit_sha=fam.root_commit,
                    instruction=(
                        f"Explain the dependency relationship between {changed[0]} and "
                        f"{related[0] if related else 'related modules'}. "
                        "Cite edge types (IMPORTS/CALLS/TESTS)."
                    ),
                    context_files=[changed[0]] + (related[:3] if related else []),
                    ground_truth={
                        "edges": [e for e in neigh["edges"] if e["src"] in changed or e["dst"] in changed][:10],
                    },
                    evidence=[f"graph-edge:{e}" for e in neigh["edges"][:3]],
                    split=split,
                )
            )

        msg = ""
        for c in manifest.commits:
            if c.sha == fam.root_commit:
                msg = c.message
                break
        tasks.append(
            TaskExample(
                task_id=f"{fam.family_id}-review",
                task_type="code_review",
                family_id=fam.family_id,
                commit_sha=fam.root_commit,
                instruction=(
                    f"Perform a code review of the change family rooted at {fam.root_commit[:12]}. "
                    f"Commit message: {msg[:200]}. List risks, missing tests, and invariant concerns."
                ),
                context_files=changed[:6],
                ground_truth={
                    "commit_message": msg,
                    "files": changed,
                    "suggested_checks": ["policy integrity", "self-hash", "detection coverage"],
                },
                evidence=[f"commit:{fam.root_commit}", f"message:{msg[:80]}"],
                split=split,
            )
        )

        if any(k in msg.lower() for k in ("fix", "bug", "crash", "bypass", "f-", "critical")):
            tasks.append(
                TaskExample(
                    task_id=f"{fam.family_id}-bugloc",
                    task_type="bug_localization",
                    family_id=fam.family_id,
                    commit_sha=fam.root_commit,
                    instruction=(
                        f"Localize the bug addressed by commit {fam.root_commit[:12]}. "
                        f"Message: {msg[:180]}. Name the primary file and symbol if possible."
                    ),
                    # Deliberate overlap: bug localization is a pick-from-candidates
                    # task, so the candidate set is the input, not a leak. The
                    # answer is which of them, not which files exist.
                    context_files=changed[:5],
                    ground_truth={"primary_files": changed[:2], "message": msg},
                    evidence=[f"commit:{fam.root_commit}"],
                    split=split,
                    difficulty="hard",
                )
            )

    for i, fr in enumerate(manifest.files):
        if fr.related_tests and fr.language == "javascript":
            split = "eval" if i % 4 == 0 else ("val" if i % 5 == 0 else "train")
            tasks.append(
                TaskExample(
                    task_id=f"synth-utgen-{fr.path.replace('/', '_')}",
                    task_type="unit_test_generation",
                    family_id="synthetic-validated",
                    commit_sha=manifest.head_sha or "head",
                    instruction=(
                        f"Generate a unit test outline for the module at {fr.path}. "
                        f"Focus on exported symbols: {fr.symbols[:5]}."
                    ),
                    context_files=[fr.path],
                    ground_truth={"existing_tests": fr.related_tests, "symbols": fr.symbols},
                    evidence=[f"existing-test:{t}" for t in fr.related_tests],
                    split=split,
                    difficulty="easy",
                )
            )

    if len(manifest.change_families) <= 2:
        js_files = [f.path for f in manifest.files if f.language == "javascript"]
        for i, path in enumerate(js_files[:6]):
            neigh = graph.impact_neighborhood([path], radius=1, max_nodes=10)
            related = sorted(n for n in neigh["nodes"] if n != path)
            tasks.append(
                TaskExample(
                    task_id=f"heldout-impact-{i}-{Path(path).stem}",
                    task_type="change_impact_prediction",
                    family_id=f"heldout-family-{i}",
                    commit_sha=f"heldout-{i}",
                    instruction=(
                        f"Predict impact neighborhood for a hypothetical change to {path}. "
                        "List likely callers, callees, and tests."
                    ),
                    # Only the seed file; `related` is the answer.
                    context_files=[path],
                    ground_truth={"impacted": related, "seeds": [path]},
                    evidence=[f"graph:{graph.version}"],
                    split="eval",
                    difficulty="medium",
                )
            )

    tasks, _ = enforce_context_answer_disjoint(tasks)
    return tasks


def write_splits(
    tasks: List[TaskExample],
    out_dir: Path,
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_split: Dict[str, List[TaskExample]] = {"train": [], "val": [], "eval": []}
    for t in tasks:
        by_split.setdefault(t.split, []).append(t)

    paths = {}
    for split, items in by_split.items():
        p = out_dir / f"{split}.jsonl"
        with open(p, "w", encoding="utf-8") as fh:
            for t in items:
                fh.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")
        paths[split] = p

    families_seen: Dict[str, Set[str]] = {}
    for t in tasks:
        families_seen.setdefault(t.family_id, set()).add(t.split)
    leakage = {fid: list(splits) for fid, splits in families_seen.items() if len(splits) > 1}
    meta = {
        "counts": {k: len(v) for k, v in by_split.items()},
        "family_split_integrity": "PASS" if not leakage else "FAIL",
        "leaking_families": leakage,
        "task_types": sorted({t.task_type for t in tasks}),
    }
    with open(out_dir / "split_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return paths
