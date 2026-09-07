"""
4–5. Historical training-data extraction & evidence-grounded task generation.

Primary split unit = change family. Temporal preference: train on earlier,
eval on later. No leakage of future state.
Tasks prioritized:
  bug localization, root-cause, change-impact, patch generation,
  test-impact, cross-file dependency reasoning, code review,
  contract/invariant verification.
Synthetic only when mechanically validated from known-good history.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .ingestion import ChangeFamily, RepoManifest
from .graph import DependencyGraph


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
    # Every task is evaluated against the repository state immediately before
    # its change family.  This prevents current-HEAD source from revealing a
    # held-out fix.
    context_commit: Optional[str] = None
    temporal_rank: Optional[int] = None


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


def family_temporal_split(
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
    assignment = family_temporal_split(manifest.change_families)
    tasks: List[TaskExample] = []
    commit_by_sha = {c.sha: c for c in manifest.commits}

    for fam in manifest.change_families:
        split = assignment.get(fam.family_id, "train")
        changed = [p for p in fam.files if Path(p).suffix.lower() in {
            ".js", ".mjs", ".cjs", ".ts", ".tsx", ".py", ".go", ".rs", ".java",
            ".md", ".json", ".yml", ".yaml",
        }]
        if not changed:
            continue
        commit = commit_by_sha.get(fam.root_commit)
        context_commit = commit.parents[0] if commit and commit.parents else None
        seeds = changed[:1]
        # Ground truth comes from the real historical change, not from the
        # current-HEAD graph.  The graph is context only.
        related = changed[1:]

        tasks.append(
            TaskExample(
                task_id=f"{fam.family_id}-impact",
                task_type="change_impact_prediction",
                family_id=fam.family_id,
                commit_sha=fam.root_commit,
                instruction=(
                    f"Given the following changed files in commit {fam.root_commit[:12]}, "
                    f"list the files most likely to be impacted (callers, callees, tests). "
                    f"Known starting file: {seeds}. Use only repository evidence."
                ),
                context_files=seeds,
                ground_truth={
                    "impacted": related[:10],
                    "seeds": seeds,
                },
                evidence=[f"graph:{graph.version}", f"commit:{fam.root_commit}"],
                split=split,
                context_commit=context_commit,
                temporal_rank=fam.temporal_rank,
            )
        )

        if context_commit:
            try:
                import git
                repo = git.Repo(manifest.root_path)
                historical_patch = repo.git.diff(
                    context_commit, fam.root_commit, "--", *changed[:3]
                )
            except Exception:
                historical_patch = ""
            if historical_patch:
                tasks.append(
                    TaskExample(
                        task_id=f"{fam.family_id}-patch",
                        task_type="patch_generation",
                        family_id=fam.family_id,
                        commit_sha=fam.root_commit,
                        instruction=(
                            f"Reproduce the change described by commit message {commit.message[:180]!r}. "
                            "Return a unified diff based only on the pre-change repository context."
                        ),
                        context_files=seeds,
                        ground_truth={
                            "files": changed[:3],
                            "seeds": seeds,
                            "patch": historical_patch,
                        },
                        evidence=[f"parent:{context_commit}", f"commit:{fam.root_commit}"],
                        split=split,
                        difficulty="hard",
                        context_commit=context_commit,
                        temporal_rank=fam.temporal_rank,
                    )
                )

        tests = []
        tests.extend(p for p in changed if "test" in p.lower() or "spec" in p.lower())
        tests = list(dict.fromkeys(tests))
        source_changes = [p for p in changed if p not in tests]
        tasks.append(
            TaskExample(
                task_id=f"{fam.family_id}-test-impact",
                task_type="test_impact_prediction",
                family_id=fam.family_id,
                commit_sha=fam.root_commit,
                instruction=(
                    f"Which tests should be re-run after changes to {source_changes[:3]}? "
                    "Answer with paths only; justify from TEST edges."
                ),
                context_files=source_changes[:3],
                ground_truth={"tests_to_run": tests},
                evidence=[f"test-heuristic:{t}" for t in tests[:5]],
                split=split,
                context_commit=context_commit,
                temporal_rank=fam.temporal_rank,
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
                        f"Identify and explain modules related to {changed[0]}. "
                        "Cite repository evidence and edge types when available."
                    ),
                    context_files=[changed[0]],
                    ground_truth={
                        "related_files": related[:10],
                    },
                    evidence=[f"commit-cochange:{fam.root_commit}"],
                    split=split,
                    context_commit=context_commit,
                    temporal_rank=fam.temporal_rank,
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
                context_files=seeds,
                ground_truth={
                    "commit_message": msg,
                    "files": changed,
                    "seeds": seeds,
                    "suggested_checks": ["policy integrity", "self-hash", "detection coverage"],
                },
                evidence=[f"commit:{fam.root_commit}", f"message:{msg[:80]}"],
                split=split,
                context_commit=context_commit,
                temporal_rank=fam.temporal_rank,
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
                    context_files=[],
                    ground_truth={"primary_files": changed[:2], "message": msg},
                    evidence=[f"commit:{fam.root_commit}"],
                    split=split,
                    difficulty="hard",
                    context_commit=context_commit,
                    temporal_rank=fam.temporal_rank,
                )
            )

    # Synthetic HEAD-derived examples are deliberately excluded.  They cannot
    # be proven temporally earlier than held-out real commits, and the experiment
    # contract requires every evaluation task to belong to a real change family.
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
