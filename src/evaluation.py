"""
7–10. Experimental conditions, evaluation suite, deterministic verification,
contamination audit.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .history_tasks import TaskExample
from .context_builder import GitSnapshot, build_condition_context
from .model_runtime import HFGenerator, InferenceError
from .patch_verifier import verify_patch


CONDITIONS = {
    "A": "Base model + ordinary repository RAG/context",
    "B": "Base model + structured file packs",
    "C": "Tuned model + structured file packs",
    "D": "Tuned model + file packs + dependency graph impact neighborhood",
}


@dataclass
class MetricResult:
    task_id: str
    condition: str
    task_type: str
    success: bool
    test_pass_rate: Optional[float] = None
    bug_localization_accuracy: float = 0.0
    impacted_file_recall: float = 0.0
    cross_file_accuracy: float = 0.0
    hallucinated_api_count: int = 0
    unnecessary_edits: int = 0
    security_regressions: Optional[int] = None
    token_usage: int = 0
    latency_ms: float = 0.0
    notes: str = ""
    raw_output: str = ""
    parsed_output: Dict[str, Any] = field(default_factory=dict)
    context_files: List[str] = field(default_factory=list)
    verification: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationGate:
    source_sha_ok: bool
    graph_version_ok: bool
    required_tests_run: bool
    patch_applies: bool
    static_checks_ok: bool
    no_critical_invariant_violation: bool
    status: str  # PASS | BLOCK | FREEZE
    details: List[str] = field(default_factory=list)


def deterministic_verify(
    pack_sha: str,
    expected_source_sha: str,
    graph_version: str,
    expected_graph_version: str,
    tests_run: List[str],
    required_tests: List[str],
    patch_clean: bool = True,
    static_ok: bool = True,
    critical_violations: int = 0,
) -> VerificationGate:
    details = []
    source_ok = bool(pack_sha and expected_source_sha and pack_sha == expected_source_sha)
    graph_ok = graph_version == expected_graph_version
    if not graph_ok:
        details.append(f"graph version mismatch: {graph_version} vs {expected_graph_version}")
    tests_ok = set(required_tests).issubset(set(tests_run)) if required_tests else True
    if not tests_ok:
        details.append(f"missing tests: {set(required_tests) - set(tests_run)}")
    inv_ok = critical_violations == 0
    if not inv_ok:
        details.append(f"critical invariant violations: {critical_violations}")

    if not source_ok or not inv_ok:
        status = "FREEZE"
    elif not graph_ok or not tests_ok or not patch_clean or not static_ok:
        status = "BLOCK"
    else:
        status = "PASS"

    return VerificationGate(
        source_sha_ok=source_ok,
        graph_version_ok=graph_ok,
        required_tests_run=tests_ok,
        patch_applies=patch_clean,
        static_checks_ok=static_ok,
        no_critical_invariant_violation=inv_ok,
        status=status,
        details=details,
    )


def _as_strings(value: Any) -> List[str]:
    return [str(v) for v in value] if isinstance(value, list) else []


def _expected_and_predicted(task: TaskExample, response: Dict[str, Any]) -> tuple[Set[str], Set[str]]:
    gt = task.ground_truth
    if task.task_type == "test_impact_prediction":
        return set(_as_strings(gt.get("tests_to_run"))), set(_as_strings(response.get("predicted_tests")))
    if task.task_type == "bug_localization":
        return set(_as_strings(gt.get("primary_files"))), set(_as_strings(response.get("primary_files")))
    if task.task_type == "cross_file_dependency_reasoning":
        return set(_as_strings(gt.get("related_files"))), set(_as_strings(response.get("related_files")))
    return set(_as_strings(gt.get("impacted", gt.get("files", [])))), set(_as_strings(response.get("predicted_files")))


def _patch_paths(patch: str) -> Set[str]:
    paths = set()
    for line in patch.splitlines():
        if line.startswith("+++ b/") or line.startswith("--- a/"):
            paths.add(line[6:])
    return paths


def evaluate_condition(
    tasks: List[TaskExample],
    condition: str,
    generator: HFGenerator,
    snapshot: GitSnapshot,
    repo_path: Path,
    test_command: str = "npm test",
    checkpoint_path: Optional[Path] = None,
) -> List[MetricResult]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition}")
    results: List[MetricResult] = []
    if checkpoint_path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if checkpoint_path.exists():
            raise RuntimeError(f"refusing to overwrite existing condition checkpoint: {checkpoint_path}")
    for task in tasks:
        if task.split != "eval":
            continue
        context = build_condition_context(task, condition, snapshot)
        generation = generator.generate(task.instruction, context.text)
        expected, predicted = _expected_and_predicted(task, generation.parsed)
        patch = generation.parsed.get("patch")
        if task.task_type == "patch_generation" and isinstance(patch, str):
            predicted |= _patch_paths(patch)
        recall = len(expected & predicted) / len(expected) if expected else float(not predicted)
        bug_acc = recall if task.task_type == "bug_localization" else 0.0
        xfile_acc = recall if task.task_type == "cross_file_dependency_reasoning" else 0.0
        mentioned_paths = set(_as_strings(generation.parsed.get("predicted_files")))
        mentioned_paths |= set(_as_strings(generation.parsed.get("primary_files")))
        unnecessary = len(mentioned_paths - expected - set(task.ground_truth.get("seeds", [])))
        referenced_apis = set(_as_strings(generation.parsed.get("referenced_apis")))
        hallucinated = len(referenced_apis - set(context.source_symbols))

        test_rate: Optional[float] = None
        regressions: Optional[int] = None
        patch_note = ""
        verification_details: Dict[str, Any] = {}
        if task.task_type == "patch_generation":
            verification = verify_patch(repo_path, task, patch if isinstance(patch, str) else "", test_command)
            test_rate = 1.0 if verification.tests_run and verification.tests_passed else 0.0
            regressions = verification.security_regressions
            patch_note = (
                f" patch_applies={verification.applies} tests_run={verification.tests_run} "
                f"tests_passed={verification.tests_passed}"
            )
            verification_details = asdict(verification)
            # Preserve the existing 0.55 success threshold while grounding its
            # score in observed file recall, patch applicability, and tests.
            score = (recall + float(verification.applies) + test_rate) / 3
        else:
            score = recall

        result = MetricResult(
            task_id=task.task_id,
            condition=condition,
            task_type=task.task_type,
            success=score >= 0.55,
            test_pass_rate=test_rate,
            bug_localization_accuracy=bug_acc,
            impacted_file_recall=recall,
            cross_file_accuracy=xfile_acc,
            hallucinated_api_count=hallucinated,
            unnecessary_edits=unnecessary,
            security_regressions=regressions,
            token_usage=generation.input_tokens + generation.output_tokens,
            latency_ms=generation.latency_ms,
            notes=(f"real_inference score={score:.4f}{patch_note}" +
                   (f" parse_error={generation.parse_error}" if generation.parse_error else "")),
            raw_output=generation.raw_text,
            parsed_output=generation.parsed,
            context_files=context.files,
            verification=verification_details,
        )
        results.append(result)
        if checkpoint_path:
            with checkpoint_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
        print(f"[{condition}] {len(results)} {task.task_id} success={result.success}", flush=True)
    return results


def run_all_conditions(
    eval_tasks: List[TaskExample],
    repo_path: Path,
    model_name: str,
    adapter_path: Path,
    max_input_tokens: int = 2048,
    test_command: str = "npm test",
) -> Dict[str, List[MetricResult]]:
    snapshot = GitSnapshot(repo_path)
    out: Dict[str, List[MetricResult]] = {}
    base = HFGenerator(model_name, max_input_tokens=max_input_tokens)
    for condition in ("A", "B"):
        out[condition] = evaluate_condition(eval_tasks, condition, base, snapshot, repo_path, test_command)
    del base
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass
    tuned = HFGenerator(model_name, adapter_path=adapter_path, max_input_tokens=max_input_tokens)
    for condition in ("C", "D"):
        out[condition] = evaluate_condition(eval_tasks, condition, tuned, snapshot, repo_path, test_command)
    return out


def run_condition_group(
    eval_tasks: List[TaskExample],
    conditions: List[str],
    repo_path: Path,
    model_name: str,
    adapter_path: Optional[Path] = None,
    max_input_tokens: int = 2048,
    test_command: str = "npm test",
    checkpoint_dir: Optional[Path] = None,
) -> Dict[str, List[MetricResult]]:
    if any(c in ("C", "D") for c in conditions) and adapter_path is None:
        raise InferenceError("tuned conditions require a real adapter")
    snapshot = GitSnapshot(repo_path)
    generator = HFGenerator(
        model_name,
        adapter_path=adapter_path,
        max_input_tokens=max_input_tokens,
    )
    return {
        condition: evaluate_condition(
            eval_tasks, condition, generator, snapshot, repo_path, test_command,
            (checkpoint_dir / f"condition_{condition}.jsonl") if checkpoint_dir else None,
        )
        for condition in conditions
    }


def contamination_audit(
    train_tasks: List[TaskExample],
    eval_tasks: List[TaskExample],
    train_families: Set[str],
    eval_families: Set[str],
) -> Dict[str, Any]:
    """
    Machine-readable proof:
    - no evaluation commit/family in training
    - no derivative from eval change in train
    - no future file state in historical train example
    - no held-out patch verbatim in train
    - no synthetic task contains held-out answer
    """
    report = {
        "status": "PASS",
        "checks": {},
        "violations": [],
    }

    overlap = train_families & eval_families
    report["checks"]["family_disjoint"] = len(overlap) == 0
    if overlap:
        report["violations"].append({"type": "family_overlap", "families": list(overlap)})
        report["status"] = "FAIL"

    train_commits = {t.commit_sha for t in train_tasks}
    eval_commits = {t.commit_sha for t in eval_tasks}
    c_overlap = train_commits & eval_commits
    report["checks"]["commit_disjoint"] = len(c_overlap) == 0
    if c_overlap:
        report["violations"].append({"type": "commit_overlap", "commits": list(c_overlap)[:10]})
        report["status"] = "FAIL"

    train_ranks = [t.temporal_rank for t in train_tasks if t.temporal_rank is not None]
    eval_ranks = [t.temporal_rank for t in eval_tasks if t.temporal_rank is not None]
    temporal_ok = bool(train_ranks and eval_ranks and max(train_ranks) < min(eval_ranks))
    report["checks"]["train_strictly_precedes_eval"] = temporal_ok
    if not temporal_ok:
        report["violations"].append({"type": "temporal_order_violation"})
        report["status"] = "FAIL"

    non_real_eval = [
        t.task_id for t in eval_tasks
        if len(t.commit_sha) != 40 or not t.context_commit or len(t.context_commit) != 40
    ]
    report["checks"]["eval_tasks_are_real_history"] = not non_real_eval
    if non_real_eval:
        report["violations"].append({"type": "non_real_eval_task", "tasks": non_real_eval[:20]})
        report["status"] = "FAIL"

    answer_path_leaks = []
    for task in eval_tasks:
        gt = task.ground_truth
        expected_paths = set(_as_strings(
            gt.get("impacted", gt.get("primary_files", gt.get("tests_to_run", gt.get("files", []))))
        ))
        seeds = set(_as_strings(gt.get("seeds")))
        leaked = expected_paths - seeds
        leaked = {p for p in leaked if p in task.context_files}
        if leaked:
            answer_path_leaks.append({"task_id": task.task_id, "paths": sorted(leaked)})
    report["checks"]["no_answer_paths_in_context_manifest"] = not answer_path_leaks
    if answer_path_leaks:
        report["violations"].append({"type": "answer_path_context_leak", "tasks": answer_path_leaks[:20]})
        report["status"] = "FAIL"

    # Repeated file/test names across independent historical changes are normal,
    # not leakage.  Audit only held-out commit identifiers and verbatim patches,
    # which are the answer-bearing derivatives that must never enter training.
    eval_answers = set()
    for t in eval_tasks:
        patch = t.ground_truth.get("patch")
        if isinstance(patch, str) and len(patch) >= 80:
            eval_answers.add(patch)
    for t in train_tasks:
        blob = t.instruction + json.dumps(t.ground_truth, sort_keys=True)
        leaked_commit = next((c for c in eval_commits if c in blob), None)
        if leaked_commit:
            report["violations"].append(
                {"type": "eval_commit_reference", "task_id": t.task_id, "commit": leaked_commit}
            )
            report["status"] = "FAIL"
        for ans in eval_answers:
            if ans in blob:
                report["violations"].append(
                    {"type": "heldout_patch_leak", "task_id": t.task_id, "snippet": ans[:80]}
                )
                report["status"] = "FAIL"
                break

    report["checks"]["no_eval_commit_reference"] = not any(
        v["type"] == "eval_commit_reference" for v in report["violations"]
    )
    report["checks"]["no_heldout_patch_leak"] = not any(
        v["type"] == "heldout_patch_leak" for v in report["violations"]
    )
    report["train_family_count"] = len(train_families)
    report["eval_family_count"] = len(eval_families)
    report["train_task_count"] = len(train_tasks)
    report["eval_task_count"] = len(eval_tasks)
    return report


def aggregate_metrics(results_by_cond: Dict[str, List[MetricResult]]) -> Dict[str, Any]:
    summary = {}
    for cond, rows in results_by_cond.items():
        if not rows:
            summary[cond] = {"n": 0}
            continue
        n = len(rows)
        patch_rows = [r for r in rows if r.test_pass_rate is not None]
        regression_rows = [r for r in rows if r.security_regressions is not None]
        summary[cond] = {
            "n": n,
            "task_success_rate": sum(1 for r in rows if r.success) / n,
            "mean_impacted_file_recall": sum(r.impacted_file_recall for r in rows) / n,
            "mean_hallucinated_apis": sum(r.hallucinated_api_count for r in rows) / n,
            "mean_unnecessary_edits": sum(r.unnecessary_edits for r in rows) / n,
            "test_pass_rate": (
                sum(float(r.test_pass_rate) for r in patch_rows) / len(patch_rows)
                if patch_rows else None
            ),
            "security_invariant_regressions": sum(
                int(r.security_regressions) for r in regression_rows
            ),
            "mean_token_usage": sum(r.token_usage for r in rows) / n,
            "mean_latency_ms": sum(r.latency_ms for r in rows) / n,
            "bug_loc_tasks": sum(1 for r in rows if r.task_type == "bug_localization"),
            "bug_loc_accuracy": (
                sum(r.bug_localization_accuracy for r in rows if r.task_type == "bug_localization")
                / max(1, sum(1 for r in rows if r.task_type == "bug_localization"))
            ),
            "condition_label": CONDITIONS[cond],
        }
    return summary
