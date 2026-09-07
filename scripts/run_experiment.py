#!/usr/bin/env python3
"""Run a fail-closed, real RSEF specialization experiment."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dashboard import build_dashboard, write_streamlit_app
from src.evaluation import aggregate_metrics, contamination_audit, run_condition_group
from src.file_packs import generate_file_packs
from src.graph import build_dependency_graph
from src.history_tasks import (
    TaskExample,
    family_temporal_split,
    generate_tasks_from_history,
    write_splits,
)
from src.ingestion import ingest_repository, save_manifest
from src.ingestion import compute_sha256
from src.model_runtime import require_cuda, validate_compute_dtype
from src.qlora_config import QLoRAHyperParams


def _read_tasks(path: Path) -> list[TaskExample]:
    return [TaskExample(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def _git_head(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def _write_dataset_lock(data_dir: Path, head_sha: str) -> None:
    relative_paths = [
        "repository_manifest.json", "dependency_graph.json",
        "splits/train.jsonl", "splits/val.jsonl", "splits/eval.jsonl",
        "splits/split_manifest.json", "contamination_report.json", "pretraining_report.json",
    ]
    lock = {
        "target_head": head_sha,
        "sha256": {rel: compute_sha256((data_dir / rel).read_bytes()) for rel in relative_paths},
    }
    (data_dir / "dataset_lock.json").write_text(json.dumps(lock, indent=2))


def _verify_dataset_lock(data_dir: Path) -> dict:
    lock = json.loads((data_dir / "dataset_lock.json").read_text())
    mismatches = []
    for rel, expected in lock.get("sha256", {}).items():
        path = data_dir / rel
        actual = compute_sha256(path.read_bytes()) if path.is_file() else None
        if actual != expected:
            mismatches.append({"path": rel, "expected": expected, "actual": actual})
    if mismatches:
        raise SystemExit("prepared dataset lock failed: " + json.dumps(mismatches[:5]))
    return lock


def prepare(repo: Path, data_dir: Path, packs_dir: Path) -> tuple[list[TaskExample], dict]:
    print("=== PRE-TRAINING: full-history ingestion ===", flush=True)
    manifest = ingest_repository(repo, repo_name="holeyfield33-art/runtime-firewall-mvp", include_content=True)
    save_manifest(manifest, data_dir / "repository_manifest.json")
    graph = build_dependency_graph(manifest)
    graph.save(data_dir / "dependency_graph.json")
    generate_file_packs(manifest, graph, packs_dir)
    tasks = generate_tasks_from_history(manifest, graph)
    write_splits(tasks, data_dir / "splits")
    train = [t for t in tasks if t.split == "train"]
    val = [t for t in tasks if t.split == "val"]
    evaluation = [t for t in tasks if t.split == "eval"]
    family_assignments = family_temporal_split(manifest.change_families)
    report = contamination_audit(
        train, evaluation, {t.family_id for t in train}, {t.family_id for t in evaluation},
    )
    (data_dir / "contamination_report.json").write_text(json.dumps(report, indent=2))
    counts = {
        "source_file_count": len(manifest.files),
        "graph_edge_count": len(graph.edges),
        "independent_change_family_count": len(manifest.change_families),
        "train_family_count": sum(s == "train" for s in family_assignments.values()),
        "val_family_count": sum(s == "val" for s in family_assignments.values()),
        "eval_family_count": sum(s == "eval" for s in family_assignments.values()),
        "train_task_count": len(train),
        "val_task_count": len(val),
        "eval_task_count": len(evaluation),
        "contamination_audit": report["status"],
    }
    (data_dir / "pretraining_report.json").write_text(json.dumps(counts, indent=2))
    _write_dataset_lock(data_dir, manifest.head_sha or "")
    print(json.dumps(counts, indent=2), flush=True)
    if report["status"] != "PASS":
        raise SystemExit("contamination audit failed; training is blocked")
    return tasks, counts


def load_prepared(repo: Path, prepared_data: Path) -> tuple[list[TaskExample], dict]:
    lock = _verify_dataset_lock(prepared_data)
    manifest = json.loads((prepared_data / "repository_manifest.json").read_text())
    if manifest["head_sha"] != _git_head(repo) or lock.get("target_head") != manifest["head_sha"]:
        raise SystemExit("target HEAD differs from frozen prepared dataset")
    tasks = []
    for split in ("train", "val", "eval"):
        tasks.extend(_read_tasks(prepared_data / "splits" / f"{split}.jsonl"))
    report = json.loads((prepared_data / "contamination_report.json").read_text())
    if report.get("status") != "PASS":
        raise SystemExit("prepared contamination report is not PASS")
    return tasks, json.loads((prepared_data / "pretraining_report.json").read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--model", choices=["qwen", "smol"], default="qwen")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "qwen")
    parser.add_argument("--prepared-data", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--test-command", default="npm test")
    parser.add_argument("--compute-dtype", choices=["bfloat16", "float16"], default="bfloat16",
                        help="Explicit precision for inference and training; use float16 on T4")
    args = parser.parse_args()

    repo = args.repo.resolve()
    if not (repo / ".git").exists():
        raise SystemExit(f"full Git clone required: {repo}")
    out = args.out.resolve()
    data_dir = out / "data"
    packs_dir = out / "file_packs"
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.prepared_data:
        tasks, counts = load_prepared(repo, args.prepared_data.resolve())
        if data_dir != args.prepared_data.resolve():
            shutil.copytree(args.prepared_data.resolve(), data_dir, dirs_exist_ok=True)
    else:
        tasks, counts = prepare(repo, data_dir, packs_dir)
    if args.prepare_only:
        print("Preparation complete; no training or inference was run.")
        return

    hardware = require_cuda()
    import torch
    validate_compute_dtype(args.compute_dtype, torch.cuda.is_bf16_supported())
    hardware["compute_dtype"] = args.compute_dtype
    (out / "hardware.json").write_text(json.dumps(hardware, indent=2))
    print("=== GPU/CUDA ===", flush=True)
    print(json.dumps(hardware, indent=2), flush=True)

    hp = QLoRAHyperParams.for_model(args.model)
    hp.bnb_4bit_compute_dtype = args.compute_dtype
    hp.output_dir = str(out / "adapters" / f"{args.model}-repo-qlora")
    hp.save(out / "qlora_config.yaml")
    adapter = Path(hp.output_dir)
    eval_tasks = [t for t in tasks if t.split == "eval"]
    if any(t.task_type == "patch_generation" for t in eval_tasks) and not (repo / "node_modules").is_dir():
        raise SystemExit("target node_modules missing; run 'npm ci --ignore-scripts' before patch evaluation")
    if hp.bnb_4bit_compute_dtype == "bfloat16":
        import torch
        if not torch.cuda.is_bf16_supported():
            raise SystemExit(
                "the frozen bfloat16 compute setting is unsupported by this GPU; "
                "stopping before changing hyperparameters"
            )

    print("=== REAL CONDITIONS A/B (UNTUNED) ===", flush=True)
    results = run_condition_group(
        eval_tasks, ["A", "B"], repo, hp.model_name_or_path,
        max_input_tokens=hp.max_seq_length, test_command=args.test_command,
        checkpoint_dir=out / "task_checkpoints",
        compute_dtype=args.compute_dtype,
    )
    (out / "task_results_ab.json").write_text(json.dumps(
        {condition: [asdict(row) for row in rows] for condition, rows in results.items()}, indent=2,
    ))

    print("=== REAL QLoRA TRAINING ===", flush=True)
    subprocess.run([
        sys.executable, str(ROOT / "scripts" / "train_qlora.py"),
        "--config", str(out / "qlora_config.yaml"),
        "--train-jsonl", str(data_dir / "splits" / "train.jsonl"),
        "--val-jsonl", str(data_dir / "splits" / "val.jsonl"),
    ], check=True)
    adapter_config = adapter / "adapter_config.json"
    adapter_weights = adapter / "adapter_model.safetensors"
    if not adapter_config.is_file() or not adapter_weights.is_file():
        raise SystemExit("adapter training did not produce real safetensors weights/config")

    print("=== REAL CONDITIONS C/D (TUNED) ===", flush=True)
    results.update(run_condition_group(
        eval_tasks, ["C", "D"], repo, hp.model_name_or_path, adapter,
        max_input_tokens=hp.max_seq_length, test_command=args.test_command,
        checkpoint_dir=out / "task_checkpoints",
        compute_dtype=args.compute_dtype,
    ))
    if set(results) != {"A", "B", "C", "D"} or any(not rows for rows in results.values()):
        raise SystemExit("one or more conditions produced no real task results")
    raw = {condition: [asdict(row) for row in rows] for condition, rows in results.items()}
    (out / "task_results_abcd.json").write_text(json.dumps(raw, indent=2))
    summary = aggregate_metrics(results)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))

    contamination = json.loads((data_dir / "contamination_report.json").read_text())
    gate = {
        "status": "PASS",
        "real_inference": True,
        "adapter_config_present": adapter_config.is_file(),
        "adapter_weights_present": adapter_weights.is_file(),
        "conditions_complete": True,
        "contamination_status": contamination["status"],
    }
    (out / "verification_gate.json").write_text(json.dumps(gate, indent=2))
    examples = []
    by_a = {r.task_id: r for r in results["A"]}
    by_d = {r.task_id: r for r in results["D"]}
    for task_id in list(by_a)[:4]:
        examples.append({"task_id": task_id, "A": by_a[task_id].raw_output, "D": by_d[task_id].raw_output})
    dashboard = build_dashboard(summary, contamination, gate, examples, out / "dashboard.html")
    write_streamlit_app(out / "streamlit_app.py")
    report = out / "report.md"
    report.write_text(
        "# RSEF measured experiment report\n\n"
        f"- Target HEAD: `{_git_head(repo)}`\n"
        f"- Model: `{hp.model_name_or_path}`\n"
        f"- GPU: `{hardware['gpu_model']}` ({hardware['vram_gib']} GiB)\n"
        f"- CUDA: `{hardware['cuda_version']}`\n"
        f"- Contamination audit: **{contamination['status']}**\n"
        f"- Verification gate: **{gate['status']}**\n\n"
        "## Pretraining counts\n\n```json\n" + json.dumps(counts, indent=2) +
        "\n```\n\n## A/B/C/D metrics\n\n```json\n" + json.dumps(summary, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    index = {
        "target_repo": "holeyfield33-art/runtime-firewall-mvp",
        "target_head": _git_head(repo),
        "model": hp.model_name_or_path,
        "hardware": hardware,
        "pretraining": counts,
        "artifacts": {
            "metrics": str(out / "metrics.json"),
            "contamination_report": str(data_dir / "contamination_report.json"),
            "adapter_config": str(adapter_config),
            "adapter_weights": str(adapter_weights),
            "task_results": str(out / "task_results_abcd.json"),
            "dashboard": str(dashboard),
            "report": str(report),
        },
        "verification_gate": gate,
    }
    (out / "experiment_index.json").write_text(json.dumps(index, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
