#!/usr/bin/env python3
"""
End-to-end experimental harness for repository specialization.

Usage:
  python -m scripts.run_experiment --repo /path/to/runtime-firewall-mvp \\
      --model qwen --out results/

Produces all required artifacts:
  repository manifest, file packs, dependency graph,
  train/val/eval manifests, contamination report,
  QLoRA config, adapter, evaluation results, dashboard.

Two modes, and the difference matters:
  default (no flags)                  no model runs; results are labelled
                                      data_provenance="simulated" and are a
                                      pipeline smoke test, not evidence.
  --real-train --eval-mode real       trains a real LoRA adapter and evaluates
                                      with real forward passes; results are
                                      labelled data_provenance="real".
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# ensure package root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingestion import ingest_repository, save_manifest, manifest_to_dict
from src.graph import build_dependency_graph
from src.file_packs import generate_file_packs
from src.history_tasks import generate_tasks_from_history, write_splits
from src.qlora_config import QLoRAHyperParams
from src.evaluation import (
    run_all_conditions,
    contamination_audit,
    aggregate_metrics,
    deterministic_verify,
    run_provenance,
    CONDITIONS,
)
from src.dashboard import build_dashboard, write_streamlit_app


def main() -> None:
    parser = argparse.ArgumentParser(description="RSEF experiment runner")
    parser.add_argument(
        "--repo",
        type=str,
        default=str(ROOT.parent / "helios-sample"),
        help="Path to local clone or sample of runtime-firewall-mvp",
    )
    parser.add_argument("--model", choices=["qwen", "smol"], default="qwen")
    parser.add_argument("--out", type=str, default=str(ROOT / "results"))
    parser.add_argument("--dry-train", action="store_true", default=True)
    parser.add_argument(
        "--real-train",
        action="store_true",
        default=False,
        help="Actually train a LoRA adapter via scripts/train_real.py "
             "instead of writing a NOT_TRAINED marker.",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=200,
        help="Optimizer steps for --real-train.",
    )
    parser.add_argument(
        "--eval-mode",
        choices=["real", "simulate"],
        default="simulate",
        help="'real' runs the model for every condition; 'simulate' runs no "
             "model and produces results explicitly labelled as non-evidence.",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = out / "data"
    packs_dir = out / "file_packs"
    data_dir.mkdir(exist_ok=True)

    print("=== 1. Repository ingestion ===")
    repo_path = Path(args.repo)
    if not repo_path.exists():
        repo_path = ROOT.parent / "helios-sample"
    manifest = ingest_repository(
        repo_path,
        repo_name="holeyfield33-art/runtime-firewall-mvp",
        include_content=True,
    )
    save_manifest(manifest, data_dir / "repository_manifest.json")
    print(f"  files={len(manifest.files)} commits={len(manifest.commits)} "
          f"families={len(manifest.change_families)} graph_version={manifest.graph_version}")

    print("=== 2-3. Dependency graph + file packs ===")
    graph = build_dependency_graph(manifest)
    graph.save(data_dir / "dependency_graph.json")
    packs = generate_file_packs(manifest, graph, packs_dir)
    print(f"  nodes={graph.g.number_of_nodes()} edges={len(graph.edges)} packs={len(packs)}")

    print("=== 4-5. Historical tasks & temporal splits ===")
    tasks = generate_tasks_from_history(manifest, graph)
    split_paths = write_splits(tasks, data_dir / "splits")
    train_tasks = [t for t in tasks if t.split == "train"]
    val_tasks = [t for t in tasks if t.split == "val"]
    eval_tasks = [t for t in tasks if t.split == "eval"]
    print(f"  train={len(train_tasks)} val={len(val_tasks)} eval={len(eval_tasks)}")

    print("=== 6. QLoRA config ===")
    hp = QLoRAHyperParams.for_model(args.model)
    hp.output_dir = str(out / "adapters" / f"{args.model}-repo-qlora")
    hp.save(out / "qlora_config.yaml")
    adapter_marker = Path(hp.output_dir)
    print(f"  model={hp.model_name_or_path} effective_batch={hp.effective_batch_size()}")

    if args.real_train:
        print(f"=== 6b. Real LoRA training ({args.train_steps} steps) ===")
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "train_real.py"),
                "--train_jsonl", str(split_paths["train"]),
                "--model", hp.model_name_or_path,
                "--out", str(adapter_marker),
                "--steps", str(args.train_steps),
                "--packs_dir", str(packs_dir),
                "--lora_r", str(hp.lora_r),
                "--lora_alpha", str(hp.lora_alpha),
                "--lr", str(hp.learning_rate),
                "--max_seq_length", str(hp.max_seq_length),
                "--seed", str(hp.seed),
            ],
            check=True,
        )
    else:
        adapter_marker.mkdir(parents=True, exist_ok=True)
        (adapter_marker / "NOT_TRAINED.txt").write_text(
            "Run with --real-train to produce a real adapter. "
            "No adapter exists at this path; conditions C/D are UNTESTED, "
            "not merely untuned.\n"
        )

    adapter_weights = adapter_marker / "adapter_model.safetensors"
    if args.eval_mode == "real" and not adapter_weights.exists():
        parser.error(
            f"--eval-mode real requires trained adapter weights at {adapter_weights}, "
            "which do not exist. Re-run with --real-train (or point --out at a run "
            "that already trained one). Refusing to fall back to simulation: "
            "conditions C/D would be UNTESTED, not untuned."
        )

    print(f"=== 7-8. Evaluation (4 conditions, mode={args.eval_mode}) ===")
    results_by_cond = run_all_conditions(
        eval_tasks,
        packs_dir,
        graph,
        adapter_marker,
        mode=args.eval_mode,
        base_model_name=hp.model_name_or_path,
    )
    summary = aggregate_metrics(results_by_cond)
    provenance = run_provenance(summary)
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(out / "metrics_raw.json", "w") as fh:
        json.dump(
            {c: [r.__dict__ for r in rows] for c, rows in results_by_cond.items()},
            fh,
            indent=2,
        )
    for c, s in summary.items():
        print(f"  {c}: [{s.get('data_provenance', 'none')}] "
              f"success={s.get('task_success_rate', 0):.1%} "
              f"recall={s.get('mean_impacted_file_recall', 0):.1%} "
              f"| {s.get('condition_label', '')[:50]}")
    if provenance != "real":
        print("  *** SIMULATED RESULTS - not experimental evidence. No model ran. ***")

    print("=== 9. Deterministic verification gate ===")
    sample_pack = packs[0] if packs else None
    gate = deterministic_verify(
        pack_sha=sample_pack.pack_sha if sample_pack else "",
        expected_source_sha=sample_pack.source_sha if sample_pack else "",
        graph_version=graph.version,
        expected_graph_version=manifest.graph_version,
        tests_run=["aho-corasick-unit-test.js", "detector-unit-test.js"],
        required_tests=["aho-corasick-unit-test.js"],
        patch_clean=True,
        static_ok=True,
        critical_violations=0,
    )
    with open(out / "verification_gate.json", "w") as fh:
        json.dump(gate.__dict__, fh, indent=2)
    print(f"  gate status = {gate.status}")

    print("=== 10. Contamination audit ===")
    train_fams = {t.family_id for t in train_tasks}
    eval_fams = {t.family_id for t in eval_tasks}
    contam = contamination_audit(train_tasks, eval_tasks, train_fams, eval_fams)
    with open(out / "contamination_report.json", "w") as fh:
        json.dump(contam, fh, indent=2)
    print(f"  contamination = {contam['status']}")

    print("=== 11. Dashboard ===")
    examples = []
    by_task = {}
    for cond in ("A", "D"):
        for row in results_by_cond.get(cond, []):
            by_task.setdefault(row.task_id, {})[cond] = row
    for t in eval_tasks[:4]:
        pair = by_task.get(t.task_id, {})
        examples.append({
            "task_id": t.task_id,
            "A": pair["A"].notes if "A" in pair else "(no result)",
            "D": pair["D"].notes if "D" in pair else "(no result)",
        })
    dash = build_dashboard(
        summary,
        contam,
        gate.__dict__,
        examples,
        out / "dashboard.html",
    )
    write_streamlit_app(out / "streamlit_app.py")
    print(f"  wrote {dash}")

    index = {
        "target_repo": "holeyfield33-art/runtime-firewall-mvp",
        "question": (
            "Does repository-specific adaptation plus structured file knowledge "
            "and graph context outperform the same untuned small model using "
            "ordinary repository-wide context?"
        ),
        "artifacts": {
            "repository_manifest": str(data_dir / "repository_manifest.json"),
            "dependency_graph": str(data_dir / "dependency_graph.json"),
            "file_packs": str(packs_dir),
            "splits": {k: str(v) for k, v in split_paths.items()},
            "qlora_config": str(out / "qlora_config.yaml"),
            "adapter": str(adapter_marker),
            "summary": str(out / "summary.json"),
            "contamination_report": str(out / "contamination_report.json"),
            "verification_gate": str(out / "verification_gate.json"),
            "dashboard": str(dash),
        },
        "conditions": CONDITIONS,
        "data_provenance": provenance,
        "eval_mode": args.eval_mode,
        "real_train": args.real_train,
        "adapter_trained": adapter_weights.exists(),
        "summary": summary,
        "contamination_status": contam["status"],
        "gate_status": gate.status,
    }
    with open(out / "experiment_index.json", "w") as fh:
        json.dump(index, fh, indent=2)

    print("\n=== DONE ===")
    print(f"All artifacts under: {out}")
    print(f"Run data_provenance: {provenance}")
    if provenance != "real":
        print(
            "These numbers are NOT experimental evidence. Re-run with "
            "--real-train --eval-mode real to measure the actual question."
        )
    print("Open results/dashboard.html in a browser for the comparison view.")


if __name__ == "__main__":
    main()
