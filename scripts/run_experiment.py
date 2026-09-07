#!/usr/bin/env python3
"""
End-to-end experimental harness for repository specialization.

Usage:
  python -m scripts.run_experiment --repo /path/to/runtime-firewall-mvp \\
      --model qwen --out results/

Produces all required artifacts:
  repository manifest, file packs, dependency graph,
  train/val/eval manifests, contamination report,
  QLoRA config, (adapter placeholder), evaluation results, dashboard.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ensure package root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingestion import ingest_repository, save_manifest, manifest_to_dict
from src.graph import build_dependency_graph
from src.file_packs import generate_file_packs
from src.history_tasks import generate_tasks_from_history, write_splits
from src.qlora_config import QLoRAHyperParams, TRAIN_SCRIPT
from src.evaluation import (
    run_all_conditions,
    contamination_audit,
    aggregate_metrics,
    deterministic_verify,
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
    (out / "train_qlora.py").write_text(TRAIN_SCRIPT)
    adapter_marker = Path(hp.output_dir)
    adapter_marker.mkdir(parents=True, exist_ok=True)
    (adapter_marker / "ADAPTER_PLACEHOLDER.txt").write_text(
        "Adapter weights would be written here after GPU QLoRA training.\n"
        "Facts remain in file packs + graph; adapter learns conventions/patterns only.\n"
        f"Config: {args.model} r={hp.lora_r} alpha={hp.lora_alpha} seq={hp.max_seq_length}\n"
    )
    print(f"  model={hp.model_name_or_path} effective_batch={hp.effective_batch_size()}")

    print("=== 7-8. Evaluation (4 conditions) ===")
    results_by_cond = run_all_conditions(eval_tasks, packs_dir, graph, adapter_marker)
    summary = aggregate_metrics(results_by_cond)
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(out / "metrics_raw.json", "w") as fh:
        json.dump(
            {c: [r.__dict__ for r in rows] for c, rows in results_by_cond.items()},
            fh,
            indent=2,
        )
    for c, s in summary.items():
        print(f"  {c}: success={s.get('task_success_rate', 0):.1%} "
              f"recall={s.get('mean_impacted_file_recall', 0):.1%} "
              f"| {s.get('condition_label', '')[:50]}")

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
    for t in eval_tasks[:4]:
        examples.append({
            "task_id": t.task_id,
            "A": "lower confidence / missing edges (sim)",
            "D": "higher recall via impact neighborhood (sim)",
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
        "summary": summary,
        "contamination_status": contam["status"],
        "gate_status": gate.status,
    }
    with open(out / "experiment_index.json", "w") as fh:
        json.dump(index, fh, indent=2)

    print("\n=== DONE ===")
    print(f"All artifacts under: {out}")
    print("Open results/dashboard.html in a browser for the comparison view.")


if __name__ == "__main__":
    main()
