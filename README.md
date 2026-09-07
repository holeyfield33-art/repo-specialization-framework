# Repository Specialization Experimental Framework (RSEF)

**Research harness** — not a product.

Falsifiable question:

> Does repository-specific adaptation plus structured file knowledge and graph context
> outperform the same untuned small model using ordinary repository-wide context?

Target repository for this experiment: **[holeyfield33-art/runtime-firewall-mvp](https://github.com/holeyfield33-art/runtime-firewall-mvp)** (Aletheia / Helios runtime firewall for Node.js).

## Supported models

| Key | Hugging Face ID | Notes |
|-----|-----------------|-------|
| `qwen` (default) | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | Primary |
| `smol` | `HuggingFaceTB/SmolLM3-3B` | Alternative; configurable |

Any 0.5B–3B instruction-tuned causal LM can be added via `QLoRAHyperParams.for_model`.

## Pipeline (matches design §1–12)

1. **Ingestion** — enumerate files, SHA-256, language, imports/exports/symbols, test map, git history, change families. Deterministic only.
2. **File packs** — one versioned pack per source file (path, source SHA, exact source, imports, exports, callers/callees, tests, edges, historical refs). Semantic fields require provenance.
3. **Dependency graph** — typed edges (`IMPORTS`, `CALLS`, `TESTS`, …) with provenance; impact neighborhood for changed files; unresolved boundaries → uncertainty.
4. **Historical training data** — split unit = **change family**. Temporal: earlier → train, later → eval. No cross-family leakage.
5. **Tasks** — evidence-grounded: bug localization, root-cause, change-impact, patch, test-impact, cross-file reasoning, code review, invariant checks. Synthetic only when mechanically validated.
6. **QLoRA** — 4-bit, configurable rank/α/dropout/seq/LR/epochs. Adapter stored separately; volatile facts stay in packs/graph.
7. **Conditions**  
   - A: Base + ordinary RAG/context  
   - B: Base + structured packs  
   - C: Tuned + packs  
   - D: Tuned + packs + graph impact neighborhood  
8. **Metrics** — task success, test pass, bug-loc accuracy, impacted-file recall, cross-file accuracy, hallucinated APIs, unnecessary edits, security regressions, tokens, latency. Unit-test gen / doc synthesis are secondary.
9. **Verification gate** — non-LLM: SHA match, graph version, tests run, patch applies, static checks, invariants → `PASS` / `BLOCK` / `FREEZE`.
10. **Contamination audit** — machine-readable proof of no eval family/commit/answer leakage into train.
11. **Dashboard** — static HTML + optional Streamlit.
12. **Artifacts** — manifest, packs, graph, splits, contamination report, QLoRA config, adapter dir, results, dashboard.

## Integrity guarantees

- The complete reachable Git history is ingested; history is never silently depth-capped.
- The change family remains the split unit and the existing temporal 70/15/15 split is preserved.
- Every evaluation task is tied to a real held-out commit and reads source from that commit's parent snapshot.
- Synthetic/current-HEAD examples are excluded because they cannot be proven earlier than held-out work.
- The contamination audit runs before model loading or training and blocks the run on failure.
- CUDA is mandatory for measured runs. There is no simulator, CPU score fallback, or adapter placeholder.
- Conditions A and B finish and are saved before QLoRA training. Conditions C and D load the saved adapter.
- Patch tasks are applied in disposable Git worktrees and the configured repository test command is executed.
- Invalid model JSON is retained as raw task output and scored as a real failure; it is never replaced with invented output.

## Colab: first real run

```bash
# Colab terminal/cell commands. Use full clones; do not add --depth.
cd /content
git clone https://github.com/holeyfield33-art/repo-specialization-framework.git
git clone https://github.com/holeyfield33-art/runtime-firewall-mvp.git

cd /content/repo-specialization-framework
pip install -r requirements.txt

# Patch verification reuses this dependency installation in disposable worktrees.
cd /content/runtime-firewall-mvp
npm ci --ignore-scripts

cd /content/repo-specialization-framework

# First inspect and freeze the full-history data, counts, splits, and audit.
python -m scripts.run_experiment \
  --repo /content/runtime-firewall-mvp \
  --model qwen \
  --out /content/rsef-results/qwen \
  --prepare-only

# Then run real A/B, QLoRA training, and real C/D with the same prepared data.
python -m scripts.run_experiment \
  --repo /content/runtime-firewall-mvp \
  --model qwen \
  --out /content/rsef-results/qwen \
  --prepared-data /content/rsef-results/qwen/data
```

The runner prints and exports GPU model, VRAM, CUDA and PyTorch versions before inference/training. The pretraining report prints source files, graph edges, independent families, train/validation/evaluation family and task counts, and contamination status.

The committed QLoRA configuration uses `bfloat16`, as specified by the original harness. The runner stops rather than silently switching precision when the assigned GPU lacks BF16 support. Changing that setting requires explicit approval because it changes a training hyperparameter.

### Second small model, unchanged data

```bash
python -m scripts.run_experiment \
  --repo /content/runtime-firewall-mvp \
  --model smol \
  --out /content/rsef-results/smol \
  --prepared-data /content/rsef-results/qwen/data
```

This reuses the Qwen run's frozen manifest and exact train/validation/evaluation JSONL files. The runner aborts if the target repository HEAD differs.

### Artifacts

Each model directory contains:

- `hardware.json`
- `data/pretraining_report.json`
- `data/contamination_report.json`
- `data/repository_manifest.json`
- `data/dependency_graph.json`
- `data/dataset_lock.json` (SHA-256 lock reused by the second model)
- `data/splits/{train,val,eval}.jsonl`
- `qlora_config.yaml`
- `adapters/<model>-repo-qlora/adapter_config.json`
- `adapters/<model>-repo-qlora/adapter_model.safetensors`
- `task_results_ab.json` (saved before training)
- `task_results_abcd.json`
- `metrics.json`
- `verification_gate.json`
- `experiment_index.json`
- `dashboard.html` and `streamlit_app.py`
- `report.md`

Do not interpret partial directories as completed experiments. A completed run must have a `PASS` verification gate and all four non-empty condition result arrays.

## Layout

```
repo-specialization-framework/
├── src/
│   ├── ingestion.py      # §1
│   ├── file_packs.py     # §2
│   ├── graph.py          # §3
│   ├── history_tasks.py  # §4–5
│   ├── qlora_config.py   # §6
│   ├── evaluation.py     # §7–10
│   └── dashboard.py      # §11
├── scripts/
│   ├── train_qlora.py    # real CUDA QLoRA training
│   └── run_experiment.py # fail-closed end-to-end runner
├── configs/
├── data/                 # intermediate (created at runtime)
├── results/              # all output artifacts
├── requirements.txt
└── README.md
```

## Non-goals

- Multi-agent hierarchy, commercial SaaS, hosted control plane, autonomous merge, one-model-per-file.
- One shared base model + one repo-specific adapter.

## License

Apache-2.0 (aligned with the target firewall repository).
