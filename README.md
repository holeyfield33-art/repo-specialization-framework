# Repository Specialization Experimental Framework (RSEF)

**Research harness** — not a product.

Falsifiable question:

> Does repository-specific adaptation plus structured file knowledge and graph context
> outperform the same untuned small model using ordinary repository-wide context?

Target repository for this experiment: **[holeyfield33-art/runtime-firewall-mvp](https://github.com/holeyfield33-art/runtime-firewall-mvp)** (Aletheia / Helios runtime firewall for Node.js).

## Two modes — read this before quoting any number

| Mode | Command | What it produces |
|------|---------|------------------|
| **simulate** (default) | `python -m scripts.run_experiment --repo <path>` | **No model runs.** Scores come from hardcoded per-condition base rates plus seeded noise, so condition D wins by construction. Every result is labelled `"data_provenance": "simulated"` and the dashboard shows a warning banner. Use it to smoke-test ingestion, packs, graph, splits and dashboard code without a GPU. |
| **real** | `python -m scripts.run_experiment --repo <path> --real-train --eval-mode real` | Trains a real LoRA adapter and evaluates with real forward passes. Results are labelled `"data_provenance": "real"`. Only these are experimental evidence. |

`--eval-mode real` without trained adapter weights **fails the run**. It never falls
back to simulation: conditions C/D without an adapter are UNTESTED, not merely untuned.
Reused adapter weights must also carry an `adapter_provenance.json` matching this run's
base model, repository HEAD and training split, or the run fails — weights with the right
filename are not proof they were trained for this experiment.

### Known limitation: temporal integrity of file packs

File packs are generated from one repository snapshot (HEAD). Training tasks come from
earlier commits, so their prompts carry state that did not exist at the task's commit —
including state from the commits held out for evaluation. The change-family split does
not prevent this and the contamination audit cannot see it, because it compares task
metadata rather than pack contents.

`train_real.py` detects and records this as `temporal_integrity` in
`training_trace.json` and `adapter_provenance.json`, and warns on stdout. Until packs are
generated per-commit, treat conditions C/D from such an adapter as temporally
contaminated. Generating packs at each task's commit is the real fix and is follow-up work.

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

## Quick start

```bash
# from repo root
pip install -r requirements.txt

# smoke-test the pipeline with no GPU (SIMULATED — not evidence)
python -m scripts.run_experiment \
  --repo ../helios-sample \
  --model qwen \
  --out results/

# the real experiment: trains an adapter, runs real inference for all 4 conditions
python -m scripts.run_experiment \
  --repo ../helios-sample \
  --model qwen \
  --real-train --train-steps 200 \
  --eval-mode real \
  --out results/

# view dashboard
open results/dashboard.html   # or xdg-open / browser

# optional Streamlit
streamlit run results/streamlit_app.py
```

### Full clone of the target (optional)

```bash
git clone --depth 50 https://github.com/holeyfield33-art/runtime-firewall-mvp.git /tmp/runtime-firewall-mvp
python -m scripts.run_experiment --repo /tmp/runtime-firewall-mvp --out results/full/
```

### Training the adapter directly

`--real-train` shells out to `scripts/train_real.py`, which can also be run on its own:

```bash
python scripts/train_real.py \
  --train_jsonl results/data/splits/train.jsonl \
  --packs_dir results/file_packs \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --out results/adapters/qwen-repo-qlora \
  --steps 200
```

The backend is auto-selected: CUDA + bitsandbytes runs real 4-bit QLoRA, CUDA alone
runs bf16 LoRA, CPU falls back to fp32 LoRA (same adapter math, slower, more RAM).
`--no_4bit` forces full precision. It refuses to report success if no weights were
written, and fails loudly on a missing split rather than substituting synthetic data.

**Training telemetry.** Every step records real block-level `dLoss/d(output)` and
`dLoss/dW` norms for all 7 adapted projections in every layer, keyed
`<layer>.<projection>` so a norm traces back to the layer it came from:

- `train_telemetry.jsonl` — one record per step
- `training_trace.json` — loss curve plus a `telemetry_summary` naming which
  projections moved and which stayed quiet

This is the evidence for whether tuning does anything condition B does not. If
`layers_moved` is 0, or `quiet_layers` is long, conditions C/D should not be
expected to differ from A/B — and that is a real finding, not a bug to hide.

Without `--real-train` the runner writes `NOT_TRAINED.txt` into the adapter directory
instead — a marker, not an adapter. Conditions C/D are then UNTESTED, not untuned.

### Tests

```bash
pip install pytest
python -m pytest tests/ -q
```

## Layout

```
repo-specialization-framework/
├── src/
│   ├── ingestion.py      # §1
│   ├── file_packs.py     # §2
│   ├── graph.py          # §3
│   ├── history_tasks.py  # §4–5
│   ├── qlora_config.py   # §6
│   ├── prompting.py      # shared prompt assembly (training + real eval)
│   ├── evaluation.py     # §7–10, real + simulated paths
│   └── dashboard.py      # §11
├── scripts/
│   ├── run_experiment.py # end-to-end
│   └── train_real.py     # real LoRA training + gradient telemetry
├── tests/
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
