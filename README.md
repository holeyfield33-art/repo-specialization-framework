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

## Quick start

```bash
# from repo root
pip install -r requirements.txt

# run against the extracted Helios sample (or a full clone)
python -m scripts.run_experiment \
  --repo ../helios-sample \
  --model qwen \
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

### Real QLoRA (requires GPU + peft/bitsandbytes)

```bash
python results/train_qlora.py \
  --config results/qlora_config.yaml \
  --train_jsonl results/data/splits/train.jsonl \
  --val_jsonl results/data/splits/val.jsonl
```

On CPU-only environments the runner writes an adapter placeholder and simulates condition scores so the full evaluation + contamination + dashboard path can still be exercised.

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
│   └── run_experiment.py # end-to-end
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
