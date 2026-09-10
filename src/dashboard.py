"""
11. Lightweight local dashboard (Streamlit + optional static HTML).
Compares base vs tuned, RAG vs packs, packs vs packs+graph.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .evaluation import run_provenance

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>RSEF Experiment Dashboard — runtime-firewall-mvp</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body { font-family: ui-sans-serif, system-ui, sans-serif; margin: 2rem; background: #0f1419; color: #e7e9ea; }
  h1, h2 { color: #1d9bf0; }
  .card { background: #1a2332; border-radius: 12px; padding: 1.25rem; margin-bottom: 1.5rem; border: 1px solid #2f3336; }
  table { width: 100%; border-collapse: collapse; }
  th, td { padding: 0.5rem 0.75rem; text-align: left; border-bottom: 1px solid #2f3336; }
  th { color: #8b98a5; font-weight: 600; }
  .pass { color: #00ba7c; }
  .fail { color: #f4212e; }
  .mono { font-family: ui-monospace, monospace; font-size: 0.9em; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  .banner { border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 1.5rem;
            font-weight: 600; letter-spacing: 0.01em; line-height: 1.5; }
  .banner-sim { background: #4a1d00; border: 2px solid #ff7a00; color: #ffd9b0; }
  .banner-mixed { background: #4a3400; border: 2px solid #f5c518; color: #ffeeb0; }
  .banner-real { background: #06301f; border: 2px solid #00ba7c; color: #b6f2d9; }
  .banner small { display: block; font-weight: 400; margin-top: 0.4rem; color: inherit; opacity: 0.85; }
  .tag { font-size: 0.75em; padding: 0.15rem 0.45rem; border-radius: 999px;
         font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; }
  .tag-simulated { background: #ff7a00; color: #1a1000; }
  .tag-real { background: #00ba7c; color: #04241a; }
  .tag-none { background: #2f3336; color: #8b98a5; }
  @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div id="provenance-banner" class="banner"></div>

<h1>Repository Specialization Experiment</h1>
<p class="mono">Target: holeyfield33-art/runtime-firewall-mvp · Framework v0.1.0</p>
<p>Falsifiable question: Does repo-specific QLoRA + structured file packs + graph neighborhood
outperform the same untuned small model with ordinary repo-wide context?</p>

<div class="card">
  <h2>Condition Summary</h2>
  <div id="bar"></div>
  <table id="summary-table">
    <thead><tr>
      <th>Condition</th><th>Provenance</th><th>Success</th><th>Impact Recall</th>
      <th>Halluc. APIs</th><th>Tokens</th><th>Latency (ms)</th><th>Label</th>
    </tr></thead>
    <tbody></tbody>
  </table>
</div>

<div class="card">
  <h2>Contamination Audit</h2>
  <pre id="contam" class="mono"></pre>
</div>

<div class="card">
  <h2>Verification Gate (sample)</h2>
  <pre id="gate" class="mono"></pre>
</div>

<div class="card">
  <h2>Failure / Success Examples (side-by-side)</h2>
  <div class="grid" id="examples"></div>
</div>

<script>
const DATA = __DATA_JSON__;

const PROVENANCE = DATA.provenance || "none";
const banner = document.getElementById("provenance-banner");
if (PROVENANCE === "real") {
  banner.className = "banner banner-real";
  banner.innerHTML = "\u2713 REAL RESULTS \u2014 every condition ran an actual model forward pass." +
    "<small>Conditions C/D loaded real LoRA adapter weights. These numbers are experimental evidence.</small>";
} else if (PROVENANCE === "mixed") {
  banner.className = "banner banner-mixed";
  banner.innerHTML = "\u26a0 MIXED RUN \u2014 some conditions are simulated." +
    "<small>A run is only evidence when every condition is real. Check the Provenance column below before quoting any figure.</small>";
} else {
  banner.className = "banner banner-sim";
  banner.innerHTML = "\u26a0 SIMULATED RESULTS \u2014 not experimental evidence." +
    "<small>No model ran. Scores come from hardcoded per-condition base rates plus seeded noise, " +
    "so condition D wins by construction. This mode exists to smoke-test the pipeline without a GPU. " +
    "Re-run with <code>--real-train --eval-mode real</code> to measure anything.</small>";
}

const tbody = document.querySelector("#summary-table tbody");
const conds = ["A","B","C","D"];
const successRates = [];
conds.forEach(c => {
  const s = DATA.summary[c] || {};
  successRates.push(s.task_success_rate || 0);
  const tr = document.createElement("tr");
  const prov = s.data_provenance || "none";
  tr.innerHTML = `<td><b>${c}</b></td>
    <td><span class="tag tag-${prov}">${prov}</span></td>
    <td>${((s.task_success_rate||0)*100).toFixed(1)}%</td>
    <td>${((s.mean_impacted_file_recall||0)*100).toFixed(1)}%</td>
    <td>${(s.mean_hallucinated_apis||0).toFixed(2)}</td>
    <td>${Math.round(s.mean_token_usage||0)}</td>
    <td>${Math.round(s.mean_latency_ms||0)}</td>
    <td>${s.condition_label||""}</td>`;
  tbody.appendChild(tr);
});

Plotly.newPlot("bar", [{
  x: conds.map(c => c + ": " + (DATA.summary[c]?.condition_label||"").slice(0,40)),
  y: successRates.map(v => v*100),
  type: "bar",
  marker: { color: ["#f4212e","#ff7a00","#1d9bf0","#00ba7c"] }
}], {
  paper_bgcolor: "#1a2332", plot_bgcolor: "#1a2332",
  font: { color: "#e7e9ea" },
  yaxis: { title: "Task success %", range: [0,100] },
  title: { text: PROVENANCE === "real" ? "Task success (real inference)"
                                       : "Task success (SIMULATED \u2014 not evidence)",
           font: { size: 14, color: PROVENANCE === "real" ? "#00ba7c" : "#ff7a00" } },
  margin: { t: 30 }
});

document.getElementById("contam").textContent = JSON.stringify(DATA.contamination, null, 2);
document.getElementById("gate").textContent = JSON.stringify(DATA.sample_gate, null, 2);

const ex = document.getElementById("examples");
(DATA.examples || []).forEach(pair => {
  // pair.A / pair.D can contain raw model output. Build with textContent so
  // markup in a completion renders as characters, never as DOM.
  const div = document.createElement("div");
  const h = document.createElement("h3");
  h.className = "mono";
  h.textContent = pair.task_id;
  div.appendChild(h);
  [["A (base RAG)", pair.A], ["D (tuned+packs+graph)", pair.D]].forEach(([label, value]) => {
    const p = document.createElement("p");
    const b = document.createElement("b");
    b.textContent = label;
    p.appendChild(b);
    p.appendChild(document.createTextNode(": " + (value == null ? "" : String(value))));
    div.appendChild(p);
  });
  ex.appendChild(div);
});
</script>
</body>
</html>
"""


def _script_safe_json(payload: Dict[str, Any]) -> str:
    """JSON encoded so it cannot terminate the enclosing <script> element."""
    return (
        json.dumps(payload)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def build_dashboard(
    summary: Dict[str, Any],
    contamination: Dict[str, Any],
    sample_gate: Dict[str, Any],
    examples: List[Dict[str, str]],
    out_path: Path,
) -> Path:
    payload = {
        "summary": summary,
        "contamination": contamination,
        "sample_gate": sample_gate,
        "examples": examples,
        # Drives the banner. A dashboard that cannot say where its numbers came
        # from must not render as though they were measured.
        "provenance": run_provenance(summary),
    }
    # In real mode, MetricResult.notes can carry raw model output. That text
    # reaches this page, so it is escaped for safe embedding inside an inline
    # <script>: a completion containing "</script>" would otherwise close the
    # tag and execute whatever followed when the dashboard is opened.
    html = DASHBOARD_HTML.replace("__DATA_JSON__", _script_safe_json(payload))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def write_streamlit_app(out_path: Path) -> None:
    code = '''
import json
from pathlib import Path
import streamlit as st
import pandas as pd

st.set_page_config(page_title="RSEF Dashboard", layout="wide")
st.title("Repository Specialization Experiment")
st.caption("Target: runtime-firewall-mvp · Qwen2.5-Coder-1.5B / SmolLM3-3B")

results_dir = Path("results")
summary_path = results_dir / "summary.json"
if not summary_path.exists():
    st.warning("Run the experiment first (python -m scripts.run_experiment)")
    st.stop()

summary = json.loads(summary_path.read_text())
contam = json.loads((results_dir / "contamination_report.json").read_text())

labels = {s.get("data_provenance", "none") for s in summary.values()} - {"none"}
if labels == {"real"}:
    st.success(
        "REAL RESULTS - every condition ran an actual model forward pass. "
        "Conditions C/D loaded real LoRA adapter weights."
    )
elif labels == {"simulated"} or not labels:
    st.error(
        "SIMULATED RESULTS - not experimental evidence. No model ran; scores come "
        "from hardcoded per-condition base rates plus seeded noise, so condition D "
        "wins by construction. Re-run with --real-train --eval-mode real to measure "
        "anything."
    )
else:
    st.warning(
        "MIXED RUN - some conditions are simulated. A run is only evidence when "
        "every condition is real; check the Provenance column before quoting a figure."
    )

st.subheader("Conditions")
rows = []
for c, s in summary.items():
    rows.append({
        "Condition": c,
        "Provenance": s.get("data_provenance", "none"),
        "Success %": round(100 * s.get("task_success_rate", 0), 1),
        "Impact Recall %": round(100 * s.get("mean_impacted_file_recall", 0), 1),
        "Halluc. APIs": round(s.get("mean_hallucinated_apis", 0), 2),
        "Tokens": int(s.get("mean_token_usage", 0)),
        "Latency ms": int(s.get("mean_latency_ms", 0)),
        "Label": s.get("condition_label", ""),
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True)
st.bar_chart(pd.DataFrame(rows).set_index("Condition")["Success %"])

st.subheader("Contamination Audit")
st.json(contam)
st.success("Audit status: " + contam.get("status", "?"))
'''
    out_path.write_text(code, encoding="utf-8")
