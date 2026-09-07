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

st.subheader("Conditions")
rows = []
for c, s in summary.items():
    rows.append({
        "Condition": c,
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
