"""
Prompt assembly shared by real training (scripts/train_real.py) and real
evaluation (src.evaluation.real_model_response).

Both paths MUST build prompts the same way, otherwise the adapter is tuned on
one format and evaluated on another and conditions C/D are handicapped by a
formatting mismatch rather than measured on their merits.

Condition context ladder (this is the experiment's independent variable):
  A  instruction + context file paths + flat untructured source concat
  B  instruction + context file paths + structured file-pack fields
  C  same context as B (difference vs B is the adapter, not the context)
  D  same as C + dependency-graph impact neighborhood
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

ANSWER_SCHEMA_HINT = (
    'Respond with JSON only, no prose, in exactly this form:\n'
    '{"impacted": ["path/one.js", "path/two.js"]}'
)

SYSTEM_PROMPT = (
    "You are a code assistant working on a single repository. "
    "Answer only from the evidence supplied in the prompt. "
    "Never invent file paths that are not present in the evidence. "
    "Reply with JSON only."
)

# Character budgets keep A/B/C/D roughly volume-matched so the measured
# difference is structure and tuning, not raw context size.
FLAT_SOURCE_BUDGET = 6000
PACK_SOURCE_BUDGET = 1200
MAX_CONTEXT_FILES = 12


def expected_answer_paths(task: Any) -> List[str]:
    """The path list a model is expected to produce, matching how
    evaluate_condition scores impacted_file_recall."""
    gt: Dict[str, Any] = getattr(task, "ground_truth", None) or {}
    for key in ("impacted", "tests_to_run", "primary_files", "existing_tests"):
        value = gt.get(key)
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            return value
    return []


def is_scoreable(task: Any) -> bool:
    """True when the task's ground truth yields a path list the evaluator can
    actually compare a prediction against.

    Tasks without one (code_review stores `files`/`suggested_checks`,
    cross_file_dependency_reasoning stores `edges`) cannot be scored by
    impacted-file overlap. They must be EXCLUDED rather than scored, because
    "no expected answer" previously meant any non-empty prediction counted as a
    success — a hallucinated path list scored 100%.
    """
    return bool(expected_answer_paths(task))


def pack_path_for(packs_dir: Path, repo_path: str) -> Path:
    """Mirror of the naming used by file_packs.generate_file_packs."""
    safe_name = repo_path.replace("/", "__").replace("\\", "__")
    return packs_dir / f"{safe_name}.pack.json"


def load_pack(packs_dir: Optional[Path], repo_path: str) -> Optional[Dict[str, Any]]:
    if packs_dir is None:
        return None
    p = pack_path_for(packs_dir, repo_path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"file pack {p} is not valid JSON: {exc}") from exc


def known_repo_paths(packs_dir: Optional[Path]) -> List[str]:
    """Every path the repository actually contains, per the pack index.
    Used to count hallucinated references in a real model answer."""
    if packs_dir is None:
        return []
    index = packs_dir / "index.json"
    if not index.exists():
        return []
    data = json.loads(index.read_text(encoding="utf-8"))
    return [entry["path"] for entry in data.get("packs", [])]


def _flat_source_block(packs_dir: Optional[Path], paths: List[str]) -> str:
    """Condition A: ordinary RAG — concatenated source, no structure."""
    chunks: List[str] = []
    remaining = FLAT_SOURCE_BUDGET
    for path in paths:
        if remaining <= 0:
            break
        pack = load_pack(packs_dir, path)
        if pack is None:
            continue
        source = (pack.get("exact_source") or "")[:remaining]
        remaining -= len(source)
        chunks.append(f"--- {path} ---\n{source}")
    if not chunks:
        return ""
    return "Repository context:\n" + "\n".join(chunks)


def _pack_block(packs_dir: Optional[Path], paths: List[str]) -> str:
    """Conditions B/C/D: structured file packs."""
    blocks: List[str] = []
    for path in paths:
        pack = load_pack(packs_dir, path)
        if pack is None:
            continue
        edges = pack.get("dependency_edges", [])[:8]
        edge_text = ", ".join(
            f"{e.get('src', '?')} -[{e.get('type', 'IMPORTS')}]-> {e.get('dst', '?')}"
            for e in edges
        )
        blocks.append(
            f"### {path}\n"
            f"source_sha: {pack.get('source_sha', '')[:12]}\n"
            f"imports: {pack.get('imports', [])}\n"
            f"exports: {pack.get('exports', [])}\n"
            f"symbols: {pack.get('symbols', [])[:12]}\n"
            f"callers: {pack.get('callers', [])}\n"
            f"callees: {pack.get('callees', [])}\n"
            f"related_tests: {pack.get('related_tests', [])}\n"
            f"edges: {edge_text}\n"
            f"source:\n{(pack.get('exact_source') or '')[:PACK_SOURCE_BUDGET]}"
        )
    if not blocks:
        return ""
    return "Structured file packs:\n" + "\n\n".join(blocks)


def _graph_block(graph: Any, seeds: List[str]) -> str:
    """Condition D only: dependency-graph impact neighborhood."""
    if graph is None or not seeds:
        return ""
    neigh = graph.impact_neighborhood(seeds, radius=1, max_nodes=25)
    edge_lines = [
        f"  {e['src']} -[{e['type']}, conf={e.get('confidence', 0.5)}]-> {e['dst']}"
        for e in neigh.get("edges", [])[:40]
    ]
    uncertainty = neigh.get("uncertainty", [])
    return (
        "Dependency graph impact neighborhood "
        f"(graph_version={neigh.get('graph_version', '')[:12]}, radius={neigh.get('radius', 1)}):\n"
        f"seeds: {neigh.get('seeds', [])}\n"
        f"nodes: {neigh.get('nodes', [])}\n"
        "edges:\n" + ("\n".join(edge_lines) if edge_lines else "  (none resolved)") + "\n"
        f"unresolved boundaries (uncertainty): {uncertainty[:5]}"
    )


def build_prompt(
    task: Any,
    condition: str,
    packs_dir: Optional[Path] = None,
    graph: Any = None,
) -> str:
    """Assemble the user-turn prompt for one task under one condition.

    NOTE ON A KNOWN TASK-DESIGN LEAK: task.context_files for
    change_impact_prediction tasks is built as `changed + related[:8]` while the
    ground truth is `related[:10]`, so the answer is partially present in the
    context for every condition. That is a property of history_tasks (out of
    scope for this change), it affects A/B/C/D equally, and evaluation.py flags
    affected tasks in MetricResult.notes so the inflation is visible rather than
    silent.
    """
    context_files = list(getattr(task, "context_files", []) or [])[:MAX_CONTEXT_FILES]
    parts: List[str] = [task.instruction, f"Context files: {context_files}"]

    if condition == "A":
        block = _flat_source_block(packs_dir, context_files)
    else:
        block = _pack_block(packs_dir, context_files)
    if block:
        parts.append(block)

    if condition == "D":
        seeds = (task.ground_truth or {}).get("seeds") or context_files[:3]
        graph_block = _graph_block(graph, [s for s in seeds if isinstance(s, str)])
        if graph_block:
            parts.append(graph_block)

    parts.append(ANSWER_SCHEMA_HINT)
    return "\n\n".join(parts)


def build_target(task: Any) -> str:
    """The supervised completion for training: the same JSON schema the
    evaluator parses."""
    return json.dumps({"impacted": expected_answer_paths(task)})


def parse_answer(completion: str) -> Optional[List[str]]:
    """Extract the predicted path list from a model completion.

    Returns None when the completion contains no parseable JSON object with an
    `impacted` list — the caller records that as a real failure, never as a
    fallback score.
    """
    start = completion.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(completion)):
            if completion[i] == "{":
                depth += 1
            elif completion[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(completion[start : i + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict):
                        value = obj.get("impacted")
                        if isinstance(value, list):
                            return [str(v) for v in value]
                    break
        start = completion.find("{", start + 1)
    return None
