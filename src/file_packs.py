"""
2. File-pack generation
One versioned pack per source file. Deterministic fields + optional semantic
fields that MUST carry provenance. Never treat LLM summaries as authoritative.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .ingestion import FileRecord, RepoManifest, compute_sha256


@dataclass
class Provenance:
    source: str  # "file:line", "test", "commit", "issue", "static"
    ref: str
    note: Optional[str] = None


@dataclass
class SemanticField:
    value: str
    provenance: List[Provenance]


@dataclass
class FilePack:
    path: str
    source_sha: str
    pack_sha: str
    language: str
    exact_source: str
    imports: List[str]
    exports: List[str]
    symbols: List[str]
    callers: List[str]
    callees: List[str]
    related_tests: List[str]
    dependency_edges: List[Dict[str, str]]
    historical_change_refs: List[str]
    # Semantic (must have provenance)
    purpose: Optional[SemanticField] = None
    contracts: Optional[SemanticField] = None
    invariants: Optional[SemanticField] = None
    known_bug_patterns: Optional[SemanticField] = None
    version: str = "1"


def build_pack(
    fr: FileRecord,
    graph_edges: List[Dict[str, str]],
    historical_refs: List[str],
    callers: List[str],
    callees: List[str],
) -> FilePack:
    content = fr.content or ""
    pack_body = {
        "path": fr.path,
        "source_sha": fr.sha256,
        "language": fr.language,
        "exact_source": content,
        "imports": fr.imports,
        "exports": fr.exports,
        "symbols": fr.symbols,
        "callers": callers,
        "callees": callees,
        "related_tests": fr.related_tests,
        "dependency_edges": graph_edges,
        "historical_change_refs": historical_refs,
    }
    pack_sha = compute_sha256(json.dumps(pack_body, sort_keys=True))
    return FilePack(
        path=fr.path,
        source_sha=fr.sha256,
        pack_sha=pack_sha,
        language=fr.language,
        exact_source=content,
        imports=fr.imports,
        exports=fr.exports,
        symbols=fr.symbols,
        callers=callers,
        callees=callees,
        related_tests=fr.related_tests,
        dependency_edges=graph_edges,
        historical_change_refs=historical_refs,
        version="1",
    )


def generate_file_packs(
    manifest: RepoManifest,
    graph: "DependencyGraph",  # forward
    out_dir: Path,
) -> List[FilePack]:
    out_dir.mkdir(parents=True, exist_ok=True)
    packs: List[FilePack] = []
    path_to_commits: Dict[str, List[str]] = {}
    for c in manifest.commits:
        for f in c.files_changed:
            path_to_commits.setdefault(f, []).append(c.sha)

    for fr in manifest.files:
        if fr.language in ("markdown", "json", "yaml", "unknown") and not fr.path.endswith(".js"):
            # still pack source of interest; skip pure docs for density if desired
            pass
        edges = graph.edges_for(fr.path)
        callers = graph.callers_of(fr.path)
        callees = graph.callees_of(fr.path)
        hist = path_to_commits.get(fr.path, [])[:20]
        pack = build_pack(fr, edges, hist, callers, callees)
        packs.append(pack)
        safe_name = fr.path.replace("/", "__").replace("\\", "__")
        pack_path = out_dir / f"{safe_name}.pack.json"
        with open(pack_path, "w", encoding="utf-8") as fh:
            d = asdict(pack)
            # keep exact_source; it is the ground truth
            json.dump(d, fh, indent=2)
    # index
    index = {
        "repo": manifest.repo_name,
        "graph_version": manifest.graph_version,
        # Packs hold the file state at this commit. Training on a historical
        # task with packs built here means the prompt carries post-cutoff
        # state; train_real checks for that and records it.
        "head_sha": manifest.head_sha,
        "pack_count": len(packs),
        "packs": [{"path": p.path, "source_sha": p.source_sha, "pack_sha": p.pack_sha} for p in packs],
    }
    with open(out_dir / "index.json", "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)
    return packs
