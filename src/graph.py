"""
3. Dependency graph
Versioned repository graph with typed edges. Provenance stored.
Impact neighborhood for changed files; unresolved boundaries recorded as uncertainty.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import networkx as nx

from .ingestion import RepoManifest


EDGE_TYPES = (
    "IMPORTS",
    "CALLS",
    "TESTS",
    "INHERITS",
    "IMPLEMENTS",
    "READS_CONFIG",
    "WRITES_DATA",
)


@dataclass
class EdgeProvenance:
    method: str  # "static-import", "static-call", "test-heuristic", "name-match"
    confidence: float
    detail: Optional[str] = None


@dataclass
class GraphEdge:
    src: str
    dst: str
    edge_type: str
    provenance: EdgeProvenance


class DependencyGraph:
    def __init__(self, version: str = ""):
        self.g = nx.DiGraph()
        self.version = version
        self.edges: List[GraphEdge] = []
        self.uncertainty: List[Dict[str, str]] = []

    def add_node(self, path: str, **attrs: Any) -> None:
        self.g.add_node(path, **attrs)

    def add_edge(
        self,
        src: str,
        dst: str,
        edge_type: str,
        method: str = "static",
        confidence: float = 0.8,
        detail: Optional[str] = None,
    ) -> None:
        if edge_type not in EDGE_TYPES:
            edge_type = "IMPORTS"
        prov = EdgeProvenance(method=method, confidence=confidence, detail=detail)
        self.edges.append(GraphEdge(src=src, dst=dst, edge_type=edge_type, provenance=prov))
        self.g.add_edge(src, dst, edge_type=edge_type, confidence=confidence, method=method)

    def edges_for(self, path: str) -> List[Dict[str, str]]:
        out = []
        for e in self.edges:
            if e.src == path or e.dst == path:
                out.append(
                    {
                        "src": e.src,
                        "dst": e.dst,
                        "type": e.edge_type,
                        "method": e.provenance.method,
                        "confidence": str(e.provenance.confidence),
                    }
                )
        return out

    def callers_of(self, path: str) -> List[str]:
        return [u for u, v, d in self.g.in_edges(path, data=True) if d.get("edge_type") in ("CALLS", "IMPORTS")]

    def callees_of(self, path: str) -> List[str]:
        return [v for u, v, d in self.g.out_edges(path, data=True) if d.get("edge_type") in ("CALLS", "IMPORTS")]

    def impact_neighborhood(
        self,
        changed_paths: List[str],
        radius: int = 2,
        max_nodes: int = 40,
    ) -> Dict[str, Any]:
        """Compute impact neighborhood rather than full repo."""
        seeds = set(changed_paths) & set(self.g.nodes)
        if not seeds:
            return {
                "seeds": changed_paths,
                "nodes": [],
                "edges": [],
                "uncertainty": ["no seeds present in graph"],
            }
        nodes: Set[str] = set(seeds)
        for s in seeds:
            try:
                ego = nx.ego_graph(self.g.to_undirected(), s, radius=radius)
                nodes.update(ego.nodes)
            except Exception:
                pass
        if len(nodes) > max_nodes:
            others = sorted(
                (n for n in nodes if n not in seeds),
                key=lambda n: self.g.degree(n) if n in self.g else 0,
                reverse=True,
            )
            nodes = set(seeds) | set(others[: max_nodes - len(seeds)])
        sub = self.g.subgraph(nodes)
        edge_list = [
            {
                "src": u,
                "dst": v,
                "type": d.get("edge_type", "IMPORTS"),
                "confidence": d.get("confidence", 0.5),
            }
            for u, v, d in sub.edges(data=True)
        ]
        unresolved = []
        for n in nodes:
            for e in self.edges:
                if e.src == n and e.dst not in self.g:
                    unresolved.append({"node": n, "missing": e.dst, "type": e.edge_type})
        return {
            "seeds": list(seeds),
            "nodes": list(nodes),
            "edges": edge_list,
            "uncertainty": unresolved[:50],
            "radius": radius,
            "graph_version": self.version,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "node_count": self.g.number_of_nodes(),
            "edge_count": len(self.edges),
            "nodes": list(self.g.nodes),
            "edges": [
                {
                    "src": e.src,
                    "dst": e.dst,
                    "type": e.edge_type,
                    "method": e.provenance.method,
                    "confidence": e.provenance.confidence,
                    "detail": e.provenance.detail,
                }
                for e in self.edges
            ],
            "uncertainty": self.uncertainty,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)


def build_dependency_graph(manifest: RepoManifest) -> DependencyGraph:
    g = DependencyGraph(version=manifest.graph_version)
    path_set = {f.path for f in manifest.files}
    path_by_stem: Dict[str, List[str]] = {}
    for f in manifest.files:
        g.add_node(f.path, language=f.language, sha=f.sha256)
        stem = Path(f.path).stem
        path_by_stem.setdefault(stem, []).append(f.path)

    for fr in manifest.files:
        for imp in fr.imports:
            candidates = []
            if imp.startswith("."):
                base = Path(fr.path).parent
                for ext in ("", ".js", ".mjs", ".cjs", "/index.js"):
                    cand = str((base / (imp + ext)).as_posix())
                    if cand in path_set:
                        candidates.append(cand)
            else:
                if not any(imp in p for p in path_set):
                    g.uncertainty.append({"src": fr.path, "import": imp, "reason": "external_or_unresolved"})
                    continue
            for c in candidates:
                g.add_edge(fr.path, c, "IMPORTS", method="static-import", confidence=0.9, detail=imp)

        for t in fr.related_tests:
            if t in path_set:
                g.add_edge(t, fr.path, "TESTS", method="test-heuristic", confidence=0.7)

        for sym in fr.symbols:
            for other in manifest.files:
                if other.path == fr.path:
                    continue
                if sym in other.symbols or any(sym in (other.content or "") for _ in [0]):
                    g.add_edge(fr.path, other.path, "CALLS", method="name-match", confidence=0.4, detail=sym)

    return g
