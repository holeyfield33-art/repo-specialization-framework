"""Leakage-safe repository snapshots and condition-specific context assembly."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from .graph import DependencyGraph
from .history_tasks import TaskExample
from .ingestion import FileRecord, extract_js_symbols, compute_sha256
from .file_packs import build_pack


SUPPORTED = {".js", ".mjs", ".cjs", ".ts", ".tsx", ".py", ".go", ".rs", ".java", ".md", ".json", ".yml", ".yaml"}
TOKEN_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_$.-]{2,}")


class SnapshotError(RuntimeError):
    pass


@dataclass
class BuiltContext:
    text: str
    files: List[str]
    graph_edges: List[dict]
    source_symbols: List[str]


class GitSnapshot:
    """Read immutable file states directly from Git without checking them out."""

    def __init__(self, repo: Path):
        self.repo = repo.resolve()
        self._file_cache: Dict[Tuple[str, str], str] = {}
        self._tree_cache: Dict[str, List[str]] = {}

    def _git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(self.repo), *args],
            check=check,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def assert_commit(self, commit: str) -> None:
        if not commit:
            raise SnapshotError("task has no pre-change context commit")
        proc = self._git("cat-file", "-e", f"{commit}^{{commit}}", check=False)
        if proc.returncode:
            raise SnapshotError(f"missing context commit {commit}: {proc.stderr.strip()}")

    def list_files(self, commit: str) -> List[str]:
        self.assert_commit(commit)
        if commit not in self._tree_cache:
            out = self._git("ls-tree", "-r", "--name-only", commit).stdout
            self._tree_cache[commit] = [
                p for p in out.splitlines() if Path(p).suffix.lower() in SUPPORTED
            ]
        return self._tree_cache[commit]

    def read(self, commit: str, path: str) -> str | None:
        key = (commit, path)
        if key in self._file_cache:
            return self._file_cache[key]
        proc = self._git("show", f"{commit}:{path}", check=False)
        if proc.returncode:
            return None
        self._file_cache[key] = proc.stdout
        return proc.stdout

    def diff(self, parent: str, commit: str, paths: Sequence[str] | None = None) -> str:
        args = ["diff", "--no-ext-diff", "--unified=3", parent, commit]
        if paths:
            args.extend(["--", *paths])
        return self._git(*args).stdout


def _lexical_rank(query: str, paths: Iterable[str], snapshot: GitSnapshot, commit: str) -> List[str]:
    terms = {t.lower() for t in TOKEN_RE.findall(query)}
    scored = []
    for path in paths:
        content = snapshot.read(commit, path) or ""
        haystack = (path + "\n" + content[:100_000]).lower()
        score = sum(3 if t in path.lower() else min(2, haystack.count(t)) for t in terms)
        if score:
            scored.append((score, path))
    return [p for _, p in sorted(scored, key=lambda x: (-x[0], x[1]))]


def _snapshot_record(path: str, source: str, all_paths: Sequence[str]) -> FileRecord:
    language = {
        ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
        ".ts": "typescript", ".tsx": "typescript", ".py": "python",
    }.get(Path(path).suffix.lower(), "unknown")
    imports, exports, symbols = ([], [], [])
    if language in ("javascript", "typescript"):
        imports, exports, symbols = extract_js_symbols(source)
    stem = Path(path).stem.lower()
    tests = [p for p in all_paths if ("test" in p.lower() or "spec" in p.lower()) and stem in p.lower()]
    return FileRecord(path, compute_sha256(source), language, len(source.encode()), imports, exports, symbols, tests, content=source)


def build_condition_context(
    task: TaskExample,
    condition: str,
    snapshot: GitSnapshot,
    max_files: int = 8,
    max_chars_per_file: int = 12_000,
) -> BuiltContext:
    """Build A/B/C/D context from the task's immutable pre-change snapshot."""
    commit = task.context_commit
    if not commit:
        raise SnapshotError(f"{task.task_id}: missing pre-change context commit")
    all_paths = snapshot.list_files(commit)

    if condition == "A":
        selected = _lexical_rank(task.instruction, all_paths, snapshot, commit)[:max_files]
        chunks = []
        symbols: List[str] = []
        for path in selected:
            source = snapshot.read(commit, path) or ""
            record = _snapshot_record(path, source, all_paths)
            symbols.extend(record.symbols)
            chunks.append(f"FILE: {path}\n{source[:max_chars_per_file]}")
        return BuiltContext("\n\n".join(chunks), selected, [], sorted(set(symbols)))

    requested = [p for p in task.context_files if p in all_paths]
    if not requested:
        requested = _lexical_rank(task.instruction, all_paths, snapshot, commit)[:max_files]
    selected = list(dict.fromkeys(requested))[:max_files]

    # Construct a snapshot-local graph.  Only evidence present before the held-out
    # change is eligible for B/C/D context.
    records = [_snapshot_record(p, snapshot.read(commit, p) or "", all_paths) for p in selected]
    graph = DependencyGraph(version=compute_sha256(commit)[:16])
    for rec in records:
        graph.add_node(rec.path, language=rec.language, sha=rec.sha256)
    for rec in records:
        for imp in rec.imports:
            if not imp.startswith("."):
                continue
            base = Path(rec.path).parent
            for ext in ("", ".js", ".mjs", ".cjs", ".ts", "/index.js"):
                candidate = str((base / (imp + ext)).as_posix())
                if candidate in all_paths:
                    graph.add_node(candidate)
                    graph.add_edge(rec.path, candidate, "IMPORTS", "static-import", .9, imp)
                    break
        for test in rec.related_tests:
            graph.add_node(test)
            graph.add_edge(test, rec.path, "TESTS", "test-heuristic", .7)

    if condition == "D":
        neighborhood = graph.impact_neighborhood(selected[:1], radius=2, max_nodes=max_files)
        selected = list(dict.fromkeys(selected + neighborhood["nodes"]))[:max_files]

    chunks = []
    symbols: List[str] = []
    for path in selected:
        source = snapshot.read(commit, path)
        if source is None:
            continue
        rec = _snapshot_record(path, source, all_paths)
        symbols.extend(rec.symbols)
        pack = build_pack(rec, graph.edges_for(path), [], graph.callers_of(path), graph.callees_of(path))
        payload = asdict(pack)
        payload["exact_source"] = source[:max_chars_per_file]
        chunks.append("FILE_PACK:\n" + json.dumps(payload, ensure_ascii=False))

    edges = [e for e in graph.to_dict()["edges"] if e["src"] in selected or e["dst"] in selected]
    if condition == "D":
        chunks.append("DEPENDENCY_GRAPH_NEIGHBORHOOD:\n" + json.dumps(edges, ensure_ascii=False))
    return BuiltContext("\n\n".join(chunks), selected, edges, sorted(set(symbols)))
