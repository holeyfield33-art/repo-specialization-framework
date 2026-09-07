"""
1. Repository ingestion
Deterministic extraction of source files, SHAs, symbols, imports/exports,
tests mapping, and git history. No LLM used for facts obtainable from Git
or static analysis.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import git
except ImportError:
    git = None  # type: ignore


SUPPORTED_EXTENSIONS = {
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".py": "python",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".md": "markdown",
    ".json": "json",
    ".yml": "yaml",
    ".yaml": "yaml",
}

IMPORT_RE = re.compile(
    r"""(?:import\s+(?:[\w*{}\s,]+\s+from\s+)?['\"]([^'\"]+)['\"]|require\s*\(\s*['\"]([^'\"]+)['\"]\s*\))""",
    re.MULTILINE,
)
EXPORT_RE = re.compile(
    r"""(?:export\s+(?:default\s+)?(?:async\s+)?(?:function|class|const|let|var|\{)|module\.exports\s*=)""",
    re.MULTILINE,
)
FUNCTION_RE = re.compile(
    r"""(?:(?:export\s+)?(?:async\s+)?function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(|class\s+(\w+))""",
    re.MULTILINE,
)
CALL_RE = re.compile(r"""(\w+)\s*\(""", re.MULTILINE)


@dataclass
class FileRecord:
    path: str
    sha256: str
    language: str
    size_bytes: int
    imports: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    symbols: List[str] = field(default_factory=list)
    related_tests: List[str] = field(default_factory=list)
    callers: List[str] = field(default_factory=list)
    callees: List[str] = field(default_factory=list)
    content: Optional[str] = None


@dataclass
class CommitRecord:
    sha: str
    author: str
    date: str
    message: str
    files_changed: List[str]
    insertions: int = 0
    deletions: int = 0
    parents: List[str] = field(default_factory=list)


@dataclass
class ChangeFamily:
    """Primary unit of splitting. All derived examples stay inside one family."""
    family_id: str
    root_commit: str
    commits: List[str]
    files: List[str]
    pr_number: Optional[int] = None
    temporal_rank: int = 0


@dataclass
class RepoManifest:
    repo_name: str
    root_path: str
    head_sha: Optional[str]
    files: List[FileRecord]
    commits: List[CommitRecord]
    change_families: List[ChangeFamily]
    graph_version: str
    created_at: str


def compute_sha256(content: bytes | str) -> str:
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def detect_language(path: Path) -> str:
    return SUPPORTED_EXTENSIONS.get(path.suffix.lower(), "unknown")


def extract_js_symbols(content: str) -> Tuple[List[str], List[str], List[str]]:
    imports = []
    for m in IMPORT_RE.finditer(content):
        imp = m.group(1) or m.group(2)
        if imp:
            imports.append(imp)
    exports = []
    for m in EXPORT_RE.finditer(content):
        exports.append(m.group(0)[:80])
    symbols = []
    for m in FUNCTION_RE.finditer(content):
        name = m.group(1) or m.group(2) or m.group(3)
        if name:
            symbols.append(name)
    return list(dict.fromkeys(imports)), list(dict.fromkeys(exports)), list(dict.fromkeys(symbols))


def map_tests_to_source(source_path: str, all_files: List[str]) -> List[str]:
    stem = Path(source_path).stem
    candidates = []
    for f in all_files:
        fl = f.lower()
        if "test" in fl or "spec" in fl:
            if stem.lower() in fl or Path(source_path).name.lower() in fl:
                candidates.append(f)
    return candidates


def enumerate_files(root: Path, exclude_dirs: Optional[Set[str]] = None) -> List[Path]:
    exclude = exclude_dirs or {
        "node_modules", ".git", "dist", "build", "coverage",
        ".next", "__pycache__", ".venv", "venv", "target",
    }
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in exclude]
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() in SUPPORTED_EXTENSIONS or p.name in (
                "package.json", "tsconfig.json", ".eslintrc.js"
            ):
                files.append(p)
    return sorted(files)


def ingest_repository(
    root_path: str | Path,
    repo_name: str = "runtime-firewall-mvp",
    include_content: bool = True,
    max_content_bytes: int = 512_000,
) -> RepoManifest:
    root = Path(root_path).resolve()
    files_paths = enumerate_files(root)
    all_rel = [str(p.relative_to(root)) for p in files_paths]

    file_records: List[FileRecord] = []
    for p in files_paths:
        rel = str(p.relative_to(root))
        raw = p.read_bytes()
        sha = compute_sha256(raw)
        lang = detect_language(p)
        content = None
        imports, exports, symbols = [], [], []
        if include_content and len(raw) <= max_content_bytes:
            try:
                content = raw.decode("utf-8", errors="replace")
                if lang in ("javascript", "typescript"):
                    imports, exports, symbols = extract_js_symbols(content)
            except Exception:
                content = None
        related = map_tests_to_source(rel, all_rel)
        file_records.append(
            FileRecord(
                path=rel,
                sha256=sha,
                language=lang,
                size_bytes=len(raw),
                imports=imports,
                exports=exports,
                symbols=symbols,
                related_tests=related,
                content=content,
            )
        )

    commits: List[CommitRecord] = []
    head_sha = None
    if git is not None:
        try:
            repo = git.Repo(root)
            head_sha = repo.head.commit.hexsha
            for c in repo.iter_commits(max_count=200):
                files_changed = list(c.stats.files.keys()) if c.stats else []
                commits.append(
                    CommitRecord(
                        sha=c.hexsha,
                        author=str(c.author),
                        date=c.committed_datetime.isoformat(),
                        message=c.message.strip()[:500],
                        files_changed=files_changed,
                        insertions=c.stats.total.get("insertions", 0) if c.stats else 0,
                        deletions=c.stats.total.get("deletions", 0) if c.stats else 0,
                        parents=[p.hexsha for p in c.parents],
                    )
                )
        except Exception:
            head_sha = compute_sha256("".join(f.sha256 for f in file_records).encode())
            commits.append(
                CommitRecord(
                    sha=head_sha or "synthetic-head",
                    author="synthetic",
                    date="2026-01-01T00:00:00+00:00",
                    message="Synthetic baseline (no git history available)",
                    files_changed=all_rel[:50],
                )
            )
    else:
        head_sha = compute_sha256("".join(f.sha256 for f in file_records).encode())
        commits.append(
            CommitRecord(
                sha=head_sha,
                author="synthetic",
                date="2026-01-01T00:00:00+00:00",
                message="Synthetic baseline",
                files_changed=all_rel[:50],
            )
        )

    change_families: List[ChangeFamily] = []
    for i, c in enumerate(reversed(commits)):
        fid = f"family-{c.sha[:12]}"
        change_families.append(
            ChangeFamily(
                family_id=fid,
                root_commit=c.sha,
                commits=[c.sha],
                files=c.files_changed,
                temporal_rank=i,
            )
        )

    graph_version = compute_sha256(
        (head_sha or "") + "".join(sorted(f.sha256 for f in file_records))
    )[:16]

    from datetime import datetime, timezone
    manifest = RepoManifest(
        repo_name=repo_name,
        root_path=str(root),
        head_sha=head_sha,
        files=file_records,
        commits=commits,
        change_families=change_families,
        graph_version=graph_version,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    return manifest


def manifest_to_dict(m: RepoManifest) -> Dict[str, Any]:
    return {
        "repo_name": m.repo_name,
        "root_path": m.root_path,
        "head_sha": m.head_sha,
        "graph_version": m.graph_version,
        "created_at": m.created_at,
        "file_count": len(m.files),
        "commit_count": len(m.commits),
        "family_count": len(m.change_families),
        "files": [asdict(f) for f in m.files],
        "commits": [asdict(c) for c in m.commits],
        "change_families": [asdict(cf) for cf in m.change_families],
    }


def save_manifest(m: RepoManifest, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = manifest_to_dict(m)
    for f in data["files"]:
        if f.get("content") and len(f["content"]) > 2000:
            f["content"] = f["content"][:2000] + "\n...[truncated in pack]"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
