"""Shared fixtures: a real git repository built on disk, ingested through the
real pipeline. Regression tests run against real manifests/graphs/packs rather
than hand-built stubs, so a break anywhere in ingestion surfaces here."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DETECTOR_JS = """const { loadPolicy } = require('./policy');
const { selfHash } = require('./hash');

function buildAutomaton(patterns) { return patterns.map(p => p.toLowerCase()); }
function detect(input) {
  const policy = loadPolicy();
  return buildAutomaton(policy.patterns).some(p => input.includes(p));
}
module.exports = { detect, buildAutomaton };
"""

POLICY_JS = """const { selfHash } = require('./hash');
function loadPolicy() { return { patterns: ['eval(', 'child_process'], hash: selfHash() }; }
module.exports = { loadPolicy };
"""

HASH_JS = """const crypto = require('crypto');
function selfHash() { return crypto.createHash('sha256').update('policy').digest('hex'); }
module.exports = { selfHash };
"""

INDEX_JS = """const { detect } = require('./detector');
function guard(payload) { return detect(payload) ? 'BLOCK' : 'PASS'; }
module.exports = { guard };
"""


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture(scope="session")
def sample_repo(tmp_path_factory) -> Path:
    """A small multi-commit JS repo with real import edges and test files."""
    repo = tmp_path_factory.mktemp("helios-sample")
    (repo / "src").mkdir()
    (repo / "test").mkdir()
    (repo / "src" / "detector.js").write_text(DETECTOR_JS)
    (repo / "src" / "policy.js").write_text(POLICY_JS)
    (repo / "src" / "hash.js").write_text(HASH_JS)
    (repo / "src" / "index.js").write_text(INDEX_JS)
    (repo / "test" / "detector-unit-test.js").write_text(
        "const { detect } = require('../src/detector');\n"
    )
    (repo / "test" / "policy-unit-test.js").write_text(
        "const { loadPolicy } = require('../src/policy');\n"
    )
    (repo / "package.json").write_text('{"name":"helios-sample","version":"0.1.0"}\n')

    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "feat: initial runtime firewall detector")

    (repo / "src" / "detector.js").write_text(
        DETECTOR_JS.replace("p.toLowerCase()", "p.toLowerCase().trim()")
    )
    _git(repo, "commit", "-qam", "fix: normalize patterns to prevent detection bypass")

    (repo / "src" / "policy.js").write_text(
        POLICY_JS.replace("'child_process'", "'child_process', 'require('")
    )
    _git(repo, "commit", "-qam", "fix: add missing require pattern, critical coverage gap")

    (repo / "src" / "index.js").write_text(INDEX_JS + "module.exports.version = '0.2.0';\n")
    _git(repo, "commit", "-qam", "chore: bump exported version")
    return repo


@pytest.fixture(scope="session")
def pipeline(sample_repo, tmp_path_factory):
    """Manifest, graph, packs and tasks from the real pipeline."""
    from src.ingestion import ingest_repository
    from src.graph import build_dependency_graph
    from src.file_packs import generate_file_packs
    from src.history_tasks import generate_tasks_from_history

    out = tmp_path_factory.mktemp("pipeline-out")
    manifest = ingest_repository(
        sample_repo, repo_name="holeyfield33-art/runtime-firewall-mvp", include_content=True
    )
    graph = build_dependency_graph(manifest)
    packs_dir = out / "file_packs"
    packs = generate_file_packs(manifest, graph, packs_dir)
    tasks = generate_tasks_from_history(manifest, graph)
    return {
        "manifest": manifest,
        "graph": graph,
        "packs": packs,
        "packs_dir": packs_dir,
        "tasks": tasks,
        "out": out,
    }
