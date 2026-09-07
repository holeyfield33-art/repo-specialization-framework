"""Apply model patches to disposable historical worktrees and run real tests."""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .history_tasks import TaskExample


@dataclass
class PatchVerification:
    applies: bool
    tests_run: bool
    tests_passed: bool
    security_regressions: int
    stdout: str
    stderr: str


def verify_patch(
    repo: Path,
    task: TaskExample,
    patch: str,
    test_command: str,
    timeout_seconds: int = 900,
) -> PatchVerification:
    if not patch.strip() or not task.context_commit:
        return PatchVerification(False, False, False, 0, "", "missing patch or context commit")

    with tempfile.TemporaryDirectory(prefix="rsef-worktree-") as tmp:
        worktree = Path(tmp) / "repo"
        add = subprocess.run(
            ["git", "-C", str(repo), "worktree", "add", "--detach", str(worktree), task.context_commit],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if add.returncode:
            return PatchVerification(False, False, False, 0, add.stdout, add.stderr)
        try:
            patch_file = Path(tmp) / "candidate.patch"
            patch_file.write_text(patch, encoding="utf-8")
            applied = subprocess.run(
                ["git", "-C", str(worktree), "apply", "--whitespace=nowarn", str(patch_file)],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            if applied.returncode:
                return PatchVerification(False, False, False, 0, applied.stdout, applied.stderr)

            # Reuse already installed dependencies without mutating the historical
            # checkout.  The Colab instructions install target dependencies once.
            source_modules = repo / "node_modules"
            target_modules = worktree / "node_modules"
            if source_modules.is_dir() and not target_modules.exists():
                os.symlink(source_modules, target_modules, target_is_directory=True)

            tested = subprocess.run(
                ["bash", "-lc", test_command], cwd=worktree, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_seconds,
            )
            passed = tested.returncode == 0
            return PatchVerification(
                True, True, passed, 0 if passed else 1,
                tested.stdout[-20_000:], tested.stderr[-20_000:],
            )
        except subprocess.TimeoutExpired as exc:
            return PatchVerification(True, True, False, 1, str(exc.stdout or ""), "test timeout")
        finally:
            subprocess.run(
                ["git", "-C", str(repo), "worktree", "remove", "--force", str(worktree)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
