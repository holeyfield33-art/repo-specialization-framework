import subprocess
import tempfile
import unittest
from pathlib import Path

from src.context_builder import GitSnapshot, build_condition_context
from src.evaluation import contamination_audit, deterministic_verify
from src.history_tasks import TaskExample
from src.model_runtime import _extract_json, InferenceError
from src.patch_verifier import verify_patch


def task(task_id, family, commit, split, parent, rank, ground_truth=None):
    return TaskExample(
        task_id=task_id,
        task_type="change_impact_prediction",
        family_id=family,
        commit_sha=commit,
        instruction="Inspect src/a.js",
        context_files=["src/a.js"],
        ground_truth=ground_truth or {"impacted": []},
        evidence=[],
        split=split,
        context_commit=parent,
        temporal_rank=rank,
    )


class IntegrityTests(unittest.TestCase):
    def test_deterministic_verify_compares_real_hashes(self):
        self.assertEqual(
            deterministic_verify("abc", "def", "g", "g", [], []).status,
            "FREEZE",
        )
        self.assertEqual(
            deterministic_verify("abc", "abc", "g", "g", [], []).status,
            "PASS",
        )

    def test_contamination_audit_passes_disjoint_temporal_real_families(self):
        train = [task("t", "f1", "1" * 40, "train", "0" * 40, 1)]
        evaluation = [task("e", "f2", "3" * 40, "eval", "2" * 40, 2)]
        report = contamination_audit(train, evaluation, {"f1"}, {"f2"})
        self.assertEqual(report["status"], "PASS")

    def test_contamination_audit_blocks_family_overlap(self):
        train = [task("t", "same", "1" * 40, "train", "0" * 40, 1)]
        evaluation = [task("e", "same", "3" * 40, "eval", "2" * 40, 2)]
        report = contamination_audit(train, evaluation, {"same"}, {"same"})
        self.assertEqual(report["status"], "FAIL")
        self.assertFalse(report["checks"]["family_disjoint"])

    def test_json_parser_never_invents_fallback_output(self):
        self.assertEqual(_extract_json('```json\n{"predicted_files": []}\n```')["predicted_files"], [])
        with self.assertRaises(InferenceError):
            _extract_json("not json")

    def test_context_uses_parent_snapshot_not_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
            (repo / "src").mkdir()
            (repo / "src" / "a.js").write_text("export const value = 'before';\n")
            subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
            subprocess.run(["git", "-C", str(repo), "commit", "-qm", "before"], check=True)
            parent = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
            (repo / "src" / "a.js").write_text("export const value = 'held-out-answer';\n")
            subprocess.run(["git", "-C", str(repo), "commit", "-qam", "held out"], check=True)
            commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
            example = task("e", "f", commit, "eval", parent, 2)
            context = build_condition_context(example, "B", GitSnapshot(repo))
            self.assertIn("before", context.text)
            self.assertNotIn("held-out-answer", context.text)

    def test_patch_verifier_applies_patch_and_runs_real_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
            (repo / "a.txt").write_text("before\n")
            subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
            subprocess.run(["git", "-C", str(repo), "commit", "-qm", "before"], check=True)
            parent = subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True,
            ).strip()
            patch = """diff --git a/a.txt b/a.txt
--- a/a.txt
+++ b/a.txt
@@ -1 +1 @@
-before
+after
"""
            example = task("patch", "f", "3" * 40, "eval", parent, 2, {"files": ["a.txt"]})
            result = verify_patch(repo, example, patch, "test \"$(cat a.txt)\" = after")
            self.assertTrue(result.applies)
            self.assertTrue(result.tests_run)
            self.assertTrue(result.tests_passed)


if __name__ == "__main__":
    unittest.main()
