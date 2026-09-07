# Changelog

## Unreleased

### Changed

- Replaced simulated A/B/C/D scoring with real Hugging Face generation and raw task-output capture.
- Replaced adapter placeholders with CUDA-only QLoRA training and required PEFT adapter artifacts.
- Made Conditions A/B execute and persist before training, followed by adapter-backed Conditions C/D.
- Ingest the complete reachable target history instead of truncating at 200 commits.
- Build evaluation context from each held-out commit's parent snapshot rather than current `HEAD`.
- Ground impact, test-impact, localization, co-change, review, and patch tasks in real commits.
- Apply generated patches in disposable worktrees and run the configured target test command.
- Export actual token counts, generation latency, file recall, API hallucinations, unnecessary edits, test results, and invariant regressions.

### Fixed

- Corrected the contamination audit's inverted real-commit filtering.
- Removed synthetic/current-HEAD evaluation examples that violated the real held-out-family requirement.
- Removed the hardcoded source-SHA verification success.
- Removed committed simulated result files so they cannot be mistaken for measured evidence.
- Use the validation split during QLoRA training.
- Fail closed when CUDA, a real adapter, a frozen target HEAD, valid prepared data, or complete condition outputs are missing.
- SHA-256 lock the prepared manifests and split files so the second model uses byte-identical tasks.

### Documentation

- Added exact Colab commands for the Qwen primary run and SmolLM3 repeat using identical frozen splits.
- Documented required artifacts and completion criteria.
