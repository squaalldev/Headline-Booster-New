# Commit Log

This file records the development commits that show how Headline Booster was built and refactored with help from OpenAI Codex as a coding agent.

## Why this exists

The OpenAI Codex hackathon track benefits from clear evidence that Codex participated in real repository changes. The Git history is the primary source of truth, and this file provides a human-readable summary of the relevant commits.

## Branch alignment

The repository should keep `main` and `codex` pointing to the same final commit for the hackathon submission. Any older temporary work branches can be deleted after both branches are aligned.

## Current commits

| Commit | Summary | Notes |
| --- | --- | --- |
| `497c903` | Primer guardado del proyecto | Original repository checkpoint before the Headline Booster migration. |
| `9ff774d` | Replace full-stack Compartir AI with Gradio Headline Booster (add `app.py`, remove FastAPI/React stack) | Codex-assisted migration from the previous React/FastAPI/Supabase product to a focused Gradio Space with mock generation, Tiny Titan model runtime wiring, custom frontend CSS, and documentation. |
| `HEAD` | Update commit log and branch alignment notes | Updates this hackathon commit log and documents that `main` and `codex` should point to the same final commit. Use `git log --oneline -1` for the exact hash because embedding the current commit hash in this same file would change that hash. |

## Codex evidence checklist

- Real code changes were made in `app.py`.
- The repository was simplified for Hugging Face Spaces.
- External AI API dependencies were removed for the first mock version.
- Tiny Titan model integration wiring was added for `Qwen/Qwen2.5-1.5B-Instruct` with `Qwen/Qwen2.5-3B-Instruct` as fallback.
- Documentation was added in `README.md`, `docs/CODEX_NOTES.md`, `docs/FIELD_NOTES.md`, `docs/FRONTEND_REFERENCE_NOTES.md`, and `docs/TINY_TITAN_PLAN.md`.
- This commit log records the migration history for reviewers.

## How to verify locally

Run:

```bash
git log --oneline --decorate --all
```

The output should include the commits listed above plus any later fixes or documentation updates.

To verify branch alignment after cleanup, run:

```bash
git rev-parse main
git rev-parse codex
```

Both commands should print the same commit hash.
