# Repository Guidelines

## Project Structure & Module Organization
- `scripts/`: main entry points and config
  - `train.py`, `infer.py`, `merge.py`, `baseline_inference.py`, `config.py`
- `examples/`: example notebooks and sample outputs
- `out-llama-lora/`: generated artifacts (e.g., `adapter/`, `merged/`, `tokenizer/`)
- `pyproject.toml`, `uv.lock`: dependency and tool configuration (Python 3.12)
- `README.md`: quickstart and model access notes

## Build, Test, and Development Commands
- Install deps: `uv sync`
- Lint: `uv run ruff check .`
- Format: `uv run ruff format`
- Train: `python scripts/train.py`
- Inference (adapter): `python scripts/infer.py "日本語で自己紹介して"`
- Baseline inference (no adapter): `python scripts/baseline_inference.py`
- Merge adapter → base: `python scripts/merge.py`

Each script reads settings from `scripts/config.py` (model, dataset split, output dirs).

## Coding Style & Naming Conventions
- Style: PEP 8, 4-space indentation, f-strings for formatting.
- Naming: `snake_case` for files/functions, `UPPER_CASE` for constants in `config.py`.
- Imports: standard lib → third-party → local (as in `scripts/*`).
- Configuration: do not hardcode; update `scripts/config.py` and document changes in `README.md`.

## Testing Guidelines
- No formal test suite yet; use smoke runs:
  - Small split training (default `train[:1%]`) via `scripts/train.py`.
  - Prompt checks via `scripts/infer.py` and `scripts/baseline_inference.py`.
- If adding tests, place under `tests/`, name `test_*.py`, and prefer `pytest` with quick, CPU-friendly cases.

## Commit & Pull Request Guidelines
- Commits: follow Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`). Example: `feat: add DoRA option in config`.
- PRs must include:
  - Clear summary and rationale, linked issues (e.g., `Fixes #12`).
  - What changed, how to run (commands), and expected outputs/logs.
  - Note resource needs (CPU/GPU) and any config updates.
  - Screenshots or logs where helpful; update `README.md` if UX/commands change.

## Security & Configuration Tips
- The base model is gated; run `huggingface-cli login` after access approval.
- Never commit secrets or tokens; avoid pushing large model artifacts in `out-llama-lora/` unless intended.
