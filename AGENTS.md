# Repository Guidelines

## Project Structure & Module Organization
- `cs336_basics/`: Source package (e.g., `pretokenization_example.py`).
- `tests/`: Pytest suite, fixtures, and reference snapshots in `tests/_snapshots/`.
- `pyproject.toml`: Tooling and dependency configuration (managed by `uv`).
- `make_submission.sh`: Runs tests and builds `cs336-spring2025-assignment-1-submission.zip`.
- `README.md` and assignment PDF: Setup, data instructions, and context.

## Build, Test, and Development Commands
- Run any script: `uv run <python_file>` (auto-solves env).
- Run tests (verbose): `uv run pytest -v`.
- Run a single test: `uv run pytest tests/test_model.py::test_transformer_block`.
- Create submission: `bash make_submission.sh`.

## Coding Style & Naming Conventions
- Python ≥ 3.11, 4‑space indentation, line length 120.
- Prefer type hints (uses `jaxtyping` for tensor shapes).
- Names: modules/functions `snake_case`, classes `CamelCase`, constants `UPPER_CASE`.
- Linting: `ruff` configured in `pyproject.toml` (UP rules enabled; some ignores on `__init__.py`). If installed, run `ruff check .`.

## Testing Guidelines
- Framework: `pytest` (configured in `pyproject.toml`).
- Naming: files `tests/test_*.py`, tests `def test_*`.
- Snapshots: do not edit files in `tests/_snapshots/`.
- Run full suite: `uv run pytest`; target failures first, then re‑run specific tests.
- Optional examples:
  - Filter by keyword: `uv run pytest -k rope`.
  - Print during tests: configured with `-s` via `pyproject.toml`.

## Commit & Pull Request Guidelines
- Commits: concise, imperative mood (e.g., "Fix RoPE typing"), group related changes. Version bump commits appear as "Version X.Y.Z: ...".
- PRs: include "what" and "why", link issues if any, and describe test coverage. Add reproduction steps or logs for bug fixes.
- Keep changes focused; avoid committing datasets or large binaries. Use `.gitignore` patterns already provided.

## Security & Configuration Tips
- Data: download to `data/` per README; do not commit.
- Environments: prefer `uv` (`uv run ...`) for reproducibility.
- Submission: use `make_submission.sh` to validate and package only required files.

## AI Assistance Policy
- 作业要求仅可就概念性或低层编码问题向 LLM 咨询；不得直接请求或使用 AI 给出完整解答代码。
- 建议关闭基于 AI 的自动补全（如 Cursor、Copilot），以确保对实现细节的独立思考。
