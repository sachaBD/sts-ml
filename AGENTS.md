This is a slay the spire agentic play project. 

**Primary goal:**
Produce the strongest completely autonomous agent to play Slay the Spire (sts). built by a single person, on a single machine.

# Agent notes

- Follow [`agents/coding-guidelines.md`](agents/coding-guidelines.md).

- Python env: `.venv`, defined by `pyproject.toml` + `uv.lock` and managed with uv. Set it up with
  `make env` / `env-onnx` / `env-jax` / `env-all` (see `make help`). Never `pip install` or `uv pip install`
  into `.venv` directly; to add or change a dependency, edit `pyproject.toml` (core or a group), run `uv lock`,
  and commit both files. Run code with `.venv/bin/python` (not `uv run`, which re-syncs `.venv` first).

- All CMake build dirs live under `build/<name>/` (default: `build/main`, via `make build`).
  Need an isolated build? Use a new `build/<name>/`. Never create `build-*` dirs in the repo root.

- Organize code by purpose: `environments/{combat,overworld}` owns state/actions/transitions;
  `agents/{combat,overworld}` owns decision-making, model architecture, inference, and learning.
  Shared predictors that are not agents live in `models/`; apps orchestrate workflows.
- Tests live beside their owners. `make test` runs native and Python tests through CTest;
  do not recursively discover tests through recorded `runs/` outputs.
- Keep model kinds, frozen architecture specs, checkpoint layouts, and native weight formats compatible
  when moving code. Results and saved weights belong in `runs/`, not source directories.
