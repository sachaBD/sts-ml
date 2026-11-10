This is a slay the spire agentic play project. 

**Primary goal:**
Produce the strongest completely autonomous agent to play Slay the Spire (sts). built by a single person, on a single machine.

# Agent notes

- Follow [`CODING_GUIDELINES.md`](CODING_GUIDELINES.md).

- Python env: `.venv`, defined by `pyproject.toml` + `uv.lock` and managed with uv. Set it up with
  `make env` (see `make help`). Never `pip install` or `uv pip install`
  into `.venv` directly; to add or change a dependency, edit `pyproject.toml` (core or a group), run `uv lock`,
  and commit both files. Run code with `.venv/bin/python` (not `uv run`, which re-syncs `.venv` first).

- All CMake build dirs live under `build/<name>/` (default: `build/main`, via `make build`).
  Need an isolated build? Use a new `build/<name>/`. Never create `build-*` dirs in the repo root.

- Organize code by purpose: `environments/combat` owns state/actions/transitions;
  `agents/combat` owns decision-making, model architecture, inference, and learning.
  Apps orchestrate workflows. The one app is `apps/gumbel_nn_combat_expert_iteration`; delete code it doesn't need.
- Tests live beside their owners. `make test` runs native and Python tests through CTest;
  do not recursively discover tests through recorded `runs/` outputs.
- `make test` is the verification bar. Do not build extra verification (GPU runs, scratch harnesses, long
  jobs) unasked; if it seems worth it, state its cost in one line and ask first.
- Keep model kinds, frozen architecture specs, checkpoint layouts, and native weight formats compatible
  when moving code. Results and saved weights belong in `runs/`, not source directories.
