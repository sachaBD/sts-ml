# Combat inspector

Inspect a replayed fight independently of training. The native inspector and browser viewer use the same
current replay, search, encodings and graph serialization. Inspection reruns search; it does not retrieve
training-time searches.

## Browser / GitHub Pages

The **Gumbel tree viewer** plays a fight, lets you try other legal moves, and browses the search graph. Each
successor links to a public state; shared states and cycle/bootstrap edges are not expanded into a fake tree.
The URL saves the checkpoint identity, fight, moves, position and search settings.

The site is a frozen demo of **one exported checkpoint** and selected fight starts. All inference and simulation
run in a worker in the browser, without a Python server, GPU, isolation headers, or external model hosting.
Both `entity` and `inventory_v1` encodings are supported via the checkpoint's `network.toml`.

From the repository root, with `.venv` prepared (`make env`) and Emscripten activated:

```sh
source /path/to/emsdk/emsdk_env.sh
make pages-runtime                         # pinned ONNX Runtime Web, downloaded under build/
make pages-build CHECKPOINT=/path/to/out/training/iter010 FIGHTS=/path/to/out/inputs/test.jsonl
make pages-preview                         # http://localhost:8734/gumbel-tree/
# After inspecting the prepared demo:
make pages-publish
```

`CHECKPOINT` must contain `network.toml` and the dynamic fp32 ONNX model and its weights. `FIGHTS` is JSONL
with `fight_id` and `start`. By default eight fights are spread across that file; use
`.venv/bin/python -m apps.combat_inspector.pages build --help` to change the selection count or artifact paths.
Set `LIGHTSPEED=/path/to/lightspeed` for a nonstandard simulator checkout.

`pages-build` compiles the **current source**, not the obsolete published simulator. It writes
`build/gumbel-site/`; no build or preview command publishes anything. The deployment includes model weights,
WASM, runtime and fight starts on GitHub Pages itself. Its manifest records the checkpoint digest, artifact
hashes, runtime version, repository/simulator revisions and whether either checkout had local changes.
Use clean checkouts for a release. Generated binaries stay out of `main`; `gh-pages` stores the published demo.

Publishing replaces only `gumbel-tree/` and `assets/gumbel/` on `origin/gh-pages`, preserving the landing page and
other reports. It uses a normal push, so concurrent deployments are rejected rather than overwritten.
The existing public URL is https://sachabd.github.io/sts-ml/gumbel-tree/ .

Browser limits: 512 simulations, 64 particles, 512 moves. Rollouts and larger searches can be slow; Stop takes
effect after the in-flight decision. Only 32 decision responses are cached. Evaluation value includes the
configured rollout blend; the viewer does not invent separate network/rollout values, terminal win counts,
or sequential-halving rounds that the current graph does not expose.

Focused checks:

```sh
.venv/bin/python -m apps.combat_inspector.test_pages
node --experimental-default-type=module apps/combat_inspector/test_graph.mjs
```

## Native / Python

Build with `make build`, then:

```python
from apps.combat_inspector import decision

result = decision('/path/to/checkpoint', fight['start'])
```

The response includes the current public board, chosen action and search graph. Observation/action hashes
are decimal strings so JavaScript cannot lose their 64-bit precision. Saved fight starts cross JavaScript as
JSON strings and are parsed only in C++ for the same reason.
