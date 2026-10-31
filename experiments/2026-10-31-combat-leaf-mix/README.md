# Combat leaf-mix benchmark

Status: implementation approved; execution awaits approval. No new fights have been run.
Branch: `benchmark/combat-leaf-mix`. Deliver changes through a PR for approval.

## Question

How does inference-time leaf mixing affect the frozen entity network on its ten training decks?
Requested variants: 75% NN / 25% guided rollout (`rollout_mix=0.25`), and
100% NN (`rollout_mix=0`). Reference: 50% NN / 50% rollout.
Hold other search parameters fixed: 256 simulations, eight particles, evaluation noise zero.
This is a trained-on-deck diagnostic, not unseen-deck generalization or a comparison of independently trained agents.

## Located artifacts (absolute local paths)

- Frozen starts: `/home/sborowsk/project/sts/combat_rl/runs/benchmark/entity-ten/eval.jsonl`
- Historical baseline outcomes: `/home/sborowsk/project/sts/combat_rl/runs/benchmark/entity-ten/baselines.json`
- Setup/provenance notes: `/home/sborowsk/project/sts/combat_rl/runs/benchmark/entity-ten/README.md`
- Training root: `/home/sborowsk/project/sts/worktrees/nn-topologies/runs/schema=gumbel_nn_combat_v1/date=2026-10-30/id=gumbel-nn-champ-entity-ten/out/training/`
- Latest complete model: training root + `iter133/model.pt` (ONNX exports alongside).
- Latest evaluated model: training root + `iter132/model.pt`.
- Frozen NN reference outcomes: training root + `test/iter132/results.jsonl`.

All three outcome maps have the same 500 unique fight IDs: 50 fights/deck.
The NN reference is checkpoint 132, NOT checkpoint 133. No checkpoint-133 frozen evaluation was found.

## Existing outcomes (wins / 50 per deck)

| Deck | Historical MCTS20k | Historical PUCT | NN50 checkpoint 132 |
|---|---:|---:|---:|
| apparition-block | 26 | 32 | 46 |
| perfected-strike | 5 | 10 | 11 |
| clash | 15 | 13 | 13 |
| corruption-fnp | 5 | 15 | 11 |
| fire-breathing | 18 | 13 | 17 |
| limit-break-rage | 19 | 24 | 28 |
| barricade-searing | 20 | 50 | 50 |
| exhume-immolate | 21 | 23 | 49 |
| demon-combust | 23 | 33 | 30 |
| panache | 23 | 24 | 33 |
| **Total / 500** | **175 (35.0%)** | **237 (47.4%)** | **288 (57.6%)** |

These are descriptive cached counts, not newly executed benchmarks. Uncertainty has not yet been calculated.
Historical simulator-version confounds are explicitly noted in the original setup README;
MCTS20k/PUCT are historical references, not controlled current-simulator algorithm comparisons.
The evaluation set was repeatedly monitored during training; it is not an untouched final test set.

## Frozen protocol

Use **checkpoint 132**, as approved, and reuse the NN50 results. Run exactly 500 fights for
NN75, then 500 for NN100, sequentially. No repeated runs, adaptive stopping, training,
re-export, or baseline replay. All artifact paths and SHA-256 hashes are in `protocol.toml`.
The canonical sweep input is the exact `test/iter132/starts.jsonl` used for the cached NN50 evaluation.

**Use the original run's frozen `out/inputs/nn_play`, not a rebuilt main binary.**
PR #18 changed rollout/root blending and RNG streams; using today's worker would confound
this comparison. The old worker also accepts model *paths* for `--gpu-bank`, unlike main's
current wrapper. `benchmark.py` therefore assembles the small historical command explicitly
instead of using the incompatible `agents.combat.nn.play` helper. Shared JSON reading and
statistics still come from the app. There are no changes to shared code or native search.

The existing ONNX exports contain the entity network; no import of the uncommitted Python
entity implementation is required. Original run metadata records revision
`1776f396f210484a910b2e8f2536f41fa33ecd6d` with a dirty source tree. Exact historical build
flags/source changes were not archived: this is binary-artifact reproducibility, **not** a
claim that the worker can be rebuilt identically from that commit. Its hash is pinned.

Environment seeds and search initialization come from each frozen start's `seed` (historical
worker `Game` initializes a separate search RNG from that seed). Same starts do not mean
identical search trajectories: the old implementation skips rollout RNG consumption at mix 0,
so NN100 also changes subsequent search RNG consumption. Preserve this historical behavior,
rather than silently changing the worker to force identical random samples.

Primary comparison: paired win-rate differences vs cached NN50, overall and per deck.
Reports use existing app statistics: Wilson 95% win-rate intervals, approximate normal
95% paired-difference intervals over 500 fights (50/deck), plus aggregate deck-level
variability across ten decks. These are descriptive exploratory comparisons; no multiplicity
correction, no claim about unseen decks, and no checkpoint selection using these new results.

Timing: measure whole native-process wall time (setup + gameplay), separately per new variant;
exclude input verification and reporting. Preserve native timing logs. One observation/variant,
so timing uncertainty is unknown. Original NN50 log reports 118.042 seconds gameplay and
12.9002 seconds evaluator setup; these historical components are context, not a same-session
speedup control. Cache state, host load, differing fight lengths and scheduling affect timing.

## Commands (execution requires separate approval)

From the benchmark worktree root, use the dependency-managed existing environment:

```sh
PY=/home/sborowsk/project/sts/combat_rl/.venv/bin/python
EXP=experiments/2026-10-31-combat-leaf-mix
$PY "$EXP/benchmark.py"            # read-only hash/coverage preflight; no GPU or fights

# ONLY after execution approval, with clean committed source and a new output directory:
$PY "$EXP/benchmark.py" --run --out runs/benchmark/leaf-mix-132-ATTEMPT
$PY "$EXP/report.py" runs/benchmark/leaf-mix-132-ATTEMPT
```

`--run` refuses existing output directories. Interrupted/failed attempts remain separate;
there is no automatic retry or resume. Per-fight outcomes, commands, copied references,
resolved protocol, hashes, source revision, installed packages, GPU/driver information,
logs and process timing remain under `runs/`. Reports refuse incomplete variants.
Native sample files are retained for inspection; no automatic destructive cleanup.
Do not overlap benchmark execution with other GPU workloads.

## Verification before PR

- The default invocation is the small, read-only preflight: verify pinned artifact hashes,
  exactly 500 unique fights (50 per deck), and matching baseline IDs with valid outcomes.
- Preflight passes against the pinned local artifacts; no combat/GPU processes launched.
- Native execution and end-to-end GPU compatibility remain deliberately unverified pending approval.
