# `sts_ml` Slime Boss entry exporter handoff

## Goal

Use the full-run/card-reward agent in sibling repository
`/home/sborowsk/project/sts_ml` to generate realistic Ascension 1 Ironclad
states at entry to the first boss, accepting only naturally generated Slime
Boss runs. These entry states will later define deck/HP scenarios for
`sts_combat_rl` training.

Hard requirements from the owner:

- Inspect the public Act 1 boss immediately after native reset.
- Reject non-Slime seeds before any step, strategic decision, model load, or
  combat search.
- Never replace another Act 1 boss with Slime Boss: the strategic agent sees
  the boss identity while making reward and route decisions.
- Run accepted seeds through Act 1 using the existing strategic policy and
  native MCTS for pre-boss combats.
- Stop at active Slime Boss entry before the first boss combat action/search.
- Keep the exporter separate from the sealed/frozen release runtime.

## Repository state

### `sts_combat_rl`

The completed fixed-deck neural-MCTS work is committed at:

```text
af20f04 Train and evaluate neural-guided Slime Boss MCTS
```

The working tree was clean before adding this handoff document.

### `sts_ml`

Source repository revision:

```text
1e69c2e78a5ab3cb44cca5ffdfcb82803a68701d
```

The exporter implementation is currently **uncommitted** in that repository:

```text
 M README.md
?? apps/export_slime_boss_entries.py
?? apps/slime_entry_export.py
?? tests/test_slime_entry_export.py
```

There is also a pre-existing unrelated untracked directory:

```text
?? .sts-combat-staging/
```

Do not add, delete, or modify that staging directory as part of the exporter
commit. Before exporter work, the implementation agent archived it to:

```text
/tmp/sts_ml-sts-combat-staging-pre-export.tgz
SHA256 f62c4e8018016b7405144fac5ab6f18c170703731f0fa006daef8784945c7d9e
```

That `/tmp` archive is only a temporary safety copy, not durable project data.

## Implemented design

The new dedicated application is:

```text
sts_ml/apps/export_slime_boss_entries.py
```

Its testable library is:

```text
sts_ml/apps/slime_entry_export.py
```

It deliberately does not modify or import the frozen `agent/run.py` execution
path. It accepts an explicitly supplied, locally built `slaythespire` extension.

### Early boss filter

`RLEnvironment.snapshot()["act_boss"]` is valid immediately after reset because
`GameContext` generates the Act 1 boss during construction/reset. The exporter:

1. resets the native environment;
2. checks `MonsterEncounter(snapshot["act_boss"]).name`;
3. returns `status="non_slime"` immediately for another boss.

This path has zero steps, policy proposals, combat searches, and Slime combat
actions. Model weights are lazily loaded only after a seed passes this check.

### Accepted-run boundary

Accepted seeds run at Ascension 1 using the existing strategic policy and
native combat search. `A20HeartProcess` receives a `boundary_predicate` requiring:

- Act 1;
- active combat;
- boss room;
- active encounter equals the public Act 1 boss;
- encounter name is `SLIME_BOSS`.

The boundary intentionally does not hardcode a floor number. The simulator's
normal first-boss entry is floor 16. `StrategicSemiMDP._advance` checks this
predicate before invoking combat search, so the exporter stops before a Slime
action.

### Cross-platform model loading

The packaged release native is Windows CPython 3.13. The exporter is explicitly
labeled:

```text
cross_platform_development_extraction_not_frozen_release_reproduction
```

It verifies the frozen config, candidate checkpoint, and human-prior bytes. It
also verifies that both checkpoints declare the original frozen native hash.
A development-only human-prior loader reconstructs fresh mutable policy state
around shared verified eval weights without requiring the locally built Linux
extension to equal the Windows binary hash. Frozen runtime files remain
unchanged.

`ExporterSession` lazily loads the candidate and human model weights once per
process. Each accepted run receives fresh policy history, fallback/controller
state, and neural correction carry.

### Output contract

The CLI writes flushed JSONL incrementally and an atomic summary sidecar. Every
attempt has one of these statuses:

```text
accepted
non_slime
died_before_boss
timeout
unsupported
error
```

Accepted rows contain:

- seed, ascension, act, floor, boss and encounter;
- HP, max HP, and gold;
- deck card ID, symbolic name, upgrade flag/count, and misc data;
- order-invariant canonical deck signature;
- relic IDs/names/counters;
- potion IDs/names/slots and capacity;
- bottled card indices;
- a strict public snapshot projection and hash;
- compact semantic strategic choice history;
- counters proving no Slime search/action;
- source/model/local-native provenance.

Hidden draw order, combat piles, simulator RNG, `transform_rng`, second-boss
identity, and stale decision internals are excluded.

## Validation completed

From `/home/sborowsk/project/sts_ml`:

```sh
ruff check \
  apps/slime_entry_export.py \
  apps/export_slime_boss_entries.py \
  tests/test_slime_entry_export.py

python3 -m unittest discover -s tests -p 'test_slime_entry_export.py'
```

Results:

```text
Ruff: passed
Unit tests: 7 passed
Python compile: passed
git diff --check: passed
```

Tests cover:

- non-Slime rejection before policy/process;
- accepted Slime boundary with zero Slime search/action;
- separate death, unsupported, timeout, and error statuses;
- NumPy-safe canonical serialization;
- hidden combat data exclusion;
- order-invariant deck signatures;
- frozen model-byte preflight.

## Current blocker

No Linux native extension has been built yet.

Incremental build/configuration log:

```text
/home/sborowsk/project/sts_ml/runs/slime-entry-build.log
```

CMake first required this compatibility option because of old vendored CMake
minimum versions:

```text
-DCMAKE_POLICY_VERSION_MINIMUM=3.5
```

A clean configure forced the required interpreter:

```text
/home/sborowsk/project/sts_combat_rl/.venv/bin/python3
Python 3.13.13
```

Configuration then failed because the machine lacks CPython 3.13 development
headers:

```text
/usr/include/python3.13/Python.h  # missing
```

Installed packages include Python 3.13, stdlib, and venv, but not
`python3.13-dev`/`libpython3.13-dev`. Apt currently advertises 3.13.15 packages,
so installing them may also upgrade the interpreter from 3.13.13. The project
owner may need to run something like:

```sh
sudo apt install python3.13-dev
```

Review the proposed package upgrade before doing so. This session did not have
passwordless sudo and installed nothing.

Do not silently fall back to a CPython 3.10 native. System Python 3.10 has
headers but does not have PyTorch installed, while the verified strategic
models need PyTorch. The intended clean path is one CPython 3.13 environment
for the extension and model runtime.

## Exact continuation plan

1. Install or otherwise provide matching CPython 3.13 headers and library.
2. Delete/reconfigure only the ignored build directory:

   ```text
   /home/sborowsk/project/sts_ml/build/slime-entry-linux
   ```

3. Configure Release with the compatibility policy and explicitly select
   `/home/sborowsk/project/sts_combat_rl/.venv/bin/python3`.
4. Append configure/build progress to:

   ```text
   sts_ml/runs/slime-entry-build.log
   ```

5. Run CLI `--preflight` to load the local extension and verify provenance.
6. Use native reset only to identify one non-Slime and one Slime seed.
7. Run the non-Slime seed alone and verify:
   - `status == "non_slime"`;
   - all counters are zero;
   - no model policy or process was invoked.
8. Run exactly one accepted Slime seed with small pre-boss budgets, e.g.:
   - ordinary: 64;
   - dangerous: 256;
   - boss budget is irrelevant because the boundary must stop first.
9. Write live logs to:

   ```text
   sts_ml/runs/slime-entry-nonslime-preflight.log
   sts_ml/runs/slime-entry-accepted-preflight.log
   ```

   and output to:

   ```text
   sts_ml/runs/slime-entry-preflight.jsonl
   ```

10. Validate the accepted record:
    - natural Slime Boss;
    - Act 1 and normal first-boss floor;
    - zero Slime searches/actions;
    - full deck/HP/relic/potion fields;
    - no hidden draw pile/RNG fields;
    - stable public/deck hashes.
11. Run the new unit tests and relevant existing `sts_ml` tests.
12. Commit only README plus the three exporter/test files in `sts_ml`.
13. Do not generate a large corpus until the preflight is accepted.

## Downstream work not yet implemented

`sts_combat_rl` does not yet import these records. After exporter acceptance,
add a versioned importer/parity validator there. The current combat encoding
supports only the fixed experiment's small card set and does not encode general
relic/potion effects. Do not silently strip unsupported context. Either:

- begin with an explicitly declared deck/HP-only projection; or
- extend encoding coverage before admitting full-fidelity generated scenarios.

Train/validation splits must group by canonical deck signature, not merely by
run seed.

## Coordination

The `sts_ml` implementation session used during this work was:

```text
01a0c0cd-9b7c-7133-987f-ef448775a44b
```

It was explicitly told to stop and preserve current state when this handoff was
written.
