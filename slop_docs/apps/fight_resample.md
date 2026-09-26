# App spec: `apps/fight_resample/`: more samples of stored fights

## Purpose

Cheap extra training data for one fight (e.g. Slime Boss) from real decks. Each stored fight of a bootstrap
run is rebuilt at its start (replaying the run: under 1 ms), then played again `samples` times by the teacher
with the same deck, relics and potions, but a new starting HP and a fresh fight RNG (draw order, monster HP
and AI, ...). Output: a `combat_v3` run whose rows have `source_episode_id` set.

```
combat_v3 bootstrap runs ──replay to each queried fight──▶ fight_resample ──▶ combat_v3 run (resampled fights)
```

## Usage

```bash
./apps/fight_resample/run.sh apps/fight_resample/slime.toml [--scratch] [--overwrite]
```

`[run]` keys (see `slime.toml`): `id`, `query` (the fights to resample, `sts_combat_rl.query`, e.g.
`select * from combat_v3 where id like 'act1-a20%' and encounter = 'slime_boss'`; they must come from bootstrap
runs, which have no inputs of their own), `samples` (per source fight, ≤ 1000), `hp_sd`, `random_potions` (default false), `workers`, optional
`fights` (first N queried fights by episode_id), optional `value_run` (value-net leaf; default: the bootstrap
guided-rollout teacher), optional `simulations` / `particles` (positive integers; default 15000 / 8) and
`random_move` (default true), optional `first_sample` (default 0: samples are k = first_sample .. first_sample + samples - 1, so
a later run with a higher first_sample plays new versions of the same fights), optional `stop_factor` (in (0, 1];
default 1) and `merge_identical_cards` (default false): opt-in search variants, see `slop_docs/search_perf.md`
(`stop_factor = 0.25`: ~20% fewer simulations, the same move at 273/275 test states, `root_value` ~0.006 lower, which
does not matter for terminal-labelled training). Oracle off. Defaults = bootstrap.

## Sample k of source fight `source_episode_id`

| thing | value |
|---|---|
| `episode_id` | `source_episode_id * 1000 + k` (fits int64: run seeds < 2^40) |
| `run_seed`, `fight_index`, `floor`, ... | the source's |
| `starting_hp` | `round(random.Random(episode_id).gauss(source starting_hp, hp_sd))`, clipped to [1, max HP], set before combat. The row column is HP at the first decision, so combat-start healing (e.g. Blood Vial) can make it higher |
| fight RNG | `GameContext` before the fight with `seed = episode_id`, `miscRng = potionRng = Random(episode_id)`; `BattleContext::init` derives every combat RNG from those |
| potions | the source's; with `random_potions`, each held potion becomes `returnRandomPotion(potionRng)` (the RNG above) |
| random move | if `random_move` (default): as bootstrap, `mt19937_64(episode_id ^ 0xe9510)`; else every move is the search's |

So a resampled fight replays from stored data like any other: replay the source run to `fight_index`, build the
game as above with `starting_hp`, then step its `chosen_action`s. Source fights must have
`source_episode_id IS NULL`, so resampled fights are never resampled again.

## Files

`worker.cpp` (replay + play all samples of one source fight), `generate.py` (loads source fights, samples HP,
runs workers, writes `part-<source_episode_id>.parquet` + `summary.json`), `run.sh` (apps/common/launch.sh;
builds `build/resample/`).
