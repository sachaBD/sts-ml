# Human-derived Champ benchmark (one-off v1)

Entry point: `.venv/bin/python -m apps.human_champ.bench {prepare,play,report}`.
No new native worker or model architecture. Starts reuse combat_v4; completed agent
plays are combat_v4 and are replay-checked by the existing `pv_worker`.

## Prepared local dataset

`runs/schema=human_champ_bench_v1/date=2026-10-05/id=v1/out/starts.parquet`

Source: `megacrit_runs_v1/2026-10-05/pull2020-a`. 981 Champ fights -> 435 exact-deck,
supported loadouts -> 870 starts. Human wins: 700/981 source fights, 208/435 retained
decks. These are descriptive corpus counts, not population skill estimates. The
large difference is a selection effect: post-win changes are harder to undo.
Do not compare the 700/981 human rate to agents on retained decks.

## Commands

Run from repo root, with `PYTHONPATH=.`. Managed runs capture commits and logs.

```sh
.venv/bin/python -m runs.run human_champ_bench_v1 <id> --no-compact \
  --input megacrit_runs_v1/2026-10-05/pull2020-a -- \
  .venv/bin/python -m apps.human_champ.bench prepare \
  --runs runs/schema=megacrit_runs_v1/date=2026-10-05/id=pull2020-a --out '{out}'

# MCTS, max 8 simultaneous fights. Add --limit 10 for five decks / two seeds.
.venv/bin/python -m runs.run combat_v4 <teacher-id> --no-compact \
  --input human_champ_bench_v1/2026-10-05/v1 -- \
  .venv/bin/python -m apps.human_champ.bench play --starts <starts.parquet> \
  --agent teacher --sims 20000 --workers 8 --out '{out}'

# Run sequentially after MCTS to keep the combined budget at 8 workers.
.venv/bin/python -m runs.run combat_v4 <pv-id> --no-compact \
  --input human_champ_bench_v1/2026-10-05/v1 -- \
  .venv/bin/python -m apps.human_champ.bench play --starts <starts.parquet> \
  --agent pv --sims 2000 --workers 8 \
  --model runs/schema=combat_v4/date=2026-10-05/id=champ-diag-D5/model/model.onnx --out '{out}'

.venv/bin/python -m apps.human_champ.bench report --starts <starts.parquet> \
  --teacher <teacher-out> --pv <pv-out> --out <benchmark-run>/out/report
```

Play isolates each fight in its own process: native crashes/timeouts are recorded,
not silently counted as losses. `results.jsonl` is flushed after each result and
includes status, actions, search telemetry, and errors; `fights-0.parquet` contains
completed plays. Directly rerunning `play` with identical settings resumes its
journal; managed `runs.run` IDs must be fresh. Default timeout is 600 seconds per
fight. Report only pairs decks with both seeds completed by both agents. Capped
and failed fights remain visible in play summaries and are excluded identically
from the paired comparison.

Report includes deck-level means ± one SE, paired PV-minus-MCTS difference,
Demon Form / scaling-card-count / HP-band splits, and human wins lost by agents.
`viewer_starts.parquet` contains up to 20 decks (40 starts) the human won and both
agents lost on both tested seeds. Their fight IDs join the completed combat_v4
files, which contain action sequences for replay/viewing.

## Caveats (guidance only)

- No potions. Runic Dome skipped for both agents. Lizard Tail and unknown
  bottled-card / accumulated-card-damage states skipped, not guessed.
- Exact **deck contents/upgrades** do not imply exact historical fight state.
  Deck order is canonical; RNG begins from the seeds; unknown cyclic relic
  counters reset to zero. Girya and Du-Vu Doll are reconstructed explicitly.
- Missing RNG state is treated as variation; many decks reduce random noise,
  but cannot erase reconstruction selection or potion/version bias.
- Modern simulator rules, no balance corrections in v1. Most source fights use
  build `2020-07-30`, predating v2.2. Mega Crit's official
  [v2.2 patch notes](https://steamstore-a.akamaihd.net/news/externalpost/steam_community_announcements/3856733252019104438)
  list relevant Bloodletting, Hemokinesis, Rupture+, and colorless changes.
  `changed_cards` is provenance only; no balance-based exclusions or analysis.
- Scaling count = copies of Demon Form, Inflame, Spot Weakness, Limit Break.
  HP bands use pre-init HP/max HP: low <50%, mid 50–<80%, high >=80%.
- One human observation per deck, two agent seeds; SE is across per-deck means,
  not independent fight starts. Tiny subgroups are descriptive, not tested effects.

Tests: `.venv/bin/python -m apps.human_champ.test_bench` (also auto-discovered by CTest).
