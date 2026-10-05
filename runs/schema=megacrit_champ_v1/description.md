# megacrit_champ_v1

Reconstructed start-of-Champ-fight state (deck with upgrades, relics, HP, max HP) of human Ironclad A20 runs from
`megacrit_runs_v1`, one row per Champ fight, plus whether the fight was won. Contract: [schema.py](schema.py).
Tool: [`apps/megacrit_dump`](../../apps/megacrit_dump/README.md). Join: `source_run_id` + `play_id` -> `megacrit_runs_v1`
`runs.play_id`. Use `exact` rows for deck statistics; non-exact rows still have a correct outcome and HP.
