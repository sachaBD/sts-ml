# agents/overworld/value/architectures/

Card-pick policy + run value networks (agents 3+4 of docs/research/card-policy-loop/README.md), one spec per file:
`kind` (class in `agents/overworld/value/<kind>.py`) + constructor args. Same freezing rule as
`models/combat_outcome/architectures/`: once a spec is trained it never changes; add a new spec or kind instead. Training code must
append `name<TAB>sha256<TAB>run` to `FROZEN.tsv` here (no trainer exists yet).

- `rp1-w32-h64-l1-d30`: `run_policy_v1` sized like the best outcome model (co2-w32-h64-l1-d30). Untrained.
