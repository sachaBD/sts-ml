# agents/overworld/value/architectures/

Card-pick policy + run value networks (agents 3+4 of docs/research/card-policy-loop/README.md), one spec per file:
`kind` (class in `agents/overworld/value/<kind>.py`) + constructor args. Same freezing rule as
`models/combat_outcome/architectures/`: once a spec is trained it never changes; add a new spec or kind instead. Training code must
record `name<TAB>sha256<TAB>run` in `FROZEN.tsv` here when first training a spec. The learner supports
`--arch-spec PATH` (TOML `kind` plus `[args]`) for new models; `--init` reloads the checkpoint's original architecture.

- `rp1-w32-h64-l1-d30`: `run_policy_v1` sized like the best outcome model (co2-w32-h64-l1-d30).
- `rp3.1-w64-h128-l2`: `run_policy_v3_1`, backward graph map + pooled cards/relics. Untrained.
- `rp3.2-w64-h128-l2`: `run_policy_v3_2`, same graph + one card/relic/global attention layer. Untrained.
