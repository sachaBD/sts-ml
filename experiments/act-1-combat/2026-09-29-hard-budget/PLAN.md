# Hard-pool fights: how much HP does each MCTS budget cost? (DRAFT; superseded by README.md: 1,000 fights, 500–10k × 2 seeds, no 20k, ~16 min)

> Follows [../2026-09-29-easy-cheap-agent/](../2026-09-29-easy-cheap-agent/README.md). Replaces the stop-rule design
> in [docs/research/combat_headroom/hard/PLAN.md](../../../docs/research/combat_headroom/hard/PLAN.md) (oracle and X0 dropped).

**Why:** macro research needs cheap combat. Hard-pool fights (every regular fight after a run's first 3; about 1.6
per act-1 run) are harder than easy ones: 4–16 HP per fight and 0.3–3.4% deaths at the harness's 2k simulations.

**Question:** how much HP (and how many deaths) does each MCTS budget from 500 to 20k cost, compared with 20k?

## Headline metric (agreed before running)

- **HP lost per fight** = starting HP − final HP; a death counts as losing all starting HP.
- **Death rate** is reported next to it, per arm.
- Δ = paired per-fight difference vs MCTS 20k.
- **Pooled:** encounters weighted by how often they occur in real runs (the `act1-eval-mcts-a20` counts). Per run =
  pooled Δ × 1.6.

## What we know (stored data: 2k simulations, `act1-all-bosses-a20-scaled-search`)

| encounter | share of hard fights | HP lost (won) | deaths | turns | decisions |
|---|---:|---:|---:|---:|---:|
| Looter | 13.6% | 5.4 | 0.3% | 2.0 | 9 |
| Blue Slaver | 12.8% | 6.7 | 0.6% | 2.1 | 11 |
| Two Fungi Beasts | 12.5% | 4.0 | 0.3% | 1.9 | 10 |
| Exordium Thugs | 11.2% | 12.7 | 2.0% | 2.9 | 14 |
| Large Slime | 10.8% | 8.3 | 1.5% | 3.4 | 15 |
| Three Louse | 10.0% | 6.9 | 0.7% | 2.0 | 10 |
| Exordium Wildlife | 9.5% | 8.6 | 0.8% | 2.4 | 12 |
| Lots of Slimes | 6.9% | 10.6 | 1.3% | 2.9 | 14 |
| Gremlin Gang | 6.4% | 15.5 | 3.4% | 3.2 | 15 |
| Red Slaver | 6.4% | 7.4 | 1.1% | 2.1 | 10 |

The dev buckets (0–1) have 253–488 fights per encounter; the confirm buckets (2–3) have 226–502.

## Design

- **Fights:** 150 per encounter × 10 = 1,500 (~1,400 after diverged replays), from buckets 0–1, chosen by hash and
  frozen in `fights.csv`. `confirm.csv`: the same from buckets 2–3, played only at stage 3.
- **Stage 1 (budget sweep):**

| arm | seeds | why |
|---|---|---|
| MCTS 20k | 0–3 | reference, plus the 20k-vs-itself noise check |
| MCTS 500 / 1k / 2k / 5k / 10k | 0–1 | the budget curve; 2k is the current harness setting |

- **Gate:** report the table and Fig 1 right after stage 1. Nothing else runs without approval.
- **Stage 2 (only if stage 1 shows 10k → 20k still gaining):** MCTS 50k, seeds 0–1, to check whether 20k is itself
  saturated.
- **Stage 3 (confirmation):** MCTS 20k plus the 2 candidate budgets on `confirm.csv`, seeds 0–1.
- **Dropped:** value/policy nets and particle counts (all worse than plain MCTS on easy fights), and the oracle.

## Expected precision (to check against the stage-1 noise check)

- Pooled Δ: ±0.2–0.3 HP (95% CI), assuming a per-fight SD of ~4–6 HP with 2 seeds. Deaths add most of the variance.
- Per encounter (n ≈ 140): ±0.7–1 HP. Enough to spot the expensive encounters, not to rank close ones.
- Deaths: ~1.5% × 1,400 × 2 seeds ≈ 40 per arm. Only differences of ≳ 1 point in death rate are detectable.

## Workflow (lessons from the easy study)

1. The metric and the headline table are fixed up front (above).
2. **Pilot:** 20 fights at 20k (~2 min) → a runtime estimate → approval before stage 1.
3. **Idle machine:** check that no other job is running before each stage. Log the load per run; timing only comes
   from clean runs.
4. Report right after stage 1 (table and Fig 1), covering the full budget range. Each extra stage is justified first.
5. `report.py` from the easy study, made generic (category, encounter weights, death rate).

## Cost (estimate; the pilot replaces it)

- Assumes ~3 s per fight at 20k (easy fights: 1.75 s; elites: 4.6 s).
- Stage 1: ~50 min.
- Stage 2: ~35 min.
- Stage 3: ~20 min.
