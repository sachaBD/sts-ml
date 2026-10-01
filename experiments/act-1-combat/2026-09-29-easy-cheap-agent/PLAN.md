# Easy-pool fights: the cheapest agent that is good enough

> Pre-registered 2026-09-29, before any main run. Supersedes the stop-rule framing of
> [docs/research/combat_headroom/easy/PLAN.md](../../../docs/research/combat_headroom/easy/PLAN.md) (oracle and X0 dropped).
> Measurement rules: [docs/research/evaluation_methodology.md](../../../docs/research/evaluation_methodology.md).

**Why:** macro research (card picking first) needs to play very many fights. A cheap combat agent for the easy pool
(and, by extension, other fights) speeds that up and frees compute in evaluations. Prior evidence: on floor-1 easy
fights MCTS is flat from 1k to 100k simulations; the act-1 evaluation already plays easy fights at 500.

**Question:** what is the cheapest fair agent (in CPU time per fight) that plays easy-pool fights about as well as
MCTS 20k?

**Goal (a target, not a hard gate):** pooled paired HP-eq ≥ −0.5 per fight vs MCTS 20k, no encounter worse than
−1.0, and no more deaths than the A/A noise (MCTS 20k vs itself).

## Fights

`fights.csv`: 250 per encounter (Cultist, Jaw Worm, Small Slimes, Two Louse) = 1,000, from buckets 0–1 of
`act1-all-bosses-a20-scaled-search`, chosen by SHA-256 of episode_id, all three easy fights of a run (natural mix,
~1/3 each of easy fight 1/2/3). `confirm.csv`: the same from buckets 2–3, untouched until the confirmation.

## Arms (value_play, fair, 8 particles unless stated, `search_salt` = seed)

| stage | arms | seeds |
|---|---|---|
| S1 reference | MCTS (guided rollout) 20k | 0–3 (A/A) |
| S1 sweep | MCTS 10 / 25 / 50 / 100 / 200 / 500 / 1k / 5k | 0–1 |
| S2 nets | gen1 value net (`act1-gen1`) 50 / 200 / 1k; t5 policy net (elite-trained priors) 10 / 50 / 200 | 0–1 |
| S2 knobs | particles 1 / 2 / 4 at the knee and at 500 | 0–1 |
| S3 confirm | the pick, MCTS 20k, (MCTS 500 = current eval setting) on `confirm.csv` | 0–1 |

The **knee** is the lowest S1 budget whose pooled point estimate is ≥ −0.5 HP vs the reference.

## Measurements

- Paired HP-eq difference vs the reference (each fight scored by its mean over seeds), 95% CI by cluster bootstrap
  over source run_seed (10,000 resamples); per encounter, per easy-fight number (1/2/3), pooled (encounters equal
  weight). Win rate and deaths per arm.
- **Cost:** `search_seconds` per fight (wall time in move choosing only, measured inside the worker; excludes the
  process start, run rebuild and row recording) → fights per CPU-second; also simulations used per fight and the
  worker's full wall time per fight. All arms run with 11 workers on the same 6-core/12-thread machine, so times are
  per-thread under full load (a realistic throughput setting, not a single-thread best case).
- **Charts (report):** density (PDF) of HP lost per fight by arm; density of the paired difference vs the reference;
  density of search time per fight; the cost-vs-quality frontier (HP-eq Δ vs fights per CPU-second, log scale).

## Pick and confirm

Pick = the cheapest arm meeting the goal on the dev set. Confirm once on `confirm.csv`: it passes if its pooled
95% CI lower bound vs MCTS 20k is above −0.5 HP.

## Caveats known up front

- The search is not the whole cost of a fight in a macro harness (state encoding, simulator stepping outside the
  search, process overhead). `search_seconds` isolates the part the agent choice changes.
- The t5 policy net was trained on elite fights only; its priors on easy fights are out of distribution.
- `stop_factor` changes nothing at ≤ 500 simulations (the early-stop check runs every 500), so it's not an S2 knob.
