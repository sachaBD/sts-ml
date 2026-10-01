# Boss fights: do they behave like the elites, and how much is left?

> **Status: DESIGN ONLY, NOT ACTIONED.** Shared rules, decision rule and X0: [../README.md](../README.md).
> Measurement rules: [../../evaluation_methodology.md](../../evaluation_methodology.md).

**Hypothesis:** bosses behave like the elites:
- MCTS saturates by ~10–20k simulations;
- the oracle wins noticeably more;
- more particles don't help;
- a policy-prior net trained on teacher plus self-play data (the elite t5 recipe) beats MCTS.

**Why this matters most:** the boss causes ~285 of ~395 deaths per 1,000 act-1 runs (32% boss death rate;
Hexaghost ~43–46%). One point of boss win rate ≈ 0.9 act-clear points. Even small gains here beat everything in
easy or hard fights.

**Prior evidence** (single seed, old simulator): gen1 value net vs MCTS on 200 fights per boss: Slime Boss +5.5
[+1.9, +9.1], Guardian +1.4, Hexaghost −0.2. Full-act evaluation, matched boss fights: net +2.3 ± 1.0 HP-eq.

## Phase 1: measurement (no training)

- **Fights:** 400 per boss from buckets 0–1 of `act1-all-bosses-a20-scaled-search` (~660 per boss available; about
  1,130 fights after diverged replays). Hash-selected and frozen in `fights.csv`, plus a 150-per-boss subset for the
  expensive arms. X0 (../README.md) uses 200 per boss from the same list.
- **Arms:**

| arm | fights | seeds | purpose |
|---|---|---|---|
| MCTS 20k | all | 0–3 | reference; noise floor; per-fight win probability; pivotal list |
| oracle 20k | all | 0 | ceiling in wins and HP-eq; which fights are unwinnable |
| oracle 50k | subset | 0 | oracle convergence (boss fights are longer than elites) |
| MCTS 5k, 50k | all | 0–1 | saturation (the harness runs bosses at 15k) |
| MCTS 20k, 32 particles | pivotal | 0–3 | hidden information |
| MCTS 20k | pivotal | 4–7 | held-out reference for the pivotal stratum |
| `ab-gen1` value net 20k | all | 0–1 | demonstrated fair headroom (value-only leaf) |

- **Report** per boss:
  - win probability and HP-eq for each arm; the unwinnable / pivotal / trivial split;
  - the seed-vs-seed noise floor; fights needed for ±1 HP-eq and ±2 win points;
  - both budget curves; the 32 vs 8 particle comparison on pivotal fights using held-out seeds.

## Phase 1 decision (pre-registered)

- **"Like elites" is confirmed** if MCTS 50k − 20k < +1 HP-eq, 32 vs 8 particles is within ±1, and the oracle wins
  ≥ 5 points more than MCTS on at least one boss.
- **Go to Phase 2** if the oracle − MCTS win gap is ≥ 5 points on any boss, **or** `ab-gen1` beats MCTS on any
  boss (interval above 0).
- **Stop** combat work on bosses only if the oracle − MCTS win gap is < 2 points on every boss (the ceiling is
  small).
- **If 50k is clearly better than 20k:** raising the boss budget in the harness is an immediate, free act-level
  gain. Do that first.

## Phase 2: a boss policy net (conditional; the elite t5 recipe)

1. **Teacher data:** MCTS 20k on ~1,700 training fights per boss (buckets 4, 6–9; `fight_resample`, HP
   `hp_sd = 10`, random potions), as in the elite-v3 stage 1. Boss fights are longer, so expect ~3–4 h. Pilot the
   rate and size the run to fit.
2. **Train:** `deep_sets_v3` + policy head, with t2's settings; ~10–20 min on the GPU.
3. **Self-play:** that net with policy priors plays ~2,000 more training fights; retrain from scratch on teacher
   + self-play data (t5).
4. **Evaluate on the Phase 1 dev fights:** seeds 0–1 against the stored MCTS runs, plus the pivotal held-out seeds.
5. **Confirm once** on fresh fights from buckets 2–3 (untouched), with a rule written down beforehand: overall
   HP-eq > 0 with the interval excluding 0, and no boss below −2 HP-eq or −2 win points.

## Cost

- Boss fights are long; assume ~10–15 s/fight/worker at 20k (the old mixed evaluation: 7.8 s). Pilot first.
- Phase 1 ≈ 14 runs of 150–1,130 fights ≈ **4–5 h** on 11 workers. The 50k and oracle-50k arms are the largest;
  run them last with time gates.
- Phase 2 ≈ **6–8 h** (teacher data dominates).

## Risks and caveats

- **Encoding coverage:** encoding v4 is Ironclad act-1 scope. Check that boss-specific statuses and moves are
  encoded before Phase 2: Guardian mode shift, Hexaghost's divider/inferno, Slime Boss split.
- **Deck quality drives boss outcomes a lot,** so per-fight win probabilities cluster by run. Keep the run-seed
  clustered intervals, and don't read small per-boss differences.
- **Macro may still dominate:** a better deck changes boss win rates by more than any combat gain seen so far.
  Phase 1 (which costs no training) gives the combat side of that comparison.
