# Champ oracle expert iteration (champ-oracle-exit) — plan, 2026-10-04

Scope: Ironclad vs The Champ only. Successor to experiments/combat-pv (see its LOG.md for history).

## Hypothesis
An expert-iteration loop whose *search* sees the true state (draw order, RNG → deterministic future) trains a
*public-observation* policy/value net cheaply and at scale. Deployed with the normal 8-particle search, it beats the
guided-rollout MCTS teacher (41.8% on the 409-start bench), and its search plans ≥ 3 player turns ahead.

Why this should beat combat-pv's attempts: those failed on (a) 1–3k noisy fights, (b) flat visit targets,
(c) ~1-turn search depth. Oracle search removes chance branching (deeper, sharper visits) and needs few sims, so we can
afford 10^4–10^5 fights per iteration.

## Design (deliberately minimal: reuse PV search, network, encoder, starts generator, compare.py)
- **Search (training):** existing PV PUCT tree (normalized Q, c=1.25, batch 32) on the true state only (no particles).
  **Tree reuse:** the future is deterministic, so after a move the chosen child subtree is exactly valid; keep it
  (visits, values, priors) as the next root. Budget S = new sims per decision on top of the reused subtree.
  The network still sees only public features (encoder already excludes draw order/RNG), so the learned policy is
  deployable as-is; it learns the oracle's choices averaged over hidden states.
- **Search (deployment/eval):** unchanged 8-particle public-belief PV search.
- **Reward:** win = 1, loss = 0 (stored as 100·won). No HP term (≈ full heal after Act 2 boss; HP term biased play toward block over scaling).
- **Iteration 0 (warm start):** cold start is not viable (terminal signal ~60 actions away). Train policy on teacher
  visits and value on win/loss from the ~4.5k teacher Champ fights (champ-train 1.5k + champ-teachergen-a 3k).
- **Iteration k:**
  1. Generate N fresh starts (apps/pv/starts.py generate; fresh seed block per iteration; augmentation on).
  2. Oracle self-play with S sims, root Dirichlet noise 0.25; sample the played move ∝ visits for the first 2 turns
     (diversity), argmax after.
  3. Encode → rows. Value target = 0.5·outcome + 0.5·oracle search root value; policy target = root visits.
  4. Train 1 epoch from the previous checkpoint on a window of the last 3 iterations' rows (each fight trained 3×).
  5. Evaluate (below).
- **Decided (sweep0 + design review):** S = 800, N = 3000/round. N rationale: iteration 0 overfit after 1 epoch on
  4.5k fights, so each update needs ≥ ~9k fights; with window 3 that is 3000/round, the smallest (freshest) round
  that clears it. Check after round 1: if val value loss rises within training, raise N to 8000.
  Real2000 bench every 3rd round (~7 min), oracle/policy benches every round.
- **N, S (original):** set by an iteration-0 sweep of S ∈ {200, 800, 3200} (with reuse) in oracle mode on the bench: choose the
  smallest S within 2 pp of the best oracle win rate, then N to fit ≈ 45 min/iteration on 9 workers.
  (Teacher's 20k is UCB with no prior and full rollouts; network PUCT needs far fewer sims. ~20k sims/s/core ⇒
  20k sims/decision ≈ 1 min/fight — too slow for sample generation; revisit if the sweep is still rising at 3200.)

## Evaluation (each iteration unless noted; bench = runs/.../champ-bench-nodome, 409 starts, never trained on)
- **Oracle bench** (S sims, true state): the ceiling the loop is learning toward. Cheap (~2 min).
- **Real bench** (8 particles, 2000 sims, deterministic): the number that matters, paired vs teacher via
  apps/pv/compare.py (win rate, McNemar b/c and p, DF/no-DF split). Every iteration while cheap, else every 2nd.
- **Policy-only bench** (1 sim, argmax prior): how much planning the net itself has absorbed.
- **Depth:** mean / p90 player turns per simulation (existing PV telemetry), oracle and real modes.
- **Champ signatures** (champ_diag.py): Strength at the 50% crossing, turns crossing→kill, Executes taken.
- **Training health:** val value BCE, policy CE / top-1, visit-target entropy vs uniform.

## Decision rules
- **Null result ⇒ stop and discuss with the user** (no autonomous redesign loops).
- **Success:** real bench > 41.8% with paired McNemar p < 0.05 → freeze model, write REPORT.md, then run value probes.
- **Plateau:** 3 iterations with no real-bench gain > 1 SE (~2.3 pp) → stop and diagnose.
- **Hidden-info gap:** oracle − real > 20 pp and widening while real is flat → the oracle's moves don't transfer.
  Next step: value targets from real (8-particle) self-play fights, keep oracle visits for policy.
- **Depth:** oracle mean < 2 player turns/sim by iteration 3 → v2 search: transposition merging within a turn
  (deterministic ⇒ exact) / turn-level macro actions.
- **Flat targets:** visit entropy not falling below teacher's (1.37 nats over ~5 moves) by iteration 2 → lower c or raise S.

## Code changes (small; for impl-26-10-3)
1. `pv_worker play --oracle`: search the true state, no particles; tree reuse across decisions; tag agent string
   `oracle=1 reuse=1`. Tests: each edge has one outcome child; reused root matches the played state; replay passes.
2. Win-only objective: terminal value = 1/0 (or 100·won to keep the current head/loss scaling); data.py value target
   from `won`; contract bump so old models are rejected.
3. `--policy-only` play (no search; argmax prior) for the diagnostic bench.
4. Encode champ-teachergen-a; train iteration 0.
5. Loop driver `experiments/champ-oracle-exit/exit.py` (adapted from combat-pv/selfplay.py): resumable stages,
   oracle self-play, evals above, RUNBOOK.md lines, pid files.

## Ops
≤ 10 cores (9 play workers + trainer on GPU). DuckDB memory_limit/threads set. Parquet + DuckDB only. pid files, no
`pkill -f`. Storage under runs/schema=…/id=champ-ox-*.

## Known risks / unknowns
- Throughput at low S is unmeasured (800 sims × 8 particles was ~1.9 s/fight/worker; oracle per-sim cost similar).
- Oracle optimism: value learned is P(win | public state, oracle play) — optimistic for real play.
- The public-feature net may still fail to value long-term state (Strength vs HP); value probes come after the loop.
- PV worktree has uncommitted rollout-mix changes (impl-26-10-3); commit or stash before branching.
