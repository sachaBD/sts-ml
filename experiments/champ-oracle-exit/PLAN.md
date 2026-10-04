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

## Phase 2 (from 2026-10-04 22:10 UTC, ~16 h compute, autonomous)
Evidence so far: per-action EXIT improves real play slowly (21.8 → 35.0% by r12, ~0.8 pp/round), oracle−real gap
widening (12.7 pp), depth flat (~1 turn). Turn search (r09 net, oracle) 45.0% vs per-action 41.8% (p=0.14) with
~1.5× deeper leaves at equal cost. Deployment is real play, so depth must reach real play.

Work items:
1. **Real-play turn search (determinized, PIMC):** at each real decision, sample K public-belief particles, run the
   oracle turn search on each, aggregate by first action (mean over particles of the best child Q among children
   whose sequence starts with that action), play argmax. Re-plan at every decision (simple; no cross-decision reuse
   in v1). Eval-only. Arms on the 409 bench with the latest model: per-action real2000 vs PIMC-turn (K=4, E=16) and
   (K=8, E=16).
2. **Turn-search training targets:** one turn search at turn start yields targets for every decision along the
   played plan: policy at a prefix = visits of root children consistent with that prefix, grouped by next action;
   value = root Q of the consistent children (visit-weighted). Emitted as normal per-decision search rows, so the
   existing encode/train path is unchanged.
3. **Larger held-out bench (bench2k):** the 409 bench decks × 5 fresh battle seeds (decks never in training sources),
   ≈ 2045 fights, no augmentation. Paired SE ≈ 1 pp. Teacher run on it once (~50 min on 9 cores).
4. **EXIT tag d:** expert = oracle turn search (E=64), init from the latest tag-c model, value-mix 0.5, N 3000/round,
   window 3, 1 epoch. Real bench via the better real-play agent from item 1.

Schedule: tag c runs to r15 (real bench ~01:30 UTC) then stops unless real ≥ 38% (clear acceleration); its cores
go to items 3 → 1 → 4. Turnbench arms 3/4 continue on CPU 11.
Decision rules: item 1 — if PIMC-turn beats per-action real2000 on bench2k (p<0.05), it becomes the real-play agent.
Item 4 — stop after 3 rounds without > 1 SE real gain; null ⇒ stop and report.

### Phase 2 queue (handed to orch 23:30 UTC; code 78c3b64)
Items 1–3 built (bd79595 PIMC, 37c694b turn targets, 44fbe58 bench2k, 78c3b64 exit flags).
PIMC smoke (r13, K4 E16, first 20): 1/20 vs per-action real r12 3/20, 16.8 s/fight — screened on 409 before bench2k.
After tag c stops: J1 PIMC screen (409) → J2 tag d (turn-search expert, E 64, init = last tag-c model, 9 rounds)
→ J3 bench2k: teacher, best per-action real, PIMC if not clearly worse.
Tag d stop rule: r06 real2000 not > M's real by 2.3 pp → stop and report.
