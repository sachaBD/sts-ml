# Phase 3 — counterfactual action branching at stalled early TRAIN states

Status 2026-10-06 22:42 BST: main COMPLETE (9.3 min, exit 0). 4 confirmed pairs, pooled +0.078 (6/1/57 particle W/L/T,
sign p=0.125, 3 path clusters); rule met mechanically but weak; teacher−baseline screen +0.000. See run out/REPORT.md.
Executor: single-fight-opus. Run: `runs/schema=combat_v4/date=2026-10-06/id=counterfactual-branch-v1/`
(logs/smoke.log, logs/main.log; out/REPORT.md always written). No training in this phase.

## Question
Can guaranteed exploration find alternative root actions that the frozen learner (rescue update3, 66/100 dev)
itself executes better, measured by completed learner continuations rather than a shallow evaluator?

## Pipeline (`branch.py main`, one chained runner, bound 45 min, 10 workers, no retries)
1. **Replay** the 85 rescue-arm early TRAIN losses (Stage2 updates 2–3) with record=true, same request/particle/
   checkpoint (u2 episodes: rescue u1; u3: rescue u2). Exact equality with the stored shard rows (all data columns,
   original float precision) and outcome; FIRST mismatch stops the run.
2. **Freeze manifest** (`states.json`, hashed) before any branch outcome. Eligible decisions: >=2 legal, turn <=
   restart turn + 1, known-top k=0 (Stage0 rule; Headbutt/Setup/Warcry selections count as known). One state per
   episode uniform by sha256(tag:state:fid); episodes per path by sha256; paths by sha256; round-robin <=3/path to 32.
3. **Query** (32) at the recorded state S (reconstructed by `chain`): rescue-u3 2k fresh-tree choice (baseline) and
   teacher guided-rollout 20k public search (no oracle, salt 0); pv::encode features must equal the replayed row.
   Candidates <=4: baseline, teacher (if different), then uniform sample of remaining legal actions (sha seed). Dedup.
4. **Screen**: per state, 8 root particles (indices 2000+64s+[0,8)), the same index for every candidate; root sampled
   at S, candidate forced inside the particle, then rescue-u3 2k greedy to the end. <=1024 continuations.
5. **Confirm**: alternatives with complete 8+8 and gap g = wins_alt − wins_base >= +2; <=1 per state (largest g,
   tie teacher, then hash); top 16 by g. Alt and baseline on 16 fresh disjoint particles (2000+64s+[16,32)). <=512.
6. **Evidence rule** (predeclared): pooled confirmation alt − baseline win rate (paired by particle) > 0 and
   path-cluster bootstrap 95% lower bound > 0. Otherwise report and pivot; no training. Screen–confirmation
   shrinkage (winner's curse) and the unselected teacher − baseline screen estimate are reported.

Errors stop; caps are counted (never losses) and pairs with missing cases cannot qualify; >5% incomplete halts.

## Native additions (apps/continuation/continuation.cpp; absent fields = Stage1/2 semantics, dev:0 parity re-passed)
`chain` (reconstruct S inside the episode's sampled world), `force` (applied inside each sampled root; illegal or
menu mismatch = hard error), `mode: query` (features + learner/teacher proposals, no labels), `perturb` (tests only),
record-mode `trace`, `root_semantic` (fingerprint without uniqueIds).

## Sampler finding
`resampleDraw` re-copies the observed draw pile, sorts by cardKey (no uniqueId) and shuffles with a public-observation
seed (Scry prefix / Frozen Eye excepted); all RNGs + seed reset from that stream. Particles are therefore canonical up
to uniqueId order among equal-key cards (semantic fingerprints identical under hidden permutation + RNG replacement;
query output and features identical). Limitation: monster miscInfo is copied (history-determined internal state);
known Headbutt top-card order is not representable (states with k>0 excluded).

## Budgets
85 replays + <=1536 branch continuations + 6 smoke games (used: 6) + 32 queries (+2 smoke test queries).
