# Stage 2 bounded curriculum pilot — locked design

Status 2026-10-06 21:59 BST: main COMPLETE (executor single-fight-opus); results in run out/stage2/REPORT.md.
Locked clarifications: particle blocks of 16 from index 1000 enumerated over (arm, update, path, level, train/probe);
old/new halves use independent RNG SeedSequence([0,u,0]) / ([0,u,1]); probe threshold 35 of intended 48 (not
normalized); stop on any error/duplicate/trainer failure, <130/144 completed per arm-update or <44/48 probes;
dev eval with frozen pv_worker (58c5c4ab) play 2k. Report written even on failure.

This pilot tests backward restarts ONLY. Forced-action counterfactual branching is deferred
so it is not bundled into the curriculum comparison.

## Fixed arms and budget

Both arms initialize from immutable update15 model/optimizer (identical AdamW moments), seed0, and use current concentration-weighted policy + actual-outcome value losses, equal weights, width, batch64, lr=3e-4 and grad clip1. Three fixed updates; 1,000 optimizer steps/update; checkpoint every update; no epoch/validation selection. Every batch is exactly32 old +32 new states: fight-uniform then state-uniform within each half, giving32,000 old+32,000 new=64,000 sampled states/update/arm. The old half is immutable original1,000 round6–15 learner fights; new half is all completed arm episodes so far (maximum432), never probe/dev. Keep optimizer moments across updates.

Each arm gets144 greedy2k/no-noise new episodes/update (12/path x12): control gets12 freshly public-sampled full-start learner episodes/path; rescue initially gets9 middle+3 late public-particle restart episodes/path. State/particle sampling is the only exploration, identically specified in spirit for both arms; this differs from original exploratory training collection. Every root has a disjoint actual particle-index range by arm/update/path/level/train-vs-probe (>=4, outside Stage1 indices); metadata alone is not treated as seed separation. All outcomes retained; caps/errors excluded from training and counted. Control/rescue ranges are disjoint unless an explicitly declared matched scenario is used.

Total collection:864 training episodes (432/arm),144 rescue-only held-out probes (3 updates x12 paths x4 particles),200 development full fights (100/arm at2k):1,208 dispatches maximum, ten gameplay workers. No final seeds, forced-action counterfactuals, teacher labels, architecture or loss changes.

## Advancement

Probe the current PRIMARY level only, with a distinct actual PROBE particle-index range; probes are never train data. At each update4 particles/path=48: middle >=35/48 moves rescue allocation to9 early+3 middle next update; then early >=35/48 moves it to9 full+3 early. Otherwise retain the preceding 75:25 allocation.35/48 is a locked heuristic with Monte-Carlo uncertainty; teacher37/48 is a noisy screen, not a ceiling. The earlier level is assessed by the next independent probe. No within-run threshold tuning.

## Adapter implementation and validation

Stage1 results contain opaque root fingerprints, actions and visits but not training encodings. Existing `pv_worker encode` can only rebuild `start+actions`; it cannot restore a resampled public-particle restart state. Replaying a teacher prefix alone would restore a different hidden realization, not the sampled episode. The leakage boundary is which features reach the model, not whether simulator state can be serialized; direct native encoding avoids an unnecessary restoration path. A safe minimal adapter is native: from the sampled `BattleContext`, write `pv::ShardWriter` rows directly (current `pv::encode` features, legal action bits, learner search visits, terminal actual outcome) for the continuation; tag root namespace/train-vs-probe and never export piles/RNG/order. It must preserve wins/losses and reject nonterminal caps. Direct recording is implemented for build/smoke validation; correctness is not established until the gates below pass. Reuse existing ShardWriter/schema and learner loss/export code; no default trainer behavior changes.

## Required tests before launch

1. Native emitted row input/value/policy parity versus existing recorded full-start worker data for at least one true-start adapter smoke; restart rows load through Dataset with finite losses and correct terminal labels. Reuse existing reference records, not five new reference games. At most8 additional adapter combat dispatches, separate from the1,208 main cap; no hidden RNG/draw-order feature additions.
2. Replayed restart legal menu and root fingerprints match Stage1; k=0 asserted for all measured starts; TRAIN and PROBE namespaces disjoint.
3. Per-arm pre-training checkpoint and optimizer hashes identical; old/new state counts and optimizer steps exact; no duplicate root/job key.
4. Control root samples are not deterministic copies; 12 distinct sampled roots/path/update asserted.
5. Caps/errors persist in ledger and stop the stage if budget/completeness gate fails.

## Estimated runtime / command shape

Collection/evaluation cannot be estimated credibly until the native row writer is smoke-timed. Stage1 288 continuations took about 2m31s wall; 1,208 dispatches include 200 complete 2k fights and training, so a 45-minute target is plausible but unverified. Proposed main outer limit is45minutes, subject to the launch review after smoke timing. Use one bounded runner chaining collection/direct row recording -> matched train -> probe -> evaluation -> automatic REPORT/status, ten workers, `logs/stage2.log`, and immediate `bg_wait`; no separate manual stages. Preserve true exit status, counts and partial outputs; no automatic timeout extensions/retries.

Every completed branch update has exactly64,000 sampled states and1,000 optimizer steps;
three updates give192,000 sampled states and3,000 steps per arm. Save update1/2/3, evaluate
only fixed endpoint3 for the primary comparison. A short run or missing update is not an
endpoint substitute. Final reserved seeds remain untouched.
