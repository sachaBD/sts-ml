# Combat rescue and rollout-assisted expert iteration

## Executive summary

**Development result:** learned search reached83/100 wins against MCTS74/100 on the
same fixed Demon Form/Pyramid vs A20 Champ starts. Paired advantage9percentage points,
approximate95% interval+1.1 to+16.9. This repeatedly used development set does **not**
establish fresh superiority. A frozen600-seed confirmation is being prepared separately.

The useful intervention was classical rollout-assisted leaf evaluation, not card-specific
rules:50%network value +50%completed heuristic rollout value inside learned PUCT search.
The agent retains its learned policy priors,2,000simulations per decision and public-belief
particles. Its model alone remains below MCTS. The final agent is **network plus search**.

## Fixed task and comparison

Ironclad deck249e5246-153d-4030-bc11-598774146147,34/52HP,A20Champ, original relic/counters,
no potions. Only combat seeds vary in complete-fight evaluations. Teacher is the deployed
guided-rollout MCTS20k with its original objective; learned search uses100/0 win units.
This is a comparison of deployed recipes, not a claim that their objectives are identical.

No global card bonuses/bans or oracle state features were introduced. Existing simulator
limitations remain: known Headbutt top-order information is not represented by the learned
features/search sampler; restart studies excluded affected roots. Public particle sampling
and copied simulator internals are not asserted to be a mathematically exact posterior.

## Development progression

All entries below use the SAME100 development starts, so do not treat the rows as
independent confirmations or their gains as additive causal effects.

| Agent / intervention | Wins/100 | Interpretation |
|---|---:|---|
| Original update15, learned2k | 62 | Starting reference |
| Teacher-fitted policy, original value,2k | 64 | Changed decisions; no established gain |
| Original update15, learned10k | 67 | ~5x cost; gain uncertain |
| Matched ordinary-training control, Stage2 | 61 | Rescue pilot control |
| Backward-restart rescue, Stage2 | 66 | +5 vs control,95% gap interval−3.5 to+13.5pp |
| Rescue +50%rollout leaf values, Phase4 | 76 | +10 vs rescue,95% gap interval+2.9 to+17.1pp |
| Rescue +100%rollout leaf values, Phase4 | 76 | Same point estimate; does not eliminate all network-value uses |
| Further ordinary-training model +50%rollout, Phase5 | 80 | Fresh-data training control |
| Further rollout-trained model +50%rollout, Phase5 | **83** | Fixed candidate for fresh confirmation |
| Further rollout-trained model, no rollout at inference | 71 | Gains not fully amortized into the network |
| Deployed MCTS20k | **74** | Paired teacher reference |

Phase5 candidate versus training control (both50%rollout inference):+3pp,
approximate95% interval−2.2 to+8.2. The rollout-assisted-training-specific advantage is
not established. Candidate versus untrained rollout-assisted rescue:+7pp,+1.3 to+12.7.
Nominal intervals/tests here do not account for repeated adaptive development comparisons.

Observed mean inference wall time (100games/agent): candidate8.0s/game, Phase5control
withrollout7.8s/game. Phase4rescue withoutrollout7.3s/game, with50%rollout7.5s/game.
These were sequential runs; system-load variability/uncertainty is unquantified. A fresh
paired confirmation will measure candidate and MCTS timing together.

## What the failed and partial approaches taught us

1. Whole-trajectory correction was weakly absorbed. Direct fitting showed the network
   could learn substantially more teacher policy, but changing priors alone did not
   establish stronger play with its old evaluator.
2. Restart training located a useful competence gradient (late positions easier), yet
   its full-fight improvement remained inconclusive.
3. Forced single-action alternatives under learner continuation were mostly unhelpful.
   Four screen winners yielded6alternative-only wins vs1baseline-only win in64fresh
   confirmation pairs; exact sign-test p=.125. A three-path bootstrap interval was
   misleadingly narrow; it is not evidence sufficient for policy training.
4. Supplying completed-rollout evidence inside search produced the first clear
   development gain. This implicates the learned-leaf evaluation/search interaction,
   not uniquely a defective network architecture or value head.
5. Training on fresh full combats with that stronger search produced a promising
   candidate. Independent confirmation is the next evidential step.

## Recipe at the candidate endpoint

Initialization: Stage2 rescue endpoint (derived from original update15 specialist).
Phase5: three updates of200fresh complete fights per arm. Strong branch uses greedy2k
search withrolloutmix.5 for collection; control usesmix0. Each update1000AdamW steps,
batch64 containing32fixed-old+32own-new states, fight-uniform then state-uniform.
Old replay1432fights; own-new replay grows to600. lr3e-4,wd.01,gradclip1, optimizer
moments resumed. Value target actual100/0outcome; policy target concentration-weighted
search visits. Fixed update3 endpoint; no validation/checkpoint selection.

This is not yet a demonstrated from-scratch general recipe: the candidate inherited
substantial prior specialist training and a small restart curriculum.

## Reproducibility / artifacts

All roots under `runs/schema=combat_v4/date=2026-10-06/`:

- `id=counterfactual-rescue-v1/out/stage2/REPORT.md`: controlled restart pilot.
- `id=counterfactual-branch-v1/out/REPORT.md`: forced-action screen/confirmation.
- `id=counterfactual-rollout-leaf-v1/out/REPORT.md`: rollout-leaf intervention.
- `id=rollout-expert-iteration-v1/out/REPORT.md`: matched expert iteration.
- Candidate checkpoint: `id=rollout-expert-iteration-v1/out/strong/update3/model/`.
- Worker: original frozenpv_worker, SHA256prefix58c5c4ab.
- Inference: `pv_worker play MODEL.onnx 2000 --rollout-mix 0.5` with the existing JSONL
  combat-start protocol. Teacher: `pv_worker teacher 20000`.

Stage2 completed all1208dispatches but its report process exited1 due to a lambda-label
bug; the report was repaired using recorded model paths, without replaying games. Phase3,
4 and5 completed normally. See phase-specific reports for all deviations and budgets.

## Confirmation and overnight work

Phase6result is pending; do not describe83vs74 as an untouched-test result.
`OVERNIGHT.md` freezes bounded two-replica continued learning, independent monitor/final
sets, checkpoint selection, failure handling and SRE ownership. Its purpose is to seek
further stable improvement without changing card rules, architecture or reward.
