# Multi-fight Champ expert iteration

Goal: one network for several fixed A20 Champ fights, trained by rollout-guided expert iteration, stronger than
MCTS20k on each. Started 2026-10-08.

**Latest result:** ten-deck update-6 model wins 1474/3000 versus MCTS20k 1051/3000 on fresh paired
seeds: +14.1 pp (reported approximate 95% CI +12.2 to +16.0). These are training decks, not unseen decks.
Full deck screens, results, checkpoint provenance and caveats: [TEN_DECK_RESULTS.md](TEN_DECK_RESULTS.md).
**Corpus pilot (9–10 Oct, unseen decks):** the ten-deck model is −11.0 pp vs MCTS20k on 20 unseen families;
after 4 rounds on 40 new families −4.5 pp [−11.0, +1.2]; +6.5 pp [+1.8, +12.3] over the start.
See [CORPUS_PILOT_RESULTS.md](CORPUS_PILOT_RESULTS.md); protocol [CORPUS_PILOT.md](CORPUS_PILOT.md).
Earlier larger-scale discussion draft: [GENERALIZATION_PLAN.md](GENERALIZATION_PLAN.md).

Historical five-fight launch notes: `overnight.sh`, `RUNBOOK.md`, `monitor.sh`, `HANDOFF.md`;
the handoff describes an earlier state, not current job status.

## Recipe

MCTS20k plays bootstrap fights; a fresh width-64 network trains on them (value = 100 x actual win, policy =
root visits). Then each update: network + PUCT 2k search plays new fights (exploration on); retrain on the last 10
batches plus tapering teacher fights (3 epochs, lr 3e-4, 64 states/fight). Monitor fights never train.
Search leaf = (1 - `rollout_mix`) network value + `rollout_mix` guided rollout:
- `0`: network search. Barricade specialist `single-deck-fresh-v1` (`experiments/single-deck-expert-iteration/`).
- `0.5`: rollout-guided, used here. Demon Form specialist Phase5 strong/update3 (`experiments/combat-agent-rescue/`).

Code: `apps/combat_expert_iteration/` (`run.sh` train, `evaluate.sh` test, `status.py` dashboard; see `config/`).
It replaces the ad-hoc scripts now in `apps_garbage/run_rl_combat/`. All runs use frozen `pv_worker` `58c5c4ab...`.

## Fights

| name | deck | HP | MCTS20k, 20-seed sweep (`human-deck-library-v1`) |
|---|---|---:|---:|
| barricade | `7a9ada48` Barricade+, Entrench+, Body Slam+ | 41/75 | 18/20 |
| demon-form | `249e5246` Demon Form, Pyramid | 34/52 | 14/20 |
| metallicize | `1676535e` 3x Metallicize, Rupture, J.A.X., Uppercut+ | 75/75 | 10/20 |
| corruption | `5d626798` Corruption+, 2x Inflame+, Flex, Heavy Blade+, 5x Bite | 37/59 | 8/20 |
| apparition | `ca29d1e5` 3x Apparition, Searing Blow+, Impervious, Offering | 27/40 | 3/20 |

No potions; original relics. Each run's exact config is in its `out/config.toml`.

## Runs (`runs/schema=combat_v4/date=2026-10-08/`)

| run | what | outcome |
|---|---|---|
| `id=barricade-rollout-v1` | rollout recipe, Barricade only | stopped by user in update 4; not used |
| `id=champ-both-rollout-v1` | one network, barricade + demon-form: 300 bootstrap + 5 x 200 learner fights per deck | 69 min |
| `id=champ-both-test-v1` | that model, 600 fresh seeds per deck | 20.5 min |
| `id=demon-paired-test-v1` | same 600 demon-form seeds: Phase5 specialist and MCTS20k (joint fights reused) | 19 min |
| `id=champ-five-rollout-v1` | one network, all five fights, 10 updates, 50 monitor fights/deck every update | overnight |
| `id=champ-five-test-v1` | update-10 model + MCTS20k, 300 fresh seeds/deck; two-fight model on its two decks | overnight |

## Results: two-fight model

| fight | agent | wins | 95% Wilson | joint minus agent, 95% interval |
|---|---|---:|---:|---:|
| Demon Form (paired) | joint | 457/600 | 72.6-79.4% | |
| | Phase5 specialist | 468/600 | 74.5-81.1% | -1.8 [-4.5, +0.9], McNemar p=.23 |
| | MCTS20k | 433/600 | 68.4-75.6% | +4.0 [+1.2, +6.8], p=.006 |
| Barricade (unpaired*) | joint | 598/600 | 98.8-99.9% | |
| | specialist | 595/600 | 98.1-99.6% | +0.5 [-0.4, +1.4] |
| | MCTS20k | 514/600 | 82.6-88.2% | +14.0 [+11.2, +16.8] |

\* against the earlier 600-seed tests on other seeds.

- One network beats MCTS20k on both fights and matches the Barricade specialist.
- On Demon Form it is not significantly different from the specialist, though the specialist had far more Demon
  Form training.
- An unpaired comparison first gave -4.7 vs the specialist. The specialist's own score moved 485 -> 468/600 between
  seed sets, so use paired tests for model comparisons.
- Caveats: one training run per model; one fixed start per deck; Barricade is at ceiling.

Plateau: collection win rates (exploration on) were barricade 87% (MCTS) -> 92, 99.5, 98, 98.5, 98% and demon-form 71%
(MCTS) -> 67, 71, 70, 76, 73% (+-6 pts each). Most of the gain came with the bootstrap network; cause of the flat
Demon Form curve not established. The overnight run monitors every update to show the curve.

## Next

Read the overnight curve and test. If Demon Form stays flat, test the update step (epochs, greedy collection) before
adding fights.
