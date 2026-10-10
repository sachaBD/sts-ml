# Human-derived Champ benchmark v1

## Question and setup

Can our learned combat agent handle human-built decks as well as guided-rollout MCTS?
Reconstruct Ironclad A20 Champ starts from the public Mega Crit run-history dump,
mostly build `2020-07-30`. Of 981 Champ fights, retain 435 decks with exact reconstructed
contents/upgrades and supported state; skip uncertain decks, Runic Dome, unknown bottled
cards, Lizard Tail usage, and accumulated card damage. Play each on its human run seed
and a deterministic fresh seed, with identical starts and no potions for both agents.

Compare MCTS (20,000 simulations) with D5 policy-value search (2,000 simulations).
Eight workers, sequential batches: approximately 32 minutes MCTS and 15 minutes D5.
D5 had four unsupported-card errors and two capped fights; exclude their four decks
from both agents. Final comparison: **431 decks, 862 completed fights per agent**.

## Results

Rates average the two agent seeds within each deck; humans have one observation per
deck. Uncertainty is **±1 standard error across decks**, not a 95% interval.

| Player | Win rate |
|---|---:|
| Human | 47.6% ± 2.4 points |
| MCTS 20k | **54.8% ± 2.2 points** |
| D5 2k | 44.4% ± 2.0 points |

Paired MCTS advantage over D5: **10.3 ± 1.6 percentage points**.
Of 205 human wins, MCTS lost both seeds on 22 decks, D5 on 34, and both on **15**.
Those 15 decks are the viewer shortlist, not proof of consistent agent failure.

Demon Form decks (n=60): MCTS 81.7% ± 4.1 points, D5 73.3% ± 4.7.
Without it (n=371): MCTS 50.4% ± 2.4, D5 39.8% ± 2.2.
Both agents benefit from scaling; the learned agent still trails on both subsets.

Manual viewing suggests unusual but potentially long-horizon play (e.g. accepting
incoming damage to reduce Blood for Blood's cost). This is a qualitative user
observation, not evidence that a particular action was optimal or causal to a win.

## Interpretation and limits

**MCTS is a strong baseline; this experiment does not establish unbeatable play.**
It still loses roughly 45% of tested fights, though the fraction recoverable by better
play is unknown. D5 uses one tenth the simulations; this is not an equal-compute test.
It also took about half the aggregate wall time here, not one tenth: neural evaluation
and the different searches have different per-simulation costs.

Do not infer a human skill ranking. Humans won 700/981 source fights (71.4%), but only
208/435 retained fights (47.8%): exact reconstruction disproportionately excludes wins.
Agents have no potions. Historical balance changes are noted but not corrected.
Canonical deck order, fresh RNG streams and zeroed unknown cyclic relic counters mean
these are approximate historical states, not exact human encounter replays. Many decks
reduce RNG noise, not these systematic biases. The human seed is not privileged fidelity.

## Recommended next experiment (not launched)

1. Collect more **usable reconstructed Champ decks**, not merely 10k raw games.
   This pull retained 435 decks from 14,886 raw runs (~2.9%); 10k comparable raw runs
   would yield roughly 290 retained decks if that yield persists. This is a corpus
   extrapolation, not a confidence interval; future windows may differ substantially.
2. Split source runs before augmentation: 9k training / 1k held-out decks if obtainable.
   Keep an untouched final test set or treat this v1 corpus as a frozen regression test;
   avoid cross-pull duplicate play IDs and shared descendants between splits. Group
   identical deck/loadout families where feasible. Repeated validation tuning is not
   an untouched final evaluation.
3. Begin with fresh seeds and conservative HP variation. Then try small, plausible
   deck edits as a separate ablation; preserve upgrade/relic/bottle consistency.
   Endless generated states are not endless independent human examples.
4. Start from MCTS policy/value supervision, then mix in learner-generated states and
   teacher corrections. Compare this against pure self-play rather than assuming pure
   self-play will improve overnight. Focus extra teacher effort on held-out-style
   training failures, never validation failures used as training examples.
5. Freeze evaluation starts/seeds. Track win rate and the paired gap per training round;
   compare at both current simulation budgets and equal measured decision time. Include
   a modest learned-value-plus-search budget sweep before claiming a training ceiling.

Success: reduce the paired win-rate gap at fixed decision time, or reach MCTS20k quality
at lower time. Beating MCTS with **learned guidance plus search** is a more plausible
near-term objective than replacing search entirely. Pure self-play has no guarantee of
escaping a weak policy without sufficient exploration and useful targets.

## Artifacts

- App: `apps/human_champ/` (prepare / play / report).
- Benchmark: `runs/schema=human_champ_bench_v1/date=2026-10-05/id=v1/`.
- Detailed splits: benchmark `out/report/REPORT.md`.
- Viewer shortlist: benchmark `out/report/viewer_starts.parquet` (15 decks / 30 starts).
- Agent runs: `combat_v4/2026-10-05/human-champ-v1-{teacher,pv}`.
- D5 model: `runs/schema=combat_v4/date=2026-10-05/id=champ-diag-D5/model/model.onnx`.
- Historical balance reference: official Mega Crit v2.2 patch notes,
  https://steamstore-a.akamaihd.net/news/externalpost/steam_community_announcements/3856733252019104438.
