# Handoff: value net v3 + policy-guided search (Act 1 elites)

Author: nn-consultant, 2026-09-28. Read with [REPORT.md](REPORT.md) (results) and [PLAN.md](PLAN.md) / [LOG.md](LOG.md)
(how the overnight deck ran).

## Where things stand

- **Result:** v3 net + policy priors at every search node, compared with guided-rollout MCTS (both at 20k simulations, 8
  particles), on the 1,410 reserved final fights: **+0.42 HP-eq/fight [−0.16, +0.98]**.
  - Nob +1.00 [+0.31, +1.70], Lagavulin +0.98 [−0.09, +2.03], Sentries −0.72 [−1.85, +0.38].
  - First learned agent to match MCTS here; not yet a clear win.
- **The loss is in long Sentries fights:** −2.31 [−4.34, −0.37], 193 fights, deaths 17 MCTS vs 26 v3.
  - In short Sentries fights v3 is even or ahead.
  - Opening target choice is the same for both agents (both focus a side sentry), so the deficit is mid/late-fight play.
- **The same net as a value-only leaf still loses:** −1.46 on validation. The gain comes from the policy-guided
  search.
- **Candidate model:** `value_net_v1/2026-09-28/elite-v3-t2` (policy loss weight 0.25), leaf `policy_net`, c_puct
  1.0 (0.5 tied), fpu_reduction 0.05, prior_floor 0.03.

## What was built (all on `main`)

| Commit | What |
|---|---|
| `209fb90` | Encoding v4 (`combat/encoding_v4.cpp`, Ironclad Act 1 scope with `EXTEND HERE` marks): player statuses from potions/relics, 15 monster statuses + previous move, potion tokens (ID + 18-number effect description), relic tokens (ID + trigger progress). Model kind `deep_sets_v3` (Python `python/sts_combat_rl/topology/deep_sets_v3.py`, C++ `topology/value_net.cpp`): output = the terminal score formula over win / HP-given-win / potions-kept heads. |
| `6fb12aa` | Policy head (each legal move scored through the trunk's own card/monster/potion encoders; C++ caches the move-only part). `encode_action`. v4 row columns as **nullable columns in `combat_v3`** (incl. `legal_actions` on decision rows), written by `agents/teacher_search.cpp`. Loader `training/data_v4.py` (visit-share policy targets, identical moves merged). v3 losses in `train_value.py` (config keys `aux_keep_weight`, `policy_weight`, required only for `deep_sets_v3`). |
| `18c3a00`, `f44b3ec` | Leaf `policy_net` (`agents/teacher_leaves.cpp` `run_policy_net_search`): sim-changer's policy-prior mode (PUCT + first-play urgency, expansion at evaluation), objective 1. Values are converted to the search's scale: `v · (55 + maxHP_leaf) / (56 + M0)`. Priors = 0.97·softmax + 0.03·uniform. Teacher settings `c_puct`, `fpu_reduction`, `prior_floor`. Replay option `skip_diverged` in value_play / fight_resample. This rundeck. |

Simulator changes by sim-changer on the same day (sts_lightspeed `28d1781` … `1eef266`):
- policy-prior search mode;
- Smoke Bomb disabled (escape used to score as a win);
- Blood Potion + Sacred Bark heal fixed;
- potion discards removed.

**Old runs (before 2026-09-28) were played on the old simulator**, so don't compare against them. About 4–6% of
stored fights no longer replay identically and are skipped.

Tests: `tests/encoding_v4_test.cpp` (encoder checks; also dumps states / runs a policy_net search for the Python
tests), `tests/test_deep_sets_v3.py` (C++ rows → parquet → loader → model vs C++ values and policy logits; losses;
search smoke), `tests/test_policy_net_apps.py`.

## Data and models produced

- **combat_v3 runs** (with v4 columns):
  - `elite-v3-teacher`: 4,884 fights, MCTS teacher, random potions.
  - `elite-v3-selfplay`: 2,335 fights, t2 + policy search.
  - Validation/final plays: `elite-v3-v-*`, `elite-v3-f-*`.
  - These are the **only** rows usable for v3 training. Older rows lack the v4 columns.
- **value_net_v1 runs:** `elite-v3-t1` (policy loss weight 0.05), `elite-v3-t2` (0.25, the candidate), `elite-v3-t3`
  (t2 fine-tuned with self-play: no better in play, worse on Sentries).
- **Fight sets:** training sources are buckets 4,6,7,8,9 of `act1-all-bosses-a20-scaled-search`. The **final fights
  (buckets 2/3) have now been used once**; they're no longer untouched. Pick a fresh final cohort before claiming a
  future win.

## Known problems / debts

1. **The potions-kept head overfits:** validation MSE is about 70× training, and rises during training. Regularise or
   down-weight it; potion outcomes are sparse.
2. **t3's validation split overlapped t2's training data,** so its training metrics are optimistic. Next time, pin the
   split to t2's.
3. **The combat_v3 view is heavy:** one query took 24 minutes at ~14 GB RSS. Two app launches at once ran out of
   memory. Owner is compacting runs; stagger launches regardless.
4. **Multi-card-selection moves get identical priors,** because their move tokens are coarse. Fine for now.
5. **Coverage is Ironclad Act 1 only.** Other characters' potions throw in the encoder by design.
6. **The search normalises by 56 + M0** (M0 = max HP at search start), while training labels use 55 + max HP.
   Converted consistently; don't "fix" one side alone.
7. **The trainer depends on the owner's uncommitted working-tree changes:** `row_weighting = "elite_sqrt_source"` in
   `train_value.py` / `apps/value_train/train.py`, and `source_episode_id` in `data.py` META. Commit them before
   anyone resets the tree.

## Suggested next steps (cheapest first)

1. **Add the diagnostic slices to `experiments/elite-v3/compare.py`** so every comparison reports them: per elite, long vs
   short fights and hard vs easy, both by the original stored teacher's result on the same start. The code for this
   analysis is in the REPORT "Deeper look" section's description; it read the parquet directly because the view was
   mid-compaction.
2. **Value-net accuracy on long Sentries positions:** error by turn, Dazed count and sentries alive, against realised
   outcomes. This tells us whether the late-game problem is value error or search.
3. **Data:** more Sentries data, weighted towards long fights; the unused half of the compute budget is enough.
4. **Search:** a hybrid that uses guided rollouts late in a fight or near the end and the net elsewhere. This needs
   sim-changer: the policy-prior mode currently requires rollout length 0.
5. **Expert iteration done properly:** a pinned split, more self-play fights, and fine-tuning evaluated per slice.
6. **Replay a few "easy Lagavulin" fights** where v3 lost HP (−4.1, n = 42) to see whether it fails to rush the kill.

## How to run things

- Train: `./apps/value_train/run.sh <config>` with `[model] kind = "deep_sets_v3"`. See `configs/s2-t2.toml` for
  every key.
- Play: `./apps/value_play/run.sh <config>`. `leaf = "policy_net"` needs `c_puct`, `fpu_reduction`, `prior_floor`;
  also `skip_diverged = true` for stored fights.
- Compare: `PYTHONPATH=python .venv/bin/python experiments/elite-v3/compare.py --baseline ID --candidate ID`.
- Audit data: `experiments/elite-v3/check_data.py OUT_DIR`.
- Render config templates: `experiments/elite-v3/render.py configs/X.toml --var ...`.
