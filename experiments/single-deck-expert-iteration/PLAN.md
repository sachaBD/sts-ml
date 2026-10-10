# Fresh-network, single-deck learning

Status: bootstrap + five updates and the reserved 600-seed comparison completed.
See [FINAL_RESULTS.md](FINAL_RESULTS.md): learned2k 595/600 versus MCTS20k 514/600.

## Goal

Learn this fixed Champ fight from scratch, then improve through search-guided self-play.
Aim for 100% wins; do not assume every combat seed is winnable. This is a practical
ceiling probe for this recipe, not a mathematical upper bound.

## Fixed fight

- Human deck `7a9ada48-a4d6-41fc-a85b-0147618df00a`.
- 19 cards: Barricade+, Entrench+, Body Slam+, Armaments+, Headbutt, Juggernaut, etc.
- Ironclad A20 Champ, **41/75 HP**, original fixed relics/counters, **no potions**.
- Only combat seeds vary initially. Add HP variation later, as a separate phase.
- MCTS20k selection check: **18/20 wins**, 95% Wilson interval **70–97%**.

## The two training targets

**Value:** `100 × actual eventual fight win` (100 for victory, 0 for defeat).
Every visited decision state gets its completed trajectory's outcome. Train with
scaled squared error: `((prediction - target)/100)^2`.

**Policy:** normalized root search visits over legal actions.
Train with cross-entropy; near-flat targets receive less policy weight. Do not replace
visits with the single played move or invent damage/survival rewards.

Values describe success under the data-generating continuation policy, not theoretical
winnability. The bootstrap teacher uses its existing HP-sensitive search objective;
its visits are useful supervision, not pure win-optimal labels. Its root scores are
**never** substituted for actual win outcomes.

## Fresh network and bootstrap

- **No D5 or other pretrained weights.** Fresh width-64 policy/value network, RNG seed 0.
- Bounded value head: `100 × sigmoid`; predictions stay in 0–100. Existing softplus
  checkpoints retain their defaults; activation is recorded and inherited on resume.
- Generate **200 MCTS20k training fights** on new seeds. Train 10 epochs, lr=1e-3.
- Separate **100 fixed monitoring seeds**, with a one-time MCTS20k reference.
  Their trajectories may supply validation losses but never gradient updates.

## Gradually move to self-learning

- Each update: **100 fresh learner + search fights**, 2k simulations, exploration on.
- Warm-start each update from the previous specialist; preserve AdamW moments.
- Three epochs/update, lr=3e-4, gradient norm cap=1. No policy sharpening.
- Replay the last ten learner batches. Sample **64 states per fight per epoch** with
  replacement so long losing fights do not dominate simply by being longer.
- Taper the old teacher replay: update 1 uses 200 teacher fights, then 20 fewer each
  update. Update 5: 120 teacher / about 500 learner fights. **Update 11: self-only.**
- No additional teacher collection during that transition. Value labels remain actual
  outcomes and policy labels become learner-search visits.

## Evaluation and plot

- Evaluate after bootstrap, then every **five updates**; cache the MCTS reference.
- Plot win rate against cumulative generated training fights, with 95% Wilson intervals
  and the MCTS line. Also plot paired win-rate gap and save CSV/JSON.
- 100 monitoring seeds: worst-case single-rate 95% half-width about ±9.8 points;
  paired-gap precision depends on disagreement. Monitoring curves are exploratory.
- Reserve **600 independent final seeds**, never played/labelled during development;
  use once on a chosen checkpoint (worst-case single-rate half-width about ±4 points).
- Caps/errors remain visible, not manufactured into labels; halt collection if >5%
  incomplete. Report paired coverage alongside win rate.

## First bounded run

Current run: **8 gameplay workers**, unchanged. User preference for **future new runs:
10 workers**. Resumption of this run keeps its recorded 8-worker configuration.

Bootstrap plus **five updates**: 200 teacher + 500 learner training fights. Show the
first curve before extending toward update 11 and self-only training. Native worker
is frozen, stages are resumable, and outputs/checkpoints live in managed runs.

Controller: `python -m apps.run_rl.single_deck`. Reuses existing play/replay/learner;
no native search/tree changes. No intercom without permission.

User preference: **always include copy-paste monitor commands in run/status updates**.
One read-only dashboard covers the whole experiment (from repo root):

```sh
watch -n 5 '.venv/bin/python -m apps.run_rl.single_deck_status --run runs/schema=combat_v4/date=2026-10-05/id=single-deck-fresh-v1'
```

Remove `watch -n 5` for a one-time status snapshot. The dashboard shows current phase,
completed stages, collection/training/evaluation progress, latest results and log paths.
Ctrl-C exits monitoring only, not the experiment.
