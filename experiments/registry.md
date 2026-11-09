# Experiment registry

Experiment material (configs, scripts, plans, and reports) is kept under this directory. “Last touched” is the date of the most recent committed change to that experiment.

- **act-1-card-selection** — Last touched: 2026-10-01. Planning for improved Act 1 card-reward decisions with a frozen combat agent.
- **act-1-combat** — Last touched: 2026-10-01. Act 1 combat value-model training, validation, and MCTS comparison work.
- **combat-v4-format** — Last touched: 2026-10-02. New minimal combat_v4 (fights + search tables): recording test, storage and write/read performance.
- **act2-boss-search** — Last touched: 2026-10-02. Act 2 bosses: 20k vs 100k rollout MCTS on 300 rebuilt fights (no meaningful gain).
- **act-1-easy-combats** — Last touched: 2026-10-01. Tests whether MCTS is near-optimal on floor-1 easy-pool fights.
- **card-outcomes** — Last touched: 2026-10-01. Staged data generation and training for card-outcome models.
- **card-search** — Last touched: 2026-10-01. Card-choice search agreement measurements.
- **elite-bench** — Last touched: 2026-10-01. Reproducible Act 1 elite benchmark and search analysis.
- **elite-v3** — Last touched: 2026-10-01. Training and evaluation of the v3 full-state, policy-head net on A20 elites.
- **joint-expert-iteration** — Uncommitted work, 2026-10-02. Staggered combat/overworld training, replay-first combat_v4 and overworld_v1 recording, paired evaluation and final handoff.
- **outcome-ui** — Last touched: 2026-10-01. Small win-rate UI for the pre-combat outcome model.
- **reuse_probe** — Last touched: 2026-10-01. C++ probes for reuse behavior and performance.
- **run-rl-v4** — Last touched: 2026-10-01. Two-round real-run RL experiment with every run decision made by the value network.
- **single-deck-expert-iteration** — Uncommitted planning, 2026-10-05. Fixed human-derived Champ loadout; HP/seed variation, teacher bootstrap, repeated search-guided learning, and paired learning curves.
- **human-deck-corpus** — Uncommitted, 2026-10-06. Joint multi-deck expert iteration over human Champ decks; reproduced on A, plateaued on hard decks. See HANDOFF.md.
- **slime-v6** — Last touched: 2026-10-01. Slime Boss A20 bootstrap, initial model, and DAgger iterations.
- **slime-v7** — Last touched: 2026-10-01. Fair-search limits assessed before expert iteration on Slime Boss.
- **slime-v8** — Last touched: 2026-10-01. Slime Boss self-play / rollout-teacher iteration.
- **topology-sweep** — Last touched: 2026-10-01. Outcome-model topology sweep.
- **multi-fight-champ-expert** — Uncommitted work, 2026-10-08. One rollout-guided expert-iteration network for several fixed A20 Champ fights (`apps/combat_expert_iteration`); 2-fight model beats MCTS20k on both, matches the specialists.
- **champ-full-corpus / inventory48 cold** — Registered: 2026-11-05. Approved overnight run of the explicit inventory/entity network with outcome/auxiliary heads; prior v2 recipe, 48/128-token GPU routing. Config and run/dashboard registration: [`champ-full-corpus/README.md`](champ-full-corpus/README.md#inventory48-cold-run--2026-11-05). Run ID: `gumbel_nn_combat_v1/2026-11-05/champ-full-corpus-v2-inventory48-outcome-aux-cold`; dashboard port 8002; previous warm iteration 70 retained.
- **champ-full-corpus / inventory48 wide128 restart** — Registered: 2026-11-05. User-approved fresh restart with wide inference batch128 instead of32, all learning/search settings unchanged. Config: `champ-full-corpus/train-v2-inventory48-wide128-64s-cold.toml`; run `gumbel_nn_combat_v1/2026-11-05/champ-full-corpus-v2-inventory48-wide128-outcome-aux-cold`; dashboard8001. Supersedes the initial inventory48 run.
