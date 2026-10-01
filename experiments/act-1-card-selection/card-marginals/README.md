# card_marginals: paired card-addition fight data (handoff, 2026-09-29)

**Goal:** a cheap, broad bootstrap dataset for the pre-combat outcome model (Ironclad A20, act 1): combat outcomes of
synthetic but natural-looking pre-combat states, with **matched card additions**, so the model can learn the marginal
effect of adding a card. Later data is meant to come from agent-played runs if the approach works. The app only
generates fights; it trains nothing. The model stays fixed at first, so data gains can be measured on their own.

## Central algorithm (agreed with the user)

1. **Build a base deck by running the simulator** (a fresh `GameContext`, seed = group_seed); sample only counts, never contents:
   - Neow: SimpleAgent (always option 0, as in the natural cohort).
   - Prior fights: a real card reward roll each (natural rarity rates and pity). Pick: SimpleAgent with `simple_pick_prob` (0.5),
     else a uniform offered card, or skip with `skip_prob` (0.1).
   - **easy:** prior fights ~ `prior_fights_weights` [.37,.35,.28]. A potion roll per prior fight (`addPotionRewards`).
     HP drawn from natural easy HP at that fight index.
   - **hard_elite / boss:** counts copied from a **donor**, a natural pre-fight state of the stage (buckets 4-9): prior fights,
     elites before (elite rooms at random positions: elite card roll + elite-tier relic), extra relics (tier roll
     50/33/17, `obtainRelic` with pickup effects; bottle screens are played by SimpleAgent), removed Strikes/Defends and
     upgrades (up to the donor count, Neow's included; SimpleAgent choice or random 50/50; removes only ever take a
     Strike/Defend), potions (random, up to the count), HP fraction, floor. Persistent relic counters (Pen Nib, Nunchaku,
     Happy Flower, Ink Bottle, Sundial, Incense Burner) are drawn from the same relic's natural values.
2. **Variants:** the base deck (skip) + `n_candidates` (4) distinct candidate cards. Each candidate is one card from a fresh reward
   roll of the deck's state, or with `uniform_frac` (0.1) uniform over the Ironclad pool.
3. **Paired fights:** every variant plays every encounter of the group on the same `seeds` (8) fight seeds.
   - easy: all 4 easy encounters.
   - hard_elite: all 3 elites + 3 random hard-pool encounters.
   - boss: all 3 bosses.
   - Room set by kind (elite / boss relic triggers).
   - Agent: fair guided-rollout MCTS, 8 particles. Budgets are set in config: `simulations` (easy 500 / hard 2000),
     `elite_simulations` 5000, `boss_simulations` 15000.
4. **Row per fight:** pre / post persistent state (post-state format as combat_transition_v1, i.e. after exitBattle,
   Burning Blood applied), won, start_hp, battle_final_hp (before Burning Blood), variant / card / source, encounter / kind,
   seed, simulations, seconds, group / group_seed / bucket (= group_seed % 10, the future split key), donor run/fight,
   group_json.

User decisions: easy fights include potions at natural rates; all 4 easy encounters per deck; hard 3 + elites 3;
boss = realistic boss states only (donors from natural boss states); every boss against the same decks.

## Code and configs

- `apps/card_marginals/worker.cpp`: modes `deck` (build + describe) and `fight`. `card_marginals.py`: the driver, which
  also prints decks with no fights: `PYTHONPATH=python:. .venv/bin/python apps/card_marginals/card_marginals.py CONFIG --decks N`.
- Configs: `easy.toml` (smoke), `easy-pilot.toml`, `hard-elite.toml` (smoke-sized), `boss.toml` (smoke-sized).
  Run: `./apps/card_marginals/run.sh CONFIG [--scratch]` (schema `card_marginals_v1`).
- CMake target `card_marginals_worker` was added to `CMakeLists.txt`. Nothing is committed; the tree has other
  sessions' uncommitted work.
- Natural distributions / charts: `scratch/card-dataset/natural_dists.{py,png}`. Printed decks:
  `scratch/card-dataset/{easy,hard_elite,boss}_decks.txt` (200 each; generated counts match the natural cohort closely;
  there are slightly fewer zero-remove decks because Neow removes/transforms independently of the donor).

## Status

- **easy:** verified. Smoke run in scratch; Burning Blood post = +6 in every row; start HP = pre HP.
  **Pilot done:** `runs/schema=card_marginals_v1/date=2026-09-29/id=easy-pilot`: 1,000 groups, 160,000 fights,
  20.6 min on 10 workers, 1 loss. **Not analysed yet**: run `scratch/card-dataset/analyze_easy_pilot.py`
  (light, single core). Partial result at 74 groups: base HP loss by encounter about 0.4 HP above natural; pairing
  halves variance (var(delta) 24.8 vs 52.3); reliability of a (group, card) mean delta over 4 encounters x 8 seeds
  ~0.71 (split-half 0.55), per encounter 0.22-0.60. This tells us how to set seeds for the expensive stages.
- **hard_elite:** compiled; decks printed and checked. **No fights played yet** (needs a smoke run: real hard/elite costs).
- **boss:** compiled; decks printed and checked. **No fights played yet.** Untested: exitBattle/post state after a
  synthetic boss fight. Smallest check: `boss.toml` with groups = 1, seeds = 1 (15 boss fights, ~2-3 min on 1 core).
- Cost estimates (earlier measurements under 10 workers): elites ~0.7-2.7 s, bosses ~8-11 s per fight; a boss group
  (5 x 3 x 8 = 120 fights) is ~20 worker-min, about 30 groups/hour. A hard/elite group (5 x 6 x 8 = 240 fights) is
  ~5 worker-min (hard cost guessed).
- profile-agent was briefed on performance (row recording in `play_fight` is wasted work for this app).

## Next steps

1. Analyse the easy pilot (seed/candidate allocation, per-card effects, noise).
2. Boss and hard/elite smoke runs (once CPUs are free or on the spare core); measure costs and check the post states.
3. Choose seeds / n_candidates / group counts per stage from 1 + 2; pilots on user approval (state runtime + log path first).
4. Then: train the fixed combat_outcome_v1 model with vs without this data and compare on natural dev (buckets 0-1)
   plus paired card deltas vs a zero-effect baseline.

Constraints: announce runtime and log path before any job over a minute, and run it in the background. The user is currently using
most CPUs (a spare core is available for compiles). Easy mode's RNG order must stay unchanged (the pilot reproduces).
