# Macro simulator: what already exists (answers from the sts-game-model session, 2026-09-30)

The macro simulator plays a whole run **outside combat** in sts_lightspeed and resolves fights with the outcome model.
Nothing below has been built, timed or checked by us; it is the other agent's code reading.

## Already there
- **Full run loop:** `sts::GameContext{IRONCLAD, seed, 20}`.
  - The current screen is in `screenState`: Neow event, map, rewards, card select, rest, shop, treasure, boss relic,
    battle.
  - Generic actions: `GameAction::getAllActionsInState` / `execute`.
  - Existing loop: `environments/overworld/act1_run.cpp` (`stsrl::act1::play(game, FightFn, max_fights)`). SimpleAgent handles
    everything outside combat and a callback plays each fight.
- **Injecting a fight result:** the `skipBattles` flag, or catch `BATTLE` and write the result first: HP, max HP, gold,
  potions, relic counters, and `outcome = PLAYER_LOSS` for a loss. Then call `afterBattle()`. Rewards are still
  generated normally.
  - Reference for what a real fight writes back: `BattleContext::exitBattle` (`src/combat/BattleContext.cpp:461`).
  - Caveat: end-of-combat relic heals (Burning Blood) run in `exitBattle`, not `afterBattle`. Our outcome model predicts
    HP **before** Burning Blood, so the heal must be applied by us.
- **Copying the game:** `GameContext` copies by value (the map is a shared, read-only pointer). Branching per option is
  just a copy.
- **State readout (C++):**
  - Game state: deck, relics (+ data), potions, HP, gold, floor, act, boss, current map node, and the map nodes (room,
    edges).
  - JSON for deck, relics and potions: `environments/overworld/game_state.hpp`. There is no map serialisation yet; `apps/gauntlet`
    walks the map.
- **Full-run baseline harness:** `apps/bootstrap` runs Act 1 with the MCTS combat player.
  - SimpleAgent handles everything else: hand-coded card priority list (`SimpleAgent::stepCardReward`), fixed-weight
    pathing, and it always takes Neow option 0.
  - **Baseline to beat: Act 1 A20 clear rate 60.6% ± 1.5 (MCTS, 1,000 paired seeds).** Per boss: Slime 63.5%, Guardian
    66.0%, Hexaghost 51.4%. The boss causes about 72% of deaths (`experiments/act-1-combat/2026-09-27-act1-eval/README.md`).

## Gaps and work for us
1. **No Python bindings.** Everything reaches Python through C++ JSON workers run as subprocesses. Options:
   - a pybind11 module;
   - or put the whole search in C++ and call the outcome model from C++ (e.g. a TorchScript/ONNX export, or a small
     hand-written forward pass like `value_net.cpp`).

   The search makes very many calls, so a subprocess per call is too slow. The search loop should live in C++, or the
   model calls must be batched.
2. **Common random numbers are not automatic.**
   - Card A vs skip draws no RNG, so futures match *only while later decisions consume the streams identically*.
   - A different path, event or number of rewards shifts cardRng, potionRng and eventRng. Skipped fights also don't
     advance potionRng.
   - Fix: reseed the relevant streams per floor (or per reward) in our own layer. `apps/gauntlet/worker.cpp` already does
     something similar.
3. **Map serialisation and path enumeration** for the policy input (`map-encoding.md`).
4. **Known sim gaps** (`// todo`s in `src/game/GameContext.cpp`):
   - Prayer Wheel's extra card reward is missing.
   - Stolen gold is placed in the wrong reward order.
   - Some events are incomplete: Necronomicon, bottled-card edge cases, Dead Adventurer.
   - Lesson Learned write-back is missing.
   - Prismatic Shard and Smoke Bomb are disabled; Lizard Tail and a few other relic hooks are todo.
   - Card rewards with full potions just drop the potion.
   - Shops and Neow are not verified against the real game.
5. **Speed is unmeasured.** The estimate is microseconds per step and well under 1 ms per Act 1 run with fights skipped.
   A benchmark needs a build.
6. **Fixed heuristics** (Neow option 0, SimpleAgent pathing) bias comparisons. Keep them identical across the policies
   being compared.
