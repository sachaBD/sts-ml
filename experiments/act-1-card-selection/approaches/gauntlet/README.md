# Gauntlet

**Status: v1 implemented** as `apps/gauntlet` (see [Implementation](#implementation-v1)); not yet run at scale.

## Idea

At a card reward, build one candidate per legal choice (each offered card, skip). Run each candidate through the same **gauntlet** of likely upcoming fights, after simulating the card picks that would happen before each fight. Pick the candidate with the best score.

It is a cheap cut-down of [CARDS](../cards/README.md): real fights instead of a learned combat model, a fixed set of target fights instead of full-run rollouts.

## Act 1 sketch

**Elite pool**
- Use the current map to find the next reachable elites; the distance gives the number of card picks before each.
- Per particle: make those n picks with a fixed policy (simple strategy, or the competitor), then fight each elite once (Gremlin Nob, Sentries, Lagavulin; deterministic AI on A20).
- Repeat for a few distances (~3).
- Output: `[distance, score per elite]`.

**Boss pool**
- Same approach against the known boss, at its single distance.
- Same output format.

**Noise control**
- Every option gets the same seeds: future card offers, and combat RNG per fight.
- Play still diverges once the decks differ; that is accepted.
- Potions and relics: open.

**Decision**
- Combine the table into one score per option (e.g. weighted HP lost with a death penalty). Pick the best.

## Why it is attractive

- **Cheap, simple, real combat:** no learned model needed.
- **It is the signal test:** paired, same-seed results show whether and how much one card moves fight outcomes. That is the key risk for CARDS, and the same data validates a future combat model.
- **Future picks are included:** the pre-fight picks capture some synergy and deck growth.
- **Upgrade path:** swap real combat for a combat approximator to make it near-instant.

## Rough cost

- **Fights:** (3 elites × ~3 distances + 1 boss) × P samples per option. With P = 10 and 4 options, that is ~400 fights per pick.
- **Time:** one measurement: a Gremlin Nob fight at 2,000 simulations took ~0.25 s (guided-rollout MCTS, 8 particles). Boss fights are longer; not yet measured.

## Implementation (v1)

```sh
./apps/gauntlet/run.sh apps/gauntlet/act1.toml [--scratch]   # schema gauntlet_v1
```

- **Reward states:** the card reward before a sampled fight of a stored bootstrap run (`[run] query`, `fight_index >= 1`). The run is replayed to that fight (stored combat actions; SimpleAgent outside combat), and the potions, relics and keys SimpleAgent would take first are taken.
- **Options:** each card of the last card-reward group, plus skip. The baseline is SimpleAgent's own choice at that reward.
- **Elites:** reachable elite nodes from the current map node. Distance = fewest combats (monster or elite rooms) before the elite. The nearest `elite_distances` distinct distances are used; each is played against Gremlin Nob, Lagavulin and Three Sentries.
- **Boss:** the known boss, with picks = the fewest combats before the boss.
- **Future picks:** SimpleAgent picks from fresh monster-room card rewards (card RNG = the sample seed).
- **Start state:** every fight starts at the reward state's HP, relics and potions (no rest-site upgrades, no HP carried between fights).
- **Seeds:** each (reward, encounter, distance, sample) has one seed shared by every option, so offers and combat RNG streams start identical.
- **Combat:** the teacher search (`leaf`, `simulations`, `particles`; fair play, no random move). One budget for all fights.
- **Score:** per fight = HP lost (death = all starting HP) + `death_penalty` if died. Option score = (1 − `boss_weight`) × mean elite + `boss_weight` × mean boss. Lower is better.
- **Outputs:**
  - `out/part-<episode_id>.parquet`: one row per fight × option.
  - `summary.json`: per reward state, each option's scores, the gauntlet and baseline choices, and the paired difference vs the baseline option (± 1 SE).
- **Not kept:** the gauntlet fights' combat rows. They would be counterfactual-deck training data for a combat approximator; add them if wanted.
