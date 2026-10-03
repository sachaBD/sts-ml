# 3. AlphaZero-style search: a network that steers and scores

Goal: 4–8 turn plans in boss fights. Champ: build Strength while keeping him above 50%, then push him through
the enrage and kill him in 1–2 turns. Terms: [1-tree-search-basics.md](1-tree-search-basics.md).
Visit counts below are **illustrative**, not measured.

Current code: [PV README](../../../agents/combat/pv/README.md). PV now has a dedicated tree, not a runtime mode
of the legacy UCB driver. Its root is evaluated before construction; `Search<PuctScore, ArgmaxSelection>` is the
default. Values use HP-equivalent points (0 on defeat; 35 + HP + 4 × potions on victory), with no max-HP rescaling
or score clipping. The 0–1 values below are teaching examples, not the implementation's units.

## The idea in one picture

```
      tree search (the prior makes it narrow and deep)          value V (learned from many whole fights)
 now ●──●──●──◆──●──●──◆──●──●──◆──○  ─────────────────────────────────────────────────────▶ fight end
     │◀───────── 2–4 turns, searched ─────────▶│◀───────── rest of the fight, predicted by V ─────────▶│
```

The tree doesn't need to reach the end of the fight. It only needs to reach a position that V already knows
is good. If V has learned

```
Champ 60% HP, not enraged, my Strength 6, my HP 50   →  V ≈ 0.8
Champ 45% HP, enraged,     my Strength 0, my HP 50   →  V ≈ 0.2
```

then a 2-turn search that can steer toward either position picks the first, even though the payoff comes
6 turns later. The long plan lives in V; the search turns it into concrete card plays.

## One network, two heads

```
 what the player sees: hand, draw / discard / exhaust piles, energy, HP, block, powers (Strength …),
                       potions, relics, monsters (HP, block, intent, statuses, phase e.g. Champ enraged)
                                            │
                             shared trunk (sets / attention over cards and monsters)
                                  ┌─────────┴─────────┐
                              π  (policy)          V  (value)
                 a probability for each legal move     one number: how good this position is
                 = the prior: "look here first"         (win and HP left, 0 … 1)
```

## Selection with a prior (PUCT)

```
score(move) = average score(move)  +  c · prior(move) · sqrt(visits of parent) / (1 + visits of move)
```

It is still "average + bonus". The difference from today: **the bonus is scaled by the prior**.

- A move the network likes gets a big bonus → it is explored early and often.
- A move with prior 0.02 gets almost no bonus → it may never be tried.
- No rule forces every move to be tried once. The current PV implementation gives an untried move Q = 0; its prior-weighted bonus determines whether it is tried.
  A parent-value-minus-margin rule is a different first-play-urgency variant, not the current default.
- A new node is scored once by V. No rollout.

Knobs that may concentrate search: smaller `c` and sharper priors. Their actual effect on useful depth and strength
must be measured; a first-play margin is not currently implemented in PV.

## What the tree looks like

Illustrative old-search shape (no prior, rollouts): visits spread over moves at every node. Actual depth on Champ
has not been measured; the earlier 1–2 turn description was an estimate, not a result.

```
root  20,000
 ├─ Inflame 5,000 ─┬─ Bash 1,200 ─┬─ …
 │                 ├─ Strike 1,000
 │                 ├─ Defend 900   …every move gets visits at every level
 ├─ Bash 4,500 …
 ├─ Strike 4,000 …
 ├─ Defend 3,500 …
 └─ end turn 3,000 …
```

Illustrative prior-guided shape: visits can concentrate on moves π likes. This diagram is not evidence that
20,000 simulations actually reach several useful turns.

```
root  20,000        prior: Inflame .55 · Bash .25 · Strike .12 · Defend .05 · end turn .03
 ├─ Inflame 11,000 ── Bash 8,000 ── Strike 6,000 ── end turn ◆
 │                                                     ├─ world 1 ~700 ── turn 2 ── ◆ ── turn 3 ── ◆ ── turn 4 … ○ V
 │                                                     ├─ world 2 ~700 ── turn 2 ── ◆ ── turn 3 … ○ V
 │                                                     └─ …
 ├─ Bash 5,000 …
 ├─ Strike 2,500
 ├─ Defend 1,000
 └─ end turn 300
```

## Why the budget can afford depth

The search samples 8 worlds (hidden draw order and enemy RNG). At the first turn end the worlds draw different
hands and split. After that each world is deterministic, so it stays **one** branch per world.
The count stays at 8 instead of multiplying every turn:

```
                  ● end of turn 1
                  ◆
   ┌────┬────┬────┼────┬────┬────┬────┐
   w1   w2   w3   w4   w5   w6   w7   w8    turn 2 (each world drew its own hand)
   ◆    ◆    ◆    ◆    ◆    ◆    ◆    ◆     end of turn 2: still one continuation per world
   ◆    ◆    ◆    ◆    ◆    ◆    ◆    ◆     end of turn 3 …
```

One 6-turn line ≈ 6 turns × 5 decisions × 8 worlds ≈ **240 nodes**. 20,000 simulations can afford several.
Selection and noisy scores are suspected obstacles; these node counts do not prove which factor limits depth.

Caveat: deep in the tree there are only 8 sampled futures, so a deep line is tuned to those 8 draws. V has
seen many fights, which is a reason to stop the tree after a few turns and trust V.

## How the network learns (expert iteration)

```
 ┌─────────────────────────────────────────────────────────────┐
 │ 1. Play fights with the search steered by (π, V).            │
 │    Record, per decision: the position, how the search split │
 │    its visits, and at the end the fight's result.           │
 └──────────────────────────────┬──────────────────────────────┘
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ 2. Train:  π ← the search's visit split  (search > raw π)    │
 │            V ← the fight's result         (what happened)    │
 └──────────────────────────────┬──────────────────────────────┘
                                ▼
                   better net → back to 1
```

How "hold Champ above 50%" gets learned: some fights hold and win (by search, by exploration noise, or from
seeded data). Their positions get a high result → V rises for "Champ above 50% + Strength" → the search steers
there → π learns the moves that get there → more such fights.

The risk: if holding never happens in the data, it is never learned. Fixes: random exploration at the root,
a random move per fight, or fights from a deliberately patient rule-based player used only as data.

**Starting point:** not a random net. The combat_v4 recordings (1.85M searched decisions with the search's visit
splits, ~1.8k of them Champ fights) give a first π (imitate today's search) and V (what today's search achieves).
Self-play has to improve on that.

## What already exists in this repo

| piece | status |
|---|---|
| Dedicated PV search (PUCT/argmax template defaults, root noise, batched values, no rollouts) | implemented; useful depth/strength unmeasured |
| Network with π and V heads (v3, `elite-v3-t2`) | exists, trained on Act 1 elites only |
| elite-v3: π + V search vs today's search, Act 1 elites | matched / edged it (+0.42 HP-eq per fight, CI −0.16 … +0.98); **V alone, without π, lost** |
| slime-v8: V-only search after one round of self-play, Slime Boss | **beat** today's search: 800 vs 746 wins of 999 |
| Recorded fights with visit splits | 134k fights, 46k from Act 2 |
| Champ PV bootstrap/export/play tooling | implemented; HP-equivalent retraining and held-out strength evaluation pending |

## Open questions

- How deep does today's search really go on Champ, and how deep does a prior-steered one go? (Measure: log
  each simulation's depth in decisions and turns.)
- With a deep tree, are 8 worlds enough, or should the tree stop earlier and lean on V?
- Will self-play find "hold" alone, or does it need seeded patient-play data?
- Cost: slime-v8's network search was slower per fight than today's (20 s vs 13 s). With a good prior,
  fewer simulations may be enough.
