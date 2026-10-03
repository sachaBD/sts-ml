# 1. Tree search, the basics

No Slay the Spire details here, just the idea. Next: [2-today.md](2-today.md), [3-alphazero.md](3-alphazero.md).

## The situation

You have to pick a move **now**. Tree search tries moves *in imagination*, looks at how good the results are,
and then plays the move that looked best. Nothing imagined is played for real.

## The tree

```
                 (now)
                /     \
           Strike     Defend           ← moves = edges
            /  \       /   \
          (A)  (B)   (C)   (D)         ← positions after two moves = nodes
```

The full tree is far too big to look at. Illustrative numbers, not measurements: Slay the Spire: ~7 moves per decision, ~5 decisions per turn,
~8 turns per boss fight → about 7^40 lines. So every tree search answers two questions:

1. **Where do I look next?** (selection)
2. **How good is a position where I stop looking?** (evaluation)

Everything else is detail.

## One simulation = 4 steps

Each node remembers two numbers: how many times it was visited (`n`) and the average score of everything
seen below it (`avg`). In this teaching example, scores are 0 (lose) … 1 (win easily);
the actual PV agent uses HP-equivalent points.

```
1. SELECT     walk down from (now), choosing a move at each node, until you reach a move never tried
2. EXPAND     add the new node for that move
3. EVALUATE   give the new node a score
4. BACKUP     add that score to every node on the path you walked (n += 1, update avg)
```

One pass of these 4 steps is one **simulation**. The **budget** is how many simulations you run before
playing a real move.

## Watching a tree grow (5 simulations)

**Sim 1.** Nothing tried yet → try Strike. New node scores 0.60.

```
(now) n=1
  │
  Strike  n=1 avg .60
```

**Sim 2.** Defend never tried → try it. Scores 0.40.

```
        (now) n=2
       /          \
 Strike n=1 .60   Defend n=1 .40
```

**Sim 3.** Strike looks better → go down Strike. Below it, try Bash. Scores 0.80.
Backup: Strike's average becomes (0.60 + 0.80) / 2 = 0.70.

```
        (now) n=3
       /          \
 Strike n=2 .70   Defend n=1 .40
   │
   Bash n=1 .80
```

**Sim 4.** Strike again. Below it, try Defend. Scores 0.50. Strike: (0.60 + 0.80 + 0.50) / 3 = 0.63.

```
          (now) n=4
         /          \
 Strike n=3 .63    Defend n=1 .40
   /       \
 Bash      Defend
 n=1 .80   n=1 .50
```

**Sim 5.** Strike, then Bash (the best line so far), one level deeper ...

```
          (now) n=5
         /          \
 Strike n=4 ...    Defend n=1 .40
   /       \
 Bash      Defend
 n=2 ...   n=1 .50
   │
  ...                 ← the tree grows deepest where things look best
```

**After the budget is spent:** play the root move with the most visits (here Strike). Then the real game moves
on, and the search starts again from the new position.

## Selection: how greedy to be

Every selection rule has this shape:

```
pick the move with the highest:   average score  +  "I haven't looked here much" bonus
```

- **Bonus too small:** greedy. One unlucky early score buries a good move forever.
- **Bonus too big:** visits are spread over everything. The tree stays wide and shallow.
- **Add a prior:** a guess, before any searching, of which moves are good (e.g. from a trained network). The
  bonus then goes mostly to moves the prior likes, so obviously bad moves are skipped and the tree gets **deep**.
  This is the key difference in [3-alphazero.md](3-alphazero.md).

## Evaluation: three ways to score a new node

```
new node ──► game over?                        use the real result          exact
         ──► play it out to the end quickly    "rollout"                    one noisy sample, and only as
             with a simple fixed policy                                     smart as that simple policy
         ──► ask a trained network             "value function V(node)"     one call; as good as V is
```

Noisy scores need many visits before two moves can be told apart. Accurate scores let the search commit early.

## Randomness

Some moves don't lead to one fixed position. In Slay the Spire, ending the turn draws a random hand. A common
trick: guess a few complete versions of the hidden information ("worlds") and search all of them in one tree.
Positions that look identical to the player share a node; when the worlds start to look different, they split.

```
          end turn
        /    |     \
   hand A  hand B  hand C      ← different worlds drew different hands
```

## Words

| word | meaning |
|---|---|
| node | an imagined position |
| edge | a move from one position to the next |
| leaf | where one simulation stops walking and needs a score |
| simulation | one select → expand → evaluate → backup pass |
| budget | simulations per real decision |
| prior | guess of which moves are good, before searching |
| value | guess of how good a position is |
