# Map encoding for the card policy: enumerate paths, max-pool whole paths

*2026-09-30. Applies to `run_policy_v1` (`python/sts_combat_rl/topology/run_policy_v1.py`). Research stage, untrained.*

## What the map input must capture
A card is worth more or less depending on the rest of the act. A deck that can beat elites wants a route with elites;
a weak deck wants rests and to avoid elites. The player does not experience "the map". They walk **one route**, and
they get to **choose** it. So the map input should answer: *given this deck and HP, what is the best route still
available, and what does it contain?*

## Designs we rejected, and why
1. **Pool all reachable nodes (sum/mean).** This says "5 elites and 3 rests are reachable". It cannot tell "I can avoid
   every elite" from "every route has 2 elites", because pooling throws away which rooms sit on the same route.
2. **Learned backward induction on the map graph.** Working from the boss down, each node combines its children's vectors
   with an element-wise max. The problem is that an element-wise max **mixes routes**: feature 1 can come from the left
   child ("rest soon") and feature 2 from the right child ("no elite"). The result describes a route that may not exist.
   It is also hard to explain what "best" means there, since no route is ever scored.

## Chosen design: enumerate every remaining path
The map is small: 15 floors, at most 3 edges per node, and a low route count in practice (confirmed by the user). So we
simply list the routes.

1. **Enumerate (in the encoder, not the network).** List every path from the current node to the boss. Merge paths with
   the same room-type sequence, because the network only sees room types. Each path is 15 slots indexed by absolute
   floor, with floors already behind us marked NONE.
2. **Embed each path, independent of the deck.** Sum room and floor vectors over the slots, then apply a small MLP. This
   gives "what this route contains, and when".
3. **Score each path for this after-state.** A small MLP takes the path embedding and a summary of the deck, relics, HP
   and boss after taking the option, and outputs one number: how good this route is for this deck. It is computed for
   every option (each card and skip), so taking a card can change which route is best.
4. **Max-pool whole paths.** Take the highest-scoring path **as a unit**. Its embedding and its score go into the network,
   together with the mean embedding over all paths (a "how good are the options in general" signal). Nothing is mixed
   across routes.

## What "best" means
- **For now:** "best" is whatever the network's score learns while predicting clear rate and picks. Gradients reach the
  score only through the winning path, as in ordinary max pooling.
- **Later:** once simulated or real runs give "P(clear the act) if I take this route", train the score on it directly.
  "Best" then literally means the highest-value route.
- **Path-choice head:** that same score gives a path-choice head at no extra cost. Walk the first step of the best route.
  The network already returns the index of the winning route for each option.

## Known limits
- **Hard max.** Only one route contributes detail. If two routes are close, the input can jump when the scores swap. A
  soft version (softmax over scores, or the top-2 routes) is an easy change if training is unstable.
- **Room types only.** Rooms are types, not contents: which elite or which event is unknown until entered. The burning
  elite (emerald key) is not encoded; it is not needed for Act 1 A20 without keys.
- **Act 1 only.** The design assumes Act 1 (one boss, 15 floors); the enumeration generalises per act.
