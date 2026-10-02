# Overworld map encoder: backward graph reasoning

**Implemented in:** `agents/overworld/value/{graph_encoding,run_policy_v3}.py`.
Both `run_policy_v3_1` and `run_policy_v3_2` use this same map encoder.
Scope: A20 Ironclad, Acts 1–3 and the Act 4 / Heart route. These are new, untrained
architectures—not demonstrated improvements in clear rate yet.

## Why replace the old encoder?

The old route summary sums `room_embedding + floor_embedding`. That separates
into two sums, losing which room belongs at which floor. It cannot distinguish
**campfire → elite** from **elite → campfire** when counts and occupied floors match.
It also summarizes enumerated complete routes rather than shared map structure.

## What the new encoder sees

The **reachable directed acyclic graph** of future rooms:

- Room type, floor position, distance from our current room, burning-elite flag.
- Legal directed edges to the next row; shared suffixes remain shared.
- Our after-state: deck, owned relics/counters, potions, boss, HP, gold, keys,
  progression and reward/event/encounter information.
- An explicit boss sink for Acts 1–3. Act 4's real rest → shop → Shield/Spear →
  Heart nodes are used directly. The hidden second A20 Act 3 boss is not revealed.

Only the current act's visible map is available. Future acts' actual maps are not
invented or revealed. Lane coordinates identify connectivity but are not embedded
as semantic features; rearranging lanes consistently should not change meaning.

## How it reasons

```text
Known boss / run context
          ↓
Encode last future row
          ↓
For each earlier room, summarize its legal successors:
    highest-scored successor + mean + learned soft-choice average + branch count
          ↓
Update the room representation with its own features and successor summary
          ↓
Repeat backward to our immediate legal destinations
          ↓
Summarize those destinations → after-state value head
```

Room and floor embeddings are **concatenated and mixed by an MLP before** backward
message passing, so the association is preserved. Successor scores are conditioned
on the run after-state. Soft pooling supplies gradients to alternatives, while
best/mean/count summaries retain preference and branching information.

For a path choice, the worker sets `map.next_xs` to that choice's first edge.
The encoder prunes unreachable nodes and scores the resulting committed graph.
This prevents every path option accidentally receiving the same full-map value.
Standard maps use 16 rows × 7 lanes (including the virtual boss), with row-local
adjacency; no complete-path enumeration is needed by the network. The exporter
still retains legacy route strings for old checkpoints.

## What this does—and does not—guarantee

**It preserves:** order, legal connectivity, shared suffixes, branching opportunities,
and dependence on our deck/HP/resources. Costs scale with nodes/edges and choices,
not the number of complete routes.

**It does not simulate:** future fights, HP changes, shops or deck edits along a route.
This is learned message passing, not an exact Bellman solver. It must learn those
consequences from training targets; short simulator lookahead remains complementary.
Internal branch scores are latent routing scores, not calibrated win probabilities.

**Compatibility:** new versioned observations and retained `map.nodes` are required.
Legacy path-only logs are rejected by v3 instead of being silently reconstructed.
Old v1/v2 checkpoints and their input encoders remain supported.

**Tests:** swapping campfire/elite order changes the representation/value; committing
different first edges changes reachability; boss/Act 4 graphs, burning flags,
permutation/padding invariance, gradients and checkpoint round trips are covered.
Gameplay comparisons at equal inference/search cost are still required.
