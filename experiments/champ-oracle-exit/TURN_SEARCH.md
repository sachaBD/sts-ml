# Turn-level oracle search — design note (v1: evaluation only)

Question: with the same network (champ-ox-c-r09), does searching over whole turns (deeper) beat the per-action PUCT
tree in oracle mode on the 409 bench? If yes → use it to generate training data (v2). If no → depth is not the bottleneck.

## Isolation (clean, separate, disable-able)
- New files only: `agents/combat/pv/turn_search.{hpp,cpp}` (+ test). Reuse the exact state key from the census tool
  (c1d687d) via a shared header; do not modify `search.hpp` or the per-action search's behavior.
- Worker: `pv_worker play MODEL E --oracle --turn-search` (E = expansions per turn). Requires `--oracle`; rejected
  otherwise. Without the flag, behavior and outputs are byte-identical to today. Revert = drop the flag / the commit.
- play.py passes the flag through; agent string tag `turn_search=1 E=… caps=…`.

## Algorithm (deterministic: true state, no particles)
- **Node** = a player-turn start state (first decision of a turn). **Children** = distinct non-terminal end-of-turn
  outcomes, each stored as (one action sequence reaching it, the next turn-start state after END_TURN and the enemy
  turn, its V). Terminal outcomes are children with known value (100 win / 0 loss); duplicates merged by outcome.
- **Expand(node):** DFS all legal action sequences to END_TURN (card choice screens included); dedupe by exact key
  of the resulting next-turn-start state; one batched network call for V of all distinct children.
- **Search:** best-first PUCT over turns. Each expansion = select from root by
  `Q(child) + c · P(child) · sqrt(N) / (1 + n)` with `P = softmax(V_child / T)` over the node's children, Q = backed-up
  mean (initialized to V_child); expand the reached leaf; back up the max child value of the new node. E expansions
  per turn decision. Reuse: the played child's subtree becomes the next root.
- **Play:** child with most visits (ties: higher Q); execute its stored action sequence step by step (recorded as
  normal combat_v4 actions so replay works).

## Hard caps (never exceed; 100% of turns either searched within caps or explicitly fall back)
- `max_sequences` per expansion (default 20,000), `max_children` (default 512), `max_seconds` per expansion (1 s).
- If any cap is hit at the **root** expansion → that turn is played by the existing per-action PUCT (800 sims);
  record `fallback=1` and the reason. A non-root node that hits a cap is treated as a leaf (its own V), never partially
  expanded.
- Abort (error) on pending action-queue callbacks at an undecided endpoint, as the census tool does.

## Telemetry (per turn decision)
seconds, expansions, network evaluations, root children, max depth in turns of the tree, mean leaf depth (turns),
fallback flag + reason.

## Tests
- Every child's stored sequence, replayed from the parent state, reproduces the child's exact key.
- Turn search on a tiny fixture finds the same best turn as exhaustive 1-turn enumeration when E = 1.
- Without `--turn-search`: per-action outputs unchanged on a fixed fight (byte-identical result line).

## Experiment (orch runs; r09 model; 409 bench; oracle)
Arms: per-action PUCT 800 (existing r09 bench-oracle: 41.3%) vs turn search E ∈ {16, 64, 256}; T = 10 (V units),
c = 1.25 on min-max normalized Q. Report win %, paired b/c + p vs per-action, s/fight, mean/p90 depth in turns,
fallback %. Smoke on 20 fights first. Run only on cores not used by training (≤ 1–2 workers while exit tag c runs).

## Clarifications (agreed with impl, 17:45)
- Tiny wiring edits allowed (worker.cpp, play.py, CMakeLists, shared key header); algorithm in new files.
- E counts every descent incl. root expansion and capped/terminal leaf backups; E=1 = exhaustive 1-turn.
  Q = V until first backup, then mean of backups. Child V clamped to [0,100].
- Choice screens during END_TURN are enumerated as part of the turn; key at the next turn's first decision.
- Time caps are cooperative (may overshoot by one synchronous call; logged). 512 actions/path.
- Children store (sequence, V, key) only; state rebuilt by replay on expansion. Tree cap 256 MB/process
  (root → fallback; non-root → leaf; reuse dropped if exceeded).
- Unchanged-behavior test: byte-identical stable JSON with timing fields removed.
- A negative result speaks to this search design, not to whether depth matters in general.
