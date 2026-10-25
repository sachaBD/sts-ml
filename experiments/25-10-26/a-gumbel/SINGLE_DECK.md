# Single-deck Gumbel training from scratch — registration (PENDING APPROVAL)

Status: approved 25 Oct 2026; **amended 10:35** (see Amendment 1, which supersedes the 39-update arm and its tests).

## Amendment 1 (25 Oct, 10:35)

The v1 run (200 games/update) was stopped after update 13 of 39. Measured: play 200 games ≈ 20 s, training ≈ 4.5 min
per update (≈93% of wall time). "39 updates = equal CPU" counted gameplay only; with training included it would cost
~4–6× the PUCT recipe, so its result would not be a fair comparison. Replaced by:

- **Gumbel v2** (`a-gumbel-panache-v2`): 1,400 games/update (= gameplay CPU of 200 PUCT2k games), 9 states/fight
  (≈ the same training states per update as 200 × 64), 6 updates; bootstrap keeps 64 states/fight. Otherwise as v1.
  Caveat: teacher fights are also sampled at 9 states/fight, so the teacher's share of replay is lower than in PUCT.
- **PUCT2k control** as registered. Both run in parallel with 6 workers each.
- **Tests** (300 fixed seeds): Gumbel v1 update 6 (equal games), Gumbel v2 update 6 (equal compute), PUCT control
  update 6; all vs stored ten-deck PUCT2k 174/300 and MCTS20k 124/300. Update-39 arms dropped.
- Driver `run_panache.sh`, progress `status.py`. Estimate ~1.5 h.

## Question

Does expert iteration with Gumbel search at 256 sims + improved-policy targets reach PUCT-2k strength on one deck,
at equal games and at roughly equal learner CPU?

## Deck: `panache` (3a7d5ed7, 43/95 HP, no potions, original relics)

Panache+, Dark Embrace+, 2x Battle Trance, Metallicize, Dark Shackles. Chosen from the ten-deck PUCT run because it
has the largest gain over MCTS20k that is not at a ceiling: test 174/300 (58.0%) vs MCTS20k 124/300 (41.3%),
+16.7 pp [+10.3, +23.1] (`champ-ten-test-u6-v1`). A ~58% win rate leaves room to detect changes in both directions.
Rejected: barricade-searing (97% = ceiling), exhume-immolate (+15.3 pp, very noisy monitor curve), others (smaller gains).

## Stored reference (no replay)

| arm | source | panache test (300 fixed seeds) |
|---|---|---:|
| PUCT2k, ten-deck network update 6 (300 teacher + 6×200 learner fights per deck) | `champ-ten-test-u6-v1/out/ten` | 174/300 |
| MCTS20k | `champ-ten-test-u6-v1/out/mcts` | 124/300 |

The same 300 test starts are reused for every new arm, so all comparisons are paired.

## Recipe: identical to the ten-deck run except the search and its targets

Reused, not replayed: the 300 panache MCTS20k bootstrap fights and 50 monitor-teacher fights from `champ-ten-rollout-v1`.

| | ten-deck PUCT (reference) | this run |
|---|---|---|
| network | fresh width 64, sigmoid value | same, fresh, panache only |
| bootstrap | 300 MCTS20k fights/deck, 10 epochs, lr 1e-3 | the same 300 panache fights (reused) |
| collection search | PUCT 2000, rollout .5, Dirichlet noise + early-turn sampling | **Gumbel 256 (m=16, gumbel_scale 1 at root, interior PUCT), rollout .5; play the Gumbel-selected action; no Dirichlet / turn sampling** |
| policy target | root visit counts, flat-policy weighting | **improved policy softmax(logits + σ(completed Q)) over all legal actions; no flat-policy weighting** (it's a visit-count noise heuristic) |
| value target | 100 × win | same |
| update | 200 fights, last 10 batches + tapering teacher, 3 epochs, lr 3e-4, 64 states/fight | same |
| updates | 6 | **39** |

- 39 updates ≈ equal learner CPU: 1,200 PUCT fights × 9.1 s ≈ 10.9k s; Gumbel ≈ 1.4 s/fight → ~7,800 fights.
- Monitor: the same 50 panache seeds every update vs stored MCTS20k (descriptive learning curve only).

## Test: pre-declared, 300 fixed panache test seeds, greedy (gumbel_scale 0)

Primary:
1. **Equal games:** update 6 + Gumbel 256 vs stored PUCT2k (174/300).
2. **Equal CPU:** update 39 + Gumbel 256 vs stored PUCT2k (174/300) and MCTS20k (124/300).

Secondary (300 games, ~5 min): update 39 network + PUCT2k search, to separate a better network from cheaper search.
No other checkpoints are evaluated on the test seeds. Report paired differences with 95% CIs.

## Caveats

- The reference network was shared across ten decks with the same per-deck budget. Transfer from the other nine
  decks could help or hurt it; a single-deck run is not a perfect control. **Optional control** (~1 h, approval
  needed): the same single-deck run with PUCT2k (6 updates), giving a like-for-like reference.
- One training run per arm. 300 paired seeds: CI roughly ±6–7 pp.
- New worker build/simulator commit vs the stored reference (assumed equivalent, as in the gate).
- Gumbel noise is currently seeded from the observation only; collection will mix in the fight seed.

## Work and time

| step | estimate |
|---|---|
| Worker: allow root halving with `--explore` (Gumbel noise, fight-seeded), emit improved policy | 1–2 h |
| Record schema + encoder + trainer: improved-policy target, without breaking old visit targets; tests | 1–2 h |
| `combat_expert_iteration`: Gumbel config keys, reuse stored bootstrap/monitor fights | ~1 h |
| Smoke run (tiny budget) | ~10 min |
| Run: 39 updates × ~2 min (200 games ~30 s, encode + train ~1 min, monitor) | ~1.5 h (rough) |
| Tests: 3 × 300 games | ~10 min |

Total ≈ half a day of implementation + ~2 h compute. Results go to `runs/schema=combat_v4/date=2026-10-25/id=a-gumbel-panache-v1/`.

## Approval needed

1. Deck: panache.
2. 39 updates (equal CPU), with update 6 as the equal-games checkpoint.
3. Turning off flat-policy weighting for improved-policy targets.
4. Optional single-deck PUCT2k control: yes / no.
