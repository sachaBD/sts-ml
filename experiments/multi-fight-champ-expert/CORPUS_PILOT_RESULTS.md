# Corpus pilot results (run 9–10 Oct 2026)

Run: `runs/schema=combat_v4/date=2026-10-09/id=champ-corpus-pilot-v1` (protocol: [CORPUS_PILOT.md](CORPUS_PILOT.md)).
Started 2026-10-09 19:12; baseline evals 19:25/19:32; restarted after cap fix 23:14; finished 2026-10-10 01:05.
Final model: `out/round4/model/`. Dashboard: `corpus_dashboard.sh`.

## Verdict

Partial result. Broader training clearly helped on **unseen** decks: round 4 vs the starting model
+6.5 pp [+1.8, +12.3]. Whether it matches or beats MCTS20k on unseen decks is **inconclusive**:
−4.5 pp [−11.0, +1.2]. One training run; 20 dev families, 5 of which nobody wins (0/20 for every arm).

## Seen vs unseen decks

| result | decks | model − MCTS20k |
|---|---|---:|
| Ten-deck update 6 ([TEN_DECK_RESULTS.md](TEN_DECK_RESULTS.md)) | 10 **training** decks, fresh seeds | +14.1 pp [+12.2, +16.0] |
| Same model = pilot round 0 | 20 **unseen** dev families | −11.0 pp [−18.8, −3.8] |
| Pilot round 4 (+40 new training families) | same 20 unseen families | −4.5 pp [−11.0, +1.2] |

The ten-deck gain did not transfer to new decks; the pilot closed about half of that gap.

## Development evaluation (20 unseen families × 20 paired seeds; greedy; 0 caps/errors)

| agent | wins/400 | vs MCTS20k (95% CI) | vs round 0 (95% CI) |
|---|---:|---:|---:|
| MCTS20k | 201 (50.3%) | — | — |
| round 0 | 157 (39.3%) | −11.0 [−18.8, −3.8] | — |
| round 1 | 163 (40.8%) | −9.5 [−16.8, −2.5] | +1.5 [−3.8, +6.8] |
| round 4 | 183 (45.8%) | −4.5 [−11.0, +1.2] | +6.5 [+1.8, +12.3] |

Intervals: family-cluster bootstrap from the controller. Rounds 2–3 were not evaluated (by protocol).

By deck type (wins/80): MCTS / R0 / R1 / R4

| type | MCTS | R0 | R1 | R4 |
|---|---:|---:|---:|---:|
| block | 60 | 39 | 44 | 55 |
| demon_form | 50 | 47 | 46 | 51 |
| exhaust | 31 | 23 | 23 | 22 |
| mixed | 15 | 15 | 22 | 19 |
| strength | 45 | 33 | 28 | 36 |

About half of the R0→R4 gain (+13 of +26 wins) comes from two block families. Exhaust and strength lag MCTS
most (worst families: −9 and −6 wins of 20). Four families per type: descriptive only.

## Notes

- Round 1 learner collection first halted on a capped fight (fail-closed gate); user fixed it and resumed
  at 23:14. Starts files and cached baseline evals (MCTS20k, round 0) predate the halt and were reused
  unchanged, so the comparison uses identical starts/worker. How the capped training games were resolved
  is not recorded here.
- Collection (exploration on) win rates, rounds 1–4: teacher 59/67/49/56%, new-deck learner 48/56/46/50%,
  revisit learner –/53/56/58%. Different waves of decks per round; not a learning curve.
- Wall time for rounds 1–4 plus R1/R4 evals: ~1h52m with 10 workers (excludes the failed first attempt).
- Not measured: retention on the original ten decks (replay-only); final 100-family test untouched.
