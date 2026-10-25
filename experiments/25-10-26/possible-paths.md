# Possible paths forward (25 Oct 2026)

Context: `experiments/multi-fight-champ-expert/`. Ten-deck model beats MCTS20k on its training decks
(+14.1 pp); on unseen decks the corpus pilot reached −4.5 pp [−11.0, +1.2] vs MCTS20k after 4 rounds.
Main complaint: sample/compute inefficiency (2k-sim PUCT + rollouts, ~8–9 s/game, ~4k games/h on 10 workers).

| path | fixes | effort | status |
|---|---|---|---|
| **A. Gumbel at 128–256 sims + improved-policy targets** | CPU per game; poor policy targets at low sims | low–medium: root Sequential Halving already in `search.hpp` (7 Oct, never played/trained) | **chosen → `a-gumbel/`** |
| B. Full search on ~25% of moves (KataGo playout cap randomization) | games per CPU hour → more value data | low | later, stacks on A |
| C. Keep subtree after moves that reveal no information | re-searching near-identical states | medium; awkward with Gumbel | decide after A timings |
| D. Merge transpositions / duplicate actions | branching | medium (duplicate check: low, unverified) | later |
| E. Less noisy value targets (mix search root value, `--value-mix`) | one win/loss bit per fight | low | candidate |
| F. JAX/mctx pipeline (`agents/combat/az`) | GPU throughput | high; untrained, partial cards, GPU-saturated | parked |
| G. More rounds of current recipe | unseen-deck gap | compute only | fallback |
