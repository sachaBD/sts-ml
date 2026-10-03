# boss_bench_v1

Recorded boss fights rebuilt from overworld_v1 pre-fight macro states (apps/boss_rebuild) and replayed with
guided-rollout MCTS at one or more simulation budgets. `out/fights.jsonl`: sampled fights (pre-fight state, original
result); `out/results.jsonl`: one line per (fight_id, sims) replay; `out/summary.json`.
