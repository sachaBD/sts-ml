# Phase 4: rollout-leaf learned search (no training)

Run `runs/schema=combat_v4/date=2026-10-06/id=counterfactual-rollout-leaf-v1/` (report `out/REPORT.md`, manifest, ledger,
logs/{smoke,main}.log). Executor single-fight-opus. Code: `rollout.py` (frozen copy and sha256 in run `out/frozen`).

Setup: frozen pv_worker 58c5c4ab running `play <rescue-u3 ONNX> 2000 --rollout-mix L`, L = 0.5 and 1.0, on the same 100 dev
starts. Leaf value = (1−L)·V_net + L·GuidedRollout (BattleScumSearcher2 mode 2, run on the leaf's public particle; 100/0
units). An internal rollout still undecided after 512 actions scores 0. Priors, root value and the depth cutoff stay
network. Semantics and units come from reading the source, corroborated by the flag and parity gates; this is not a
proof about the binary.

Result 2026-10-06 22:48 (200/200 games, 0 caps/errors, about 7.5 s/game, 2.6 min wall):
- Both arms won 76/100, against rescue 66 and MCTS 74.
- vs rescue: mix0.5 +10pp (12/2 discordant, McNemar p=.013); mix1.0 +10pp (15/5, p=.041).
- vs MCTS: +2pp for both, 95% interval about −6 to +11pp; not established.

These are reused development starts, so this is not confirmation. Next (pending Astra): fresh confirmation, then distillation.
