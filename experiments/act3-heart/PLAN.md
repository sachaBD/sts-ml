# Act 3 + Heart — overnight 2026-10-03 (approved by user ~23:20 UTC; run until user returns)

True goal: strongest autonomous A20 Ironclad agent; the metric is the Heart kill rate over the full game.
Tonight (instrumental): (1) the full game (Neow→Heart, real keys) as a working, crash-free data loop;
(2) a full-game corpus: overworld_v1 records with v1 public observations (keys, map graph) + combat_v4 MCTS fights;
(3) honest baseline numbers (Heart, Act 1/2/3 clears, keys, boss entry HP, s/run) on paired dev seeds;
(4) first answer on overworld topology: can v3.1 / v3.2 learn the full game from this data and beat v1 in play?

Decisions (user-agreed):
- Combat frozen: guided-rollout MCTS, budgets 500/5k/10k/10k/20k (Heart = boss), 8 particles. No rest lookahead.
- Target `full` = 0.3 floor/56 + .05 [A1] + .10 [A2] + .15 [A3 both bosses] + .10 [entered Act 4] + .30 [Heart].
- Key crutch for the key-blind v1 net (`--key-rule`): ruby recall at the last Act 3 campfire; route toward the
  burning elite in Acts 2–3 while emerald is missing. Sapphire (worker, all policies): first chest in Act 3 only.
  v3 nets see keys/burning elite and should learn keys; evaluate them with and without the rule.
- Seeds: collect 971e9+, dev 961e9+ (600, paired), fresh 981e9+ (final only).
- Hand-written rules are crutches; prefer learned behaviour and data.

Stages: smoke → collect 2×2000 (v1+rule, eps .05, route .15) + dev baseline → train v1/v3.1/v3.2 on GPU
(offline per-act comparison) → paired dev gates → round 2 with the winner → fresh final + report. Continue while time remains.
Agents: impl-26-10-3 (contained code), orch-26-10-03 (runs launch scripts, monitors, SRE).
