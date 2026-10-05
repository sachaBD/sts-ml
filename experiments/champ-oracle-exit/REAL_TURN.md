# Real-play one-turn expectimax ("real-turn") — design note

Problem: per-particle (PIMC) turn search lost 31.3% vs 39.1% (strategy fusion: each particle maximizes over its own
future plans with its draws known). The value net already sees only public information, so it already averages
over hidden state; the long horizon should live in V (trained from deep oracle turn search), while the real-play
search only needs to choose *this turn's* actions without peeking.

## Algorithm (at every non-forced real decision; re-plan each decision)
1. Sample K public-belief particles (existing public_particles; K=8 default).
2. Enumerate action sequences from the current decision, applying each action to all K particles in lockstep
   (actions matched by public action key, as the PV tree does).
3. A sequence ends (leaf) at the first point where information is revealed or the turn ends:
   - END_TURN executed (enemy turn + draw follow), or
   - after an action, the particles' public observation keys are not all equal (a draw/random outcome revealed), or
   - terminal outcome in any particle.
4. Leaf value = mean over particles of value(particle state): 100/0 if terminal, else network V clamped [0,100]
   (batched). No max over per-particle continuations → no strategy fusion.
5. Dedupe leaves by the tuple of per-particle exact keys (reuse the exact key) to save evaluations.
6. Play the first action of the best leaf sequence (ties: higher mean V of the action's best leaf, then stable order).

## Caps / fallback (same philosophy as turn search)
max_sequences 20,000, max_leaves 2,048, cooperative max_seconds 2 s per decision; 512 actions/path. Any cap hit →
this decision is played by the existing per-action real PUCT (2000 sims, 8 particles); record fallback + reason.
Pending action callbacks at a leaf → error as before.

## Isolation
New files; flag `play --real-turn [--particles K]` (incompatible with --oracle / --turn-search). No-flag behavior
byte-identical (timing excluded). Telemetry per decision: seconds, sequences, leaves, network evaluations, mean leaf
depth in actions, fraction of leaves ended by END_TURN vs reveal vs terminal, fallback + reason.

## Experiment
D5 model, bench2k: real-turn K=8 vs per-action real2000 (paired). Smoke 20 fights first. Expect ~0.1–1 s/decision.
Success: real-turn ≥ per-action with p<0.05 → becomes the real-play agent (and the real bench of later EXIT runs).
