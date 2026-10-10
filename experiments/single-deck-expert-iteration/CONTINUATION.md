# Teacher continuations from learner states — machinery, gates, protocol (labels NOT yet run)

Requested by single-fight-astra 2026-10-06. No gradient steps authorized. Run dir
`runs/schema=combat_v4/date=2026-10-06/id=demon-form-continuation-v1/out`; code `continuation.py`, tests
`test_continuation.py`.

## Semantics (read before interpreting)

update15 V(s) estimates the win probability of the *learner's* continuation; a teacher-continuation win frequency
T(s) estimates a *different policy* from the same public state. T − V or T − learner outcome is teacher
**opportunity**, not learner miscalibration, and does not by itself decide "value labels vs ranking".
Four Bernoulli continuations per state are very noisy: no per-state ground truth, AUC or dominance claims.
Learner outcome labels come from exploratory collection (root noise, sampled early turns) by round-6..15
models, not update15.

## Machinery (source-level audit)

Snapshot of `build/champ-viewer/champ_session` (built 2026-10-05 18:06; sha in `frozen/snapshot.json`) plus
copies of the relevant sources and git heads/dirty status; the shared build is never used or overwritten.
- State = start + learner action prefix (`act` ops; each checked `isValidAction` natively).
- `playout SIMS FROM N`: N complete teacher fights from public-belief particles FROM..FROM+N-1 of that state.
  Particle (`teacher_leaves.cpp sample_particle`): copy of the true state with seed and all RNG streams replaced
  and the draw pile reshuffled uniformly (only a Scry prefix or Frozen Eye preserved). Hand, discard, exhaust,
  player/monster public fields, current Champ intent (moveHistory) and other non-RNG internal fields (e.g. Champ
  stance counters) are copied from the true state. Known limitation: a card put on top of the draw pile by
  Headbutt is NOT kept on top (belief ignores that public knowledge); the learner's own search has the same
  belief model. Sampled states after a Headbutt in the same turn are flagged.
- Continuation player = frozen teacher policy semantics: fresh fair search per decision (8 particles, guided
  rollout, 20k sims, early stop), forced moves unsearched, no oracle, search salt = particle index + 1 (the
  recorded teacher used salt 0). Label = actual PLAYER_VICTORY flag. No turn cap inside a playout: each state
  runs in its own process with a wall timeout (error, no retry).
- Action branching via `act` then `playout` resamples particles from the *post-action true* state. For actions
  that reveal hidden information (end turn draws the next hand and rolls the Champ intent) this peeks at the actual
  future, so it is invalid for them. For non-revealing actions it is valid but branches use independent particle
  samples (no common random numbers). True common-random branching would need a small native change.
- update15's 2k greedy search action cannot be computed at an arbitrary mid-fight state with existing binaries
  (worker `play` starts from the start); only the raw prior (champ_session `pv`) is available without native code.

## Gates (run, <1 min, no labels) — all passed

- A replay/encoding (30 training states over rounds 6/10/15: Dual Wield and Headbutt selections, setup and other
  plays, 6 forced): legal-move menu identical in order 30/30; update15 value from the rebuilt state bitwise equal to
  frozen-worker evaluation of the stored training row 30/30; priors max |Δ| 2.8e-17.
- B teacher search parity (12 states: first 3 decisions of 4 recorded teacher-play fights on training starts):
  salt-0 search reproduces recorded visits per move, root value and simulation count exactly 12/12.
- C terminal (6 learner fights, 2 wins/4 losses): full action prefix reproduces recorded `won`; playout refused on
  a finished fight.
- D playout determinism deferred to smoke (consumes continuations).

## Proposed protocol (awaiting review)

Sample (to be frozen before any label): 300 of the 1,000 active-replay training fights uniformly (seed 20261007),
one searched decision state (>1 legal) uniformly within each; inclusion probability 0.3/n_decisions(fight)
recorded. Training starts only (asserted disjoint from monitor/final/dev). A dry run (not frozen) gave 173/300
learner wins; strata setup-legal 104, other 178, Dual Wield 11, Headbutt 7; 22 after a same-turn Headbutt.

Smoke (12 continuations): two states x 4 particles, plus state 0 repeated (determinism). Measures runtime.
Labels: 4 continuations per state (1,200), 10 parallel processes, per-state timeout 1,800 s, no retries.

Analysis (aggregate only; one state per fight => independent units; fight-uniform estimand unweighted, state-level
estimand with 1/inclusion weights): mean T, mean learner outcome, mean V with SEs; paired T − learner outcome
(teacher opportunity) and T − V; debiased mean squared error of V vs T, (V − Ȳ)² − Ȳ(1 − Ȳ)/(k − 1), which is
unbiased per state for (V − T)²; Brier of V vs learner outcome; the same by V bin, turn bin and stratum with
counts. No per-state claims.

Optional action screen (proposal only): 24 states preselected from the frozen sample by declared categories
(Dual Wield selections and setup-legal plays where the update15 prior argmax differs from the teacher salt-0
search choice; deterministic order, outcome-blind), 2 non-revealing actions each (update15 prior argmax vs
teacher recommendation; validity check that the action leaves draw pile / turn / Champ intent unchanged), 8
continuations per action with particle indices 100–107 (independent of the selection search) = 384. Independent
branch samples, so noisier than paired common random numbers.

## Review corrections (single-fight-astra, 2026-10-06)

- Teacher-continuation minus recorded learner outcome is a *descriptive* difference between teacher continuation
  and historical behaviour return (exploratory, older checkpoints, one observed continuation, root hidden state
  not resampled), NOT teacher advantage over update15. V − T is a cross-policy discrepancy, not calibration error.
  Uncertainty is conditional on the fixed models and sampling design (one state per fight avoids within-fight
  clustering; trajectories share adaptive training history), not blanket independence.
- Action screen deferred until root-before-action particle sampling exists; a draw/turn/intent check is not
  sufficient for all revealing or random effects, and the raw prior is not the played 2k action.

## Sample freeze and smoke (run)

Sample frozen 12:48:06–12:48:08: 300 states, sha `134db62c…` (identical to the dry run), 173 learner wins; no
duplicate encoded inputs among sampled states; 32 turn-0 states.
Smoke 12:48:15–12:48:26 (bg_wait exit 0): 12 continuations = 8 unique (2 states x 4) + 4 exact repeats of state 0.
All completed; legal menu matched the training row; repeat identical in won/hp/turns (deterministic). Per-playout
0.7–2.7 s. Results: state learner-10:1 step 9 (update15 V 99.9) teacher 4/4 wins; learner-10:10 step 24 (V 27.7)
1/4. Smoke labels carry `sampler = current-public-particles … known Headbutt top card NOT preserved`; the repeat
is not an independent label. Timeout: `subprocess.run(timeout=…)` kills the single champ_session child (it
spawns none); no champ_session processes remained.

Seed derivation (source): particle i of state s uses splitmix64 stream seeded by `publicObservation(s)` (a hash of
public fields only, not history), i-th draw; within a particle that seed sets all RNG streams and the draw-pile
shuffles. Teacher searches inside continuation i use `publicObservation(state) ^ splitmix(i+1)` at each decision.
Continuations are deterministic functions of (public observation, i): different states get unrelated streams;
identical public observations would get identical particles (none among the 300 sampled states).

## Known-information audit

Deck/relic mechanics that create public draw-order knowledge here: only Headbutt (discard card → top of draw).
No Frozen Eye, Scry, Warcry or similar in deck/relics (Burning Blood, Neow's Lament, Horn Cleat, Bronze Scales,
Dream Catcher, Tiny Chest, Runic Pyramid, Potion Belt, Anchor, Maw Bank); Anger copies go to discard, Dual Wield to
hand. Lifetime: a Headbutted card stays known on top until drawn (normally the next turn's draw; with Runic
Pyramid and a full hand it can survive a turn boundary); reshuffle only occurs after the pile is drawn empty.
Inference from consecutive public views (tested): known prefix k = +1 per Headbutt selection, minus cards drawn.

- Sample: 22/300 states have k = 1 (all after a same-turn Headbutt), 278 have k = 0.
- Whole pool (1,000 fights): 1,967 decision states k = 1, 4 states k = 2; 27,619 k = 0 (6.7% known); 4 states
  where a known card survives a turn boundary.
- Neither the PV encoder (draw pile as an unordered multiset) nor either search's belief represents the known
  top-of-draw ORDER. The Headbutt choice still visibly changes draw/discard multisets; only the order information
  is missing. A plausible shared limitation; not established as a strength ceiling.

Estimand for the first labels (P1, approved): sampled states with k = 0, i.e. current public-particle sampler with
no identified known-order violation. This does not prove full public-posterior correctness (RNG / internal-counter
dependencies / observation conditioning are not formally verified).

## Bulk labels (P1) — run 12:53:32–12:59:41, bg_wait exit 0

Eligible manifest frozen first (`eligible.json`, sha `4fdf8c09…`; 278 eligible, 22 excluded k=1). 2 states reused
from smoke (identical protocol; repeats not counted), 276 dispatched × 4 = 1,104 new continuations, 10 processes,
per-state timeout 300 s. 278/278 states completed with exactly particles 0–3; 0 menu mismatches; no champ_session
left. Analysis: `analysis.json` (`continuation.py analyze`).

Wins out of 4 per state: 0: 42, 1: 26, 2: 35, 3: 34, 4: 141 (states).

Aggregate (fight-uniform, n = 278 states/fights; state-weighted effective n 269; SE conditional on the fixed models,
frozen sample design and Monte Carlo):

| Quantity | fight-uniform | state-weighted |
|---|---:|---:|
| T: teacher continuation win frequency | 0.685 ± 0.023 | 0.704 ± 0.022 |
| historical learner behaviour outcome | 0.586 ± 0.030 | 0.623 ± 0.029 |
| V: update15 value | 0.634 ± 0.023 | 0.652 ± 0.023 |
| T − historical outcome (descriptive) | +0.099 [0.054, 0.144] | +0.082 [0.035, 0.128] |
| T − V (cross-policy discrepancy) | +0.051 [0.019, 0.084] | +0.052 [0.019, 0.085] |
| debiased mean (V − T)² | 0.056 [0.040, 0.073] | 0.056 [0.039, 0.073] |
| Brier V vs historical outcome | 0.126 | 0.129 |

By update15 value bin (states = fights; mean V / T / historical outcome / debiased MSE):
[0,.2) 65: .053 / .258 / .077 / .080; [.2,.4) 26: .310 / .510 / .231 / .141; [.4,.6) 22: .489 / .636 / .591 / .129;
[.6,.8) 25: .695 / .690 / .560 / .051; [.8,.9) 14: .848 / .679 / .786 / .069; [.9,1] 126: .990 / .950 / .905 / .014.
By stratum: Dual Wield 10: V .361 T .450; Headbutt 7: V .747 T .500; setup-legal 97: V .572 T .668; other 164: V .682 T .718.

Descriptive reading: V tracks historical learner outcomes closely in low bins; the teacher continuation wins
markedly more from low-V states (+20 pp in [0,.4)), while at V ≥ 0.9 both continuations fall short of V
(T .950, historical .905 vs V .990). This shows teacher opportunity concentrated in positions update15 rates as
likely lost; it does not show update15 value error with respect to its own policy (historical outcomes come from
exploratory older checkpoints; no same-root update15 continuation exists).
