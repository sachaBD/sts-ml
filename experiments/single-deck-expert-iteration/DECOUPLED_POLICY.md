# Evaluator-decoupled policy intervention (Demon Form)

Approved by single-fight-astra 2026-10-06; staged approvals, no automatic follow-on.

Question: with the original update15 evaluator (value) fixed, does replacing the search prior with the
teacher-fitted policy (Phase 2 unweighted epoch 30, fixed pre-selected endpoint) change fixed-2k-simulation play?
It is not meant to isolate dose or weighting, establish teacher optimality, or support equal-wall-time claims.
The visited state distribution can change because the policy changes.

Implementation: one composite ONNX graph, (value of A, logits of B) on the same six inputs. It is loaded by the
unchanged frozen worker (sha `58c5c4ab…`), so every root and leaf evaluation uses A's value and B's logits; the
native encoder and legal-move order are shared. No C++ or frozen artifact changes. Code:
`decoupled_policy.py` (stages export/gate0/gates12/seeds/main/report), tests `test_decoupled_policy.py`
(routing, padding mask, eval-mode/no dropout, dynamic batch/token/action shapes vs ORT, fail-closed schema/metadata,
seed generation). Run dir `runs/schema=combat_v4/date=2026-10-06/id=demon-form-decoupled-policy-v1/out`
(`export.json` fingerprints all ONNX + external-data files and sources).

## Gates (all passed exactly)

- G0 (no gameplay, native `pv_worker evaluate`, 7,031 teacher+monitor states, 430 mixed-size batches):
  composite(A,A) = original A ONNX bitwise (values + legal logits); composite(A,B) value = A bitwise and logits =
  standalone B bitwise (7,031/7,031); native B vs torch B max |Δlogit| 3.8e-6, |Δvalue| 1.1e-4; tampered contract
  rejected by worker and checker. Informative: native outputs are not batch-composition invariant (10/7,031 values,
  22/7,031 logit vectors differ in last bits under a different batching), in both arms alike.
- G1 (5 consumed monitor seeds): original A reproduces recorded iter015/eval actions and root visits 5/5.
- G2: composite(A,A) reproduces G1 actions, visits and root values 5/5. Wall/game 4.4 vs 5.3 s (n=5).

### Protocol deviation (recorded before main dispatch)

The proposal stated a 1e-4 native-vs-torch tolerance for B; the script used 1e-4 for logits but 1e-3 for value.
Observed: B logits 3.8e-6 (pass); B VALUE 1.1e-4 exceeds the stated 1e-4. B's value output is unused by the
composite (value comes from A). Reviewer (single-fight-astra) waived the tolerance for the unused B value only;
no relaxation for actual composite outputs, whose native comparisons (A value, B logits) are bitwise exact.

Gate runtimes/logs: export finished 12:16:20, G0 finished 12:17:12 (start not logged; run under a 58 s timeout);
G1 5 games ~7 s, G2 5 games ~7 s (gates12 finished 12:17:46); logs `out/logs/{export,gate0,gates12,g1,g2}.log`,
results `out/{gate0,gates12}.json`, game outputs `out/gates12/{g1,g2}`. Process note: the 10 gate games ran under the
conditional approval without a separate launch relay; future >1 min launches wait for relay ACK.

## Main (development seeds, not final confirmation)

100 fresh seeds `demon-form-decoupled-policy-v1:dev:*`, starts sha `0221e66d…`, frozen before any main game;
disjoint from 120,808 recorded seeds in 74 starts-like files (all monitor/bootstrap/final-reserved/learner/
teacher/library selection/human pool). Loadout identical to the monitor row except seed/RNG/IDs.
Baseline original update15 ONNX vs intervention composite(A,B); 2k sims, no exploration, 10 workers, no retries.

Run: background job, bg_wait exit 0; baseline finished 12:21:19, intervention 12:22:56, report 12:22:59 (about
3 min total). 200/200 games completed, no errors/caps/retries; no pv_worker processes remained afterwards.
Command: `PYTHONPATH=. timeout --kill-after=15 1800 .venv/bin/python experiments/single-deck-expert-iteration/decoupled_policy.py main`.

| | Baseline (update15) | Intervention (A value + B policy) |
|---|---:|---:|
| Wins / intended 100 (completed pairs 100) | 62 | 64 |
| Wall s/game mean / median (n=100) | 6.58 / 6.46 | 8.47 / 8.29 |
| Searched-decision s/game mean / median | 6.34 / 6.36 | 8.19 / 8.17 |

Paired gap intervention − baseline **+2 pp**, paired SE 4.7 pp, approximate 95% normal interval **−7.2 to +11.2 pp**
(conditional on these 100 development seeds and one fitted policy model; no training-seed variability).
Discordance: both won 52, baseline-only 10, intervention-only 12, both lost 26 (exact McNemar two-sided p 0.83).
Timing variability from system load is unquantified; intervention costs ~29% more time at fixed 2k sims.

Descriptive (decision counts, not independent trials; replay-encoded with the frozen worker):
Dual Wield selections baseline Clash+ 46 / Boomerang+ 46 / Anger+ 5 / Bite 1 / Cleave+ 1;
intervention Clash+ 70 / Boomerang+ 20 / Anger+ 9 / Bite 3 / Cleave+ 1. Demon Form played in 100/100 both,
mean first turn 1.85 vs 1.87; Corruption 99/100 both, 1.98 vs 2.03.

Interpretation: the teacher-fitted prior visibly changed search behaviour at the flagged decision (Boomerang
copies 46→20) with the original evaluator fixed, but produced no detectable win-rate change on 100 seeds;
the interval cannot exclude effects of roughly ±7–11 pp. Observation, not a causal conclusion that Dual Wield
choice is unimportant; one fitted model, one seed set. No further arms launched.
