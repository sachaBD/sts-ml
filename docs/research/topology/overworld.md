# Selected Act 2 overworld topology

The neighboring `assessment.md` and `plan.md` concern **combat** Deep Sets networks. The selected
`experiments/act2/REPORT.md` overworld incumbent is a different model:
`runs/schema=value_net_v1/date=2026-10-02/id=act2-r02-train/out/model.pt`.

New, untrained successors are now implemented: [v3.1/v3.2](overworld-v3.md), with
[a redesigned backward graph map encoder](overworld-map.md). The selected incumbent is unchanged.

## Interactive visual report

[Open the self-contained HTML report](../../../runs/schema=topology_report/date=2026-10-02/id=act2-selected/out/index.html).
Open the file in a browser; no server or internet connection is required. Generated outputs remain
local under `runs/`, so on another checkout regenerate them:

```sh
.venv/bin/python -m apps.topology_report.generate
```

The report includes architecture flow, parameter allocation, checkpoint SHA-256/provenance,
card-pair interaction heatmap, per-deck card-addition values, HP sensitivity, and a route-order
invariance demonstration. Raw probe data is alongside it in `probes.json`.

## Findings

- `run_policy_v1`: **67,800 total parameters**, token width 32, head width 64, one residual block.
  The total includes HP and policy heads that the Act 2 learner does not train or use.
- Cards are independently encoded with boss/HP/floor context, then summed and averaged. The
  nonlinear downstream head can represent combinations, but there is no explicit card–card
  encoder or attention. Card mechanics are not explicit inputs: relevance must be learned from IDs.
- Relics and potions are pooled separately. Relic–card effects are possible only downstream.
- Output is expected **floors score**, not calibrated clear probability. The checkpoint records
  9,000 training runs, TD lambda 0.7 and data decay 0.85.
- **Map order is lost:** summing `room_embedding + floor_embedding` separates the two sums.
  Given the same occupied-floor mask and room counts, swapping a campfire and elite leaves the
  path vector unchanged up to floating-point rounding. Legal routes are still enumerated by the
  worker; this limitation is specific to the neural path summary.

### Checkpoint probes (32 contexts)

Deterministic first qualifying Act 2 state per run file, sorted by filename, floor >=24, with
1–100 remaining paths. This is a non-random collection-data sample, not held-out evaluation.
Each of ten unupgraded cards is added singly and in pairs to the recorded deck. Interaction is
`L(deck+A+B) - L(deck+A) - L(deck+B) + L(deck)` in **logit units**.

| Combination | Mean interaction ± 1 SE across contexts |
|---|---:|
| Body Slam + Barricade | +0.00700 ± 0.00204 |
| Corruption + Feel No Pain | +0.00564 ± 0.00179 |
| Corruption + Dark Embrace | +0.00548 ± 0.00164 |
| Inflame + Heavy Blade | −0.00236 ± 0.00154 |

These are evidence of model non-additivity, **not proof of correct gameplay synergies**. Generic
head curvature, deck-size effects and route switching contribute. Counterfactual decks can be
out of distribution. The report selects extremes from many pairs; these are not prespecified
significance tests. SE describes variability in this sample, not population uncertainty.

The route-order probe produced a maximum path-vector difference of `2.98e-8`, with equal output
scores at displayed precision. Reversing deck order changed its logit by `1.79e-7`, consistent
with permutation invariance and floating-point summation.

## Recommendation

First test binding room identity to floor before pooling (joint room×floor table or per-slot MLP
on concatenated embeddings). Use a new frozen architecture kind/spec, not an edit to the selected
checkpoint's architecture. Improve targets for difficult HP/upgrade decisions as recommended by
Act 2's report. Deck attention is a useful controlled follow-up, not an established upgrade:
compare paired held-out gameplay at equal cost.

The Act 2 report's v2 reload warning is historical. Current `run_policy_v2.py` contains a compatibility
loader for old scorer keys, with tests in `agents/overworld/value/test_run_policy_v2.py`. That does
not make v2 the selected incumbent.
