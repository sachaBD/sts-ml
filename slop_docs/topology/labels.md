# Value labels: `shift` mode

See assessment.md, "The label issue". Code: `assign_targets` in python/sts_combat_rl/training/data.py.

## `label = shift` (opt-in; the default stays `blend`)

For each decision row d, the outcome correction is δ_d = z_d − v_d. It is 0 if `decision_index` ≤ the
fight's last random-move `decision_index` (the same rule `blend` uses).

- decision row d: target = v_d + blend·δ_d. This is identical to `blend`.
- child row c: target = v_c + blend·δ_parent. The parent is the decision row with the same
  (`episode_id`, `decision_index`).
- Targets are clipped to [0, 1], and the number of clipped rows is printed.
- If any child has no parent, it raises an error. Use `shift` only with queries that include decision
  rows.

This keeps the teacher's ordering between siblings and moves each sibling group towards z. The
chosen action no longer gets a bonus just for being on the trajectory. `--blend` is the shift weight.

Use it with `--label shift` in train_value.py, or `label = "shift"` in the apps/value_train config.

## Diagnostics (act1-all-bosses-a20-scaled-search, 4,507,202 rows, blend = 0.5)

"Bonus" is the mean target of decision rows minus the mean target of child rows.

| category | rows | child share | blend bonus | shift bonus | blend mean | shift mean | shift clipped |
|---|---:|---:|---:|---:|---:|---:|---:|
| boss | 1,123,731 | 0.73 | +0.060 | +0.026 | 0.153 | 0.178 | 1.63% (18,272) |
| easy | 1,199,815 | 0.69 | +0.033 | +0.011 | 0.644 | 0.660 | 0.00% (9) |
| elite | 1,089,514 | 0.73 | +0.072 | +0.042 | 0.289 | 0.311 | 0.85% (9,223) |
| event | 126,893 | 0.77 | +0.055 | +0.020 | 0.322 | 0.348 | 0.85% (1,072) |
| hard | 967,249 | 0.78 | +0.044 | +0.021 | 0.528 | 0.546 | 0.17% (1,644) |
| all | 4,507,202 | 0.73 | +0.059 | +0.032 | 0.402 | 0.422 | 0.67% (30,220) |

Notes:
- What's left of the bonus is mostly the teacher's own sibling gap: children average about 0.02–0.04
  below their parent's v. Clipping at 0 raises the child means a little.
- All clipping is at 0; none is at 1. Most clipped rows are children in lost boss or elite fights,
  where δ_parent is negative and v_c is already near 0.
- 482 decision rows have a slightly negative root_value (min −0.0006, a rounding artefact). `shift`
  clips these to 0; `blend` does not.
- Clipping is 1.6% on bosses, which is more than "very few". If that matters, use a smaller shift
  weight, or scale δ multiplicatively near 0.
