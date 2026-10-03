# The Champ — step 1: what decks win? (2026-10-03)

Question: Act 2 boss losses — combat execution or deck? Step 1 looks only at the deck (and HP) at the start of
recorded Champ fights. Observational: correlations, not causal effects.

## Data

- All Champ boss fights in the act3-heart combat_v4 recordings (`runs/schema=overworld_v1/*/id=ah-*/out/combat`,
  collections c01/c02/c04 with ε 0.05 + all dev/fresh greedy plays; ah-c03 and the running ah-fresh-r0b-v1td not
  included). Guided-rollout MCTS, 20k sims, 8 particles. 1,817 fights; **1,706 after removing exact duplicates**
  (ah-dev-hyb = ah-dev-inc through Act 2). Overall win 41.6%, mean start HP 59.
- Scripts: `load.py` (→ `champ_fights.parquet`), `cards.py`, `demon_form.py`, `no_df.py`, `df_offers.py`,
  `pick_rates.py`; outputs in the matching `.log` files. ± = 1 binomial SE.

## Findings

1. **Demon Form is the dominant factor.** With it: **80% win (n=371)**; without: **31% (n=1,335)**. Mean HP
   almost the same (61.6 vs 58.3). The gap holds in every HP band:

   | start HP | no DF | DF |
   |---|---:|---:|
   | < 40 | 8% (282) | 38% (53) |
   | 40–55 | 22% (201) | 66% (58) |
   | 55–70 | 37% (324) | 84% (106) |
   | ≥ 70 | 43% (528) | **96% (154)** |

   Unupgraded ≈ upgraded (80% vs 79%); two copies 91% (n=32).
2. **Without Demon Form, Strength scaling shows a clean dose–response** (no DF, HP ≥ 55, n=852): Inflame /
   Spot Weakness / Limit Break copies 0 → 30% (399), 1 → 44% (303), 2 → 57% (111), 3+ → 82% (39). Top single
   cards: Spot Weakness 59%, Juggernaut 54%; top relic Shuriken 75% (n=52, also Strength).
3. **HP matters a lot but less than scaling:** no-DF decks at ≥ 70 HP win 43%, DF decks at 40–55 HP win 66%.
4. **The combat agent can use Strength scaling** against Champ: a 96% win rate with DF at full-ish HP is not the
   signature of a search that can't play scaling. Execution errors may still exist (20% DF losses, including 23
   at ≥ 55 HP, often dying fast: ~25–35 decisions) — step 2 / the decompressed traces test that.
5. **The overworld already takes Demon Form when offered** (fresh, Acts 1–2 card rewards): incumbent 71%,
   v3.2-distilled 88% of offers. DF is offered in only ~25% of runs (mostly Act 1 boss rewards), so its presence is
   largely luck of the rare roll — which also makes finding 1 less confounded than most deck correlations.
   Pick rates when offered (incumbent / v3.2d): Inflame 86/81%, Spot Weakness 58/81%, **Limit Break 12/1%**,
   **Feel No Pain 5/0%**, **Barricade 0/0%**, Shockwave 92/99%, Uppercut 81/96%.

## Reading

For Champ, the dominant problem is the deck, not execution: ~75% of decks reach Champ with little or no Strength
scaling and win ~30–40% even at high HP. The picker is fine on Demon Form/Inflame; the gap is (a) getting *more*
scaling (dose–response continues to 3+), and possibly (b) cards it never takes (Limit Break, Barricade, Feel No
Pain), whose value is unknown here because they're almost never in decks. Caveats: observational; collections
include ε-random picks; per-card rates are confounded with each other and with run quality.

## Next

- Step 2 (rebuild, causal): add Inflame+/Spot Weakness/Demon Form to real losing no-DF decks and replay the same
  fight → measured marginal value of scaling under our combat agent. Also replay DF losses with the oracle to see
  whether those are execution errors.
- Decompressed traces (when available): turn of enrage vs Strength at that point, for DF wins vs losses.
