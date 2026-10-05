# Human vs agent Champ decks (descriptive)

Humans: 145 Ironclad A20 Champ fights from the megacrit pilot (50 files, 2018-10..12 and 2020-10..11).
Ours: `experiments/act2/champ/champ_fights.parquet`, 1817 rows, 1706 after dropping rows identical in every column except `fight_id` and `run` (same seed and state replayed in several runs).
All rates are `% ± 1 binomial SE`; means are `mean ± SE`. **n is small; nothing here is a tested effect.**

Human sets: *all* (145; outcome and HP are always right), *exact* (60; whole start state reconstructed
cleanly), *base-exact* (108; card/relic contents right, but the upgrade state of some cards is unknown because the
dump's event logs omit "+N", or a bottled relic's card is unknown). Card, relic and scaling tables use *base-exact*
(base cards, ignoring upgrades). Mean upgraded cards uses *exact* only.

**Bias to keep in mind.** A run that died at Champ is always exact. A run that won picked a boss relic afterwards, and
Astrolabe / Empty Cage / Dolly's Mirror / Whetstone / War Paint etc. cannot be undone. So exact and base-exact sets
over-represent losses and their win rate is **too low**; use *all* for the human win rate. Win rates "with/without a card"
inherit this and are only descriptive. Our agent's decks are Act-1-heavy self-play runs, not human-comparable play
(different relics/potions/pick policy), so gaps are descriptions, not skill estimates.

## 1. Overview

| group | n | Champ win rate | mean hp/max_hp | mean hp | mean max_hp | deck size | upgraded cards | curses |
|---|---|---|---|---|---|---|---|---|
| humans, all | 145 | 69±4% | 0.73±0.02 | 57.14±1.57 | 78.72±1.24 | n/a | n/a | n/a |
| humans, base-exact | 108 | 58±5% | 0.72±0.02 | 56.19±1.70 | 78.73±1.22 | 22.75±0.44 | n/a | 1.44±0.08 |
| humans, exact | 60 | 38±6% | 0.71±0.02 | 54.37±1.96 | 77.75±1.63 | 23.33±0.64 | 7.20±0.46 | 1.48±0.12 |
| ours | 1706 | 42±1% | 0.78±0.01 | 59.04±0.50 | 76.14±0.24 | 24.29±0.08 | 7.16±0.10 | 1.31±0.02 |

## 2. Cards (base card, upgrades ignored; humans = base-exact, n=108; ours n=1706)

Top 40 by human presence. "copies" = mean copies over all decks. Win with/without = Champ win rate of decks with / without the card.

| card | humans in deck | humans copies | humans win with / without | ours in deck | ours copies | ours win with / without |
|---|---|---|---|---|---|---|
| ascenders_bane | 100±0% | 1.00 | 58±5% (n=108) / n/a (n=0) | 100±0% | 1.00 | 42±1% (n=1706) / n/a (n=0) |
| bash | 97±2% | 0.97 | 59±5% (n=105) / 33±27% (n=3) | 99±0% | 0.99 | 42±1% (n=1686) / 40±11% (n=20) |
| defend_red | 96±2% | 3.65 | 59±5% (n=104) / 50±25% (n=4) | 86±1% | 2.91 | 39±1% (n=1466) / 56±3% (n=240) |
| strike_red | 83±4% | 2.66 | 53±5% (n=90) / 83±9% (n=18) | 82±1% | 4.10 | 39±1% (n=1401) / 51±3% (n=305) |
| shrug_it_off | 55±5% | 0.66 | 61±6% (n=59) / 55±7% (n=49) | 48±1% | 0.68 | 43±2% (n=820) / 40±2% (n=886) |
| armaments | 32±5% | 0.35 | 46±8% (n=35) / 64±6% (n=73) | 30±1% | 0.37 | 42±2% (n=507) / 42±1% (n=1199) |
| battle_trance | 32±5% | 0.35 | 60±8% (n=35) / 58±6% (n=73) | 34±1% | 0.41 | 46±2% (n=576) / 39±1% (n=1130) |
| spot_weakness | 31±4% | 0.33 | 74±8% (n=34) / 51±6% (n=74) | 26±1% | 0.31 | 52±2% (n=446) / 38±1% (n=1260) |
| shockwave | 31±4% | 0.31 | 61±9% (n=33) / 57±6% (n=75) | 42±1% | 0.52 | 44±2% (n=721) / 40±2% (n=985) |
| pommel_strike | 30±4% | 0.40 | 59±9% (n=32) / 58±6% (n=76) | 63±1% | 0.98 | 45±2% (n=1083) / 37±2% (n=623) |
| true_grit | 30±4% | 0.33 | 72±8% (n=32) / 53±6% (n=76) | 6±1% | 0.06 | 37±5% (n=98) / 42±1% (n=1608) |
| whirlwind | 28±4% | 0.31 | 53±9% (n=30) / 60±6% (n=78) | 11±1% | 0.11 | 43±4% (n=183) / 41±1% (n=1523) |
| flame_barrier | 27±4% | 0.30 | 52±9% (n=29) / 61±5% (n=79) | 26±1% | 0.31 | 45±2% (n=446) / 40±1% (n=1260) |
| inflame | 26±4% | 0.32 | 64±9% (n=28) / 56±6% (n=80) | 36±1% | 0.43 | 47±2% (n=611) / 39±1% (n=1095) |
| twin_strike | 26±4% | 0.33 | 61±9% (n=28) / 58±6% (n=80) | 35±1% | 0.46 | 42±2% (n=605) / 42±1% (n=1101) |
| heavy_blade | 25±4% | 0.29 | 56±10% (n=27) / 59±5% (n=81) | 2±0% | 0.02 | 56±8% (n=36) / 41±1% (n=1670) |
| immolate | 25±4% | 0.26 | 52±10% (n=27) / 60±5% (n=81) | 26±1% | 0.29 | 38±2% (n=441) / 43±1% (n=1265) |
| offering | 25±4% | 0.28 | 59±9% (n=27) / 58±5% (n=81) | 18±1% | 0.19 | 37±3% (n=307) / 43±1% (n=1399) |
| disarm | 24±4% | 0.28 | 62±10% (n=26) / 57±5% (n=82) | 20±1% | 0.22 | 49±3% (n=341) / 40±1% (n=1365) |
| feel_no_pain | 24±4% | 0.29 | 65±9% (n=26) / 56±5% (n=82) | 3±0% | 0.03 | 58±7% (n=55) / 41±1% (n=1651) |
| headbutt | 24±4% | 0.27 | 62±10% (n=26) / 57±5% (n=82) | 32±1% | 0.41 | 40±2% (n=547) / 42±1% (n=1159) |
| metallicize | 24±4% | 0.29 | 65±9% (n=26) / 56±5% (n=82) | 34±1% | 0.43 | 49±2% (n=572) / 38±1% (n=1134) |
| limit_break | 23±4% | 0.28 | 76±9% (n=25) / 53±5% (n=83) | 4±0% | 0.05 | 49±6% (n=76) / 41±1% (n=1630) |
| reaper | 23±4% | 0.28 | 68±9% (n=25) / 55±5% (n=83) | 24±1% | 0.26 | 46±2% (n=407) / 40±1% (n=1299) |
| sword_boomerang | 23±4% | 0.27 | 68±9% (n=25) / 55±5% (n=83) | 17±1% | 0.20 | 45±3% (n=294) / 41±1% (n=1412) |
| carnage | 19±4% | 0.19 | 62±11% (n=21) / 57±5% (n=87) | 25±1% | 0.29 | 45±2% (n=426) / 41±1% (n=1280) |
| cleave | 19±4% | 0.22 | 48±11% (n=21) / 61±5% (n=87) | 33±1% | 0.43 | 37±2% (n=563) / 44±1% (n=1143) |
| iron_wave | 19±4% | 0.23 | 52±11% (n=21) / 60±5% (n=87) | 17±1% | 0.19 | 36±3% (n=287) / 43±1% (n=1419) |
| anger | 19±4% | 0.19 | 55±11% (n=20) / 59±5% (n=88) | 40±1% | 0.54 | 41±2% (n=686) / 42±2% (n=1020) |
| flex | 19±4% | 0.25 | 55±11% (n=20) / 59±5% (n=88) | 3±0% | 0.03 | 45±7% (n=51) / 42±1% (n=1655) |
| thunderclap | 19±4% | 0.19 | 60±11% (n=20) / 58±5% (n=88) | 42±1% | 0.54 | 38±2% (n=717) / 44±2% (n=989) |
| clothesline | 17±4% | 0.17 | 50±12% (n=18) / 60±5% (n=90) | 35±1% | 0.43 | 41±2% (n=599) / 42±1% (n=1107) |
| power_through | 17±4% | 0.21 | 56±12% (n=18) / 59±5% (n=90) | 13±1% | 0.14 | 43±3% (n=217) / 41±1% (n=1489) |
| bludgeon | 16±4% | 0.17 | 53±12% (n=17) / 59±5% (n=91) | 14±1% | 0.15 | 44±3% (n=246) / 41±1% (n=1460) |
| body_slam | 16±4% | 0.18 | 53±12% (n=17) / 59±5% (n=91) | 2±0% | 0.02 | 47±9% (n=34) / 42±1% (n=1672) |
| feed | 16±4% | 0.19 | 59±12% (n=17) / 58±5% (n=91) | 16±1% | 0.17 | 37±3% (n=275) / 42±1% (n=1431) |
| fiend_fire | 16±4% | 0.18 | 41±12% (n=17) / 62±5% (n=91) | 18±1% | 0.18 | 40±3% (n=300) / 42±1% (n=1406) |
| burning_pact | 15±3% | 0.15 | 69±12% (n=16) / 57±5% (n=92) | 2±0% | 0.02 | 40±9% (n=30) / 42±1% (n=1676) |
| entrench | 15±3% | 0.15 | 50±12% (n=16) / 60±5% (n=92) | 0±0% | 0.01 | 50±18% (n=8) / 42±1% (n=1698) |
| impervious | 15±3% | 0.15 | 50±12% (n=16) / 60±5% (n=92) | 19±1% | 0.21 | 44±3% (n=329) / 41±1% (n=1377) |

Largest presence gaps (any card):

| card | humans in deck | ours in deck | humans − ours |
|---|---|---|---|
| perfected_strike | 15±3% | 66±1% | -51 pts |
| pommel_strike | 30±4% | 63±1% | -34 pts |
| clash | 6±2% | 32±1% | -27 pts |
| true_grit | 30±4% | 6±1% | +24 pts |
| thunderclap | 19±4% | 42±1% | -24 pts |
| blood_for_blood | 5±2% | 28±1% | -23 pts |
| heavy_blade | 25±4% | 2±0% | +23 pts |
| anger | 19±4% | 40±1% | -22 pts |
| uppercut | 14±3% | 35±1% | -21 pts |
| feel_no_pain | 24±4% | 3±0% | +21 pts |
| limit_break | 23±4% | 4±0% | +19 pts |
| clothesline | 17±4% | 35±1% | -18 pts |
| whirlwind | 28±4% | 11±1% | +17 pts |
| flex | 19±4% | 3±0% | +16 pts |
| entrench | 15±3% | 0±0% | +14 pts |

## 3. Strength scaling (humans = base-exact)

| side | split | n with | win with | n without | win without |
|---|---|---|---|---|---|
| humans | Demon Form | 14 | 71±12% | 94 | 56±5% |
| ours | Demon Form | 371 | 80±2% | 1335 | 31±1% |

Scaling cards = copies of demon_form, inflame, spot_weakness, limit_break.

| side | bucket | n | win rate |
|---|---|---|---|
| humans | 0 scaling cards | 42 | 48±8% |
| humans | 1 scaling cards | 33 | 55±9% |
| humans | 2 scaling cards | 20 | 80±9% |
| humans | 3+ scaling cards | 13 | 69±13% |
| ours | 0 scaling cards | 609 | 22±2% |
| ours | 1 scaling cards | 639 | 45±2% |
| ours | 2 scaling cards | 318 | 61±3% |
| ours | 3+ scaling cards | 140 | 70±4% |

## 4. Relics (top 20 by human presence; humans = base-exact)

| relic | humans | ours |
|---|---|---|
| burning_blood | 89±3% | 43±1% |
| neows_lament | 38±5% | 2±0% |
| red_skull | 25±4% | 8±1% |
| vajra | 23±4% | 8±1% |
| preserved_insect | 20±4% | 6±1% |
| bronze_scales | 19±4% | 6±1% |
| orichalcum | 18±4% | 7±1% |
| bag_of_marbles | 17±4% | 6±1% |
| pen_nib | 17±4% | 8±1% |
| war_paint | 17±4% | 8±1% |
| letter_opener | 16±4% | 7±1% |
| red_mask | 16±4% | 11±1% |
| bag_of_preparation | 15±3% | 7±1% |
| kunai | 15±3% | 6±1% |
| lantern | 15±3% | 7±1% |
| potion_belt | 15±3% | 6±1% |
| sozu | 15±3% | 5±1% |
| matryoshka | 14±3% | 5±1% |
| mummified_hand | 14±3% | 5±1% |
| akabeko | 13±3% | 8±1% |

## 5. Reconstruction quality (all 145 human Champ fights)

Exact: 60/145 = 41±4%.

| issue reason | fights affected | issue entries |
|---|---|---|
| event_card_upgrade_unknown | 44 | 47 |
| deck_changing_relic_after_champ | 37 | 39 |
| bottled_relic | 28 | 30 |
| remove_card_not_in_deck | 6 | 7 |
| unmapped | 2 | 2 |

Unmapped names: {'relic:Dodecahedron': 2}.

`remove_card_not_in_deck` fights that also have `deck_changing_relic_after_champ`: 6/6 (the card was changed by a relic pickup we cannot undo).

Exact among Champ losses: 82±6% (n=45); among wins: 23±4% (n=100).
