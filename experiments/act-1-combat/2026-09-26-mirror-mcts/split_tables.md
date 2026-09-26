## Fights per encounter and part

| category | encounter | train | dev | dev_eval | confirm | train rows % |
|---|---|---|---|---|---|---|
| boss | slime_boss | 3739 | 999 | 999 | 972 | 23.9 |
| easy | cultist | 3280 | 856 | 250 | 861 | 5.9 |
| easy | jaw_worm | 3263 | 904 | 250 | 839 | 6.6 |
| easy | small_slimes | 3325 | 865 | 250 | 847 | 8.3 |
| easy | two_louse | 3209 | 866 | 250 | 861 | 6.8 |
| elite | gremlin_nob | 1889 | 508 | 250 | 504 | 4.7 |
| elite | lagavulin | 1877 | 526 | 250 | 468 | 8.2 |
| elite | three_sentries | 1892 | 519 | 250 | 495 | 11.4 |
| event | gremlin_nob | 116 | 33 | 33 | 34 | 0.3 |
| event | lagavulin_event | 124 | 31 | 31 | 30 | 0.5 |
| event | mushrooms_event | 372 | 89 | 89 | 93 | 1.4 |
| event | three_sentries | 128 | 25 | 25 | 29 | 0.7 |
| hard | blue_slaver | 870 | 225 | 225 | 243 | 2.1 |
| hard | exordium_thugs | 721 | 170 | 170 | 196 | 2.5 |
| hard | exordium_wildlife | 722 | 181 | 181 | 169 | 2.2 |
| hard | gremlin_gang | 515 | 136 | 136 | 130 | 2.2 |
| hard | large_slime | 737 | 205 | 205 | 179 | 2.6 |
| hard | looter | 871 | 247 | 247 | 241 | 1.8 |
| hard | lots_of_slimes | 407 | 124 | 124 | 99 | 2.1 |
| hard | red_slaver | 503 | 147 | 147 | 128 | 1.1 |
| hard | three_louse | 746 | 215 | 215 | 221 | 2.3 |
| hard | two_fungi_beasts | 900 | 233 | 233 | 207 | 2.3 |

## Totals

| part | runs | fights | decision_rows | all_rows |
|---|---|---|---|---|
| confirm | 1179 | 7846 | 110983 | 473315 |
| dev | 1198 | 8104 | 114928 | 492769 |
| train | 4518 | 30206 | 427993 | 1823011 |

## Teacher (stored bootstrap, 15k sims, one random move) on train fights

| category | encounter | fights | win % | HP lost (wins) | sd |
|---|---|---|---|---|---|
| boss | slime_boss | 3739 | 70.2 | 22.8 | 17.2 |
| easy | cultist | 3280 | 100.0 | 4.0 | 5.3 |
| easy | jaw_worm | 3263 | 99.9 | 8.9 | 6.2 |
| easy | small_slimes | 3325 | 100.0 | 6.5 | 6.2 |
| easy | two_louse | 3209 | 99.9 | 4.5 | 4.8 |
| elite | gremlin_nob | 1889 | 93.5 | 24.8 | 13.3 |
| elite | lagavulin | 1877 | 89.3 | 27.9 | 14.7 |
| elite | three_sentries | 1892 | 84.5 | 28.7 | 15.5 |
| event | gremlin_nob | 116 | 91.4 | 19.9 | 10.3 |
| event | lagavulin_event | 124 | 66.9 | 32.1 | 14.6 |
| event | mushrooms_event | 372 | 97.0 | 13.7 | 10.6 |
| event | three_sentries | 128 | 86.7 | 22.2 | 14.9 |
| hard | blue_slaver | 870 | 99.7 | 6.9 | 6.5 |
| hard | exordium_thugs | 721 | 98.1 | 13.3 | 10.6 |
| hard | exordium_wildlife | 722 | 99.0 | 8.4 | 9.9 |
| hard | gremlin_gang | 515 | 96.9 | 14.4 | 12.6 |
| hard | large_slime | 737 | 99.6 | 7.5 | 9.6 |
| hard | looter | 871 | 99.8 | 5.1 | 5.7 |
| hard | lots_of_slimes | 407 | 99.0 | 9.8 | 9.4 |
| hard | red_slaver | 503 | 100.0 | 7.6 | 6.6 |
| hard | three_louse | 746 | 99.6 | 6.4 | 7.6 |
| hard | two_fungi_beasts | 900 | 100.0 | 3.4 | 5.5 |

## Dev eval cost proxy (stored teacher simulations)

| boss | fights | decisions | M sims |
|---|---|---|---|
| False | 3811 | 48091 | 526.0 |
| True | 999 | 26804 | 287.0 |
