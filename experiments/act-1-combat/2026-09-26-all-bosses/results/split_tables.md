## Fights per encounter and part

| category | encounter | train | dev | dev_eval | confirm | train rows % |
|---|---|---|---|---|---|---|
| boss | hexaghost | 1890 | 642 | 200 | 665 | 7.6 |
| boss | slime_boss | 2011 | 697 | 200 | 683 | 8.0 |
| boss | the_guardian | 2017 | 649 | 200 | 668 | 9.1 |
| easy | cultist | 5215 | 1709 | 15 | 1728 | 5.4 |
| easy | jaw_worm | 5308 | 1764 | 17 | 1776 | 6.7 |
| easy | small_slimes | 5233 | 1770 | 17 | 1775 | 8.0 |
| easy | two_louse | 5255 | 1762 | 16 | 1727 | 6.6 |
| elite | gremlin_nob | 2956 | 1012 | 100 | 987 | 4.6 |
| elite | lagavulin | 3044 | 1076 | 100 | 1008 | 8.2 |
| elite | three_sentries | 3083 | 974 | 100 | 1074 | 11.5 |
| event | gremlin_nob | 201 | 59 | 1 | 59 | 0.3 |
| event | lagavulin_event | 207 | 69 | 0 | 70 | 0.5 |
| event | mushrooms_event | 591 | 189 | 0 | 183 | 1.4 |
| event | three_sentries | 198 | 59 | 0 | 58 | 0.7 |
| hard | blue_slaver | 1444 | 487 | 6 | 458 | 2.2 |
| hard | exordium_thugs | 1141 | 379 | 4 | 356 | 2.4 |
| hard | exordium_wildlife | 1139 | 367 | 5 | 390 | 2.2 |
| hard | gremlin_gang | 723 | 272 | 3 | 282 | 2.0 |
| hard | large_slime | 1256 | 397 | 2 | 423 | 2.9 |
| hard | looter | 1370 | 442 | 4 | 482 | 1.9 |
| hard | lots_of_slimes | 651 | 253 | 1 | 226 | 2.1 |
| hard | red_slaver | 750 | 276 | 1 | 271 | 1.1 |
| hard | three_louse | 1189 | 406 | 2 | 414 | 2.3 |
| hard | two_fungi_beasts | 1434 | 488 | 6 | 502 | 2.4 |

## Totals

| part | runs | fights | decision_rows | all_rows |
|---|---|---|---|---|
| confirm | 2421 | 16265 | 244621 | 910901 |
| dev | 2421 | 16198 | 242186 | 895136 |
| train | 7264 | 48306 | 728186 | 2701165 |

## Stored teacher (scaled search, one random move) by part: win % / mean terminal value

| category | encounter | train win % | dev win % | confirm win % | train HP lost (wins) |
|---|---|---|---|---|---|
| boss | hexaghost | 53.7 | 55.0 | 57.0 | 37.8 |
| boss | slime_boss | 70.8 | 72.6 | 69.7 | 23.0 |
| boss | the_guardian | 65.0 | 66.3 | 66.5 | 35.6 |
| easy | cultist | 100.0 | 100.0 | 99.9 | 4.5 |
| easy | jaw_worm | 100.0 | 100.0 | 99.9 | 9.0 |
| easy | small_slimes | 100.0 | 99.9 | 100.0 | 7.0 |
| easy | two_louse | 100.0 | 100.0 | 100.0 | 4.4 |
| elite | gremlin_nob | 92.8 | 93.6 | 94.3 | 24.3 |
| elite | lagavulin | 89.1 | 90.2 | 91.2 | 28.3 |
| elite | three_sentries | 84.3 | 84.0 | 84.4 | 28.7 |
| event | gremlin_nob | 90.5 | 89.8 | 98.3 | 22.9 |
| event | lagavulin_event | 64.7 | 58.0 | 57.1 | 31.6 |
| event | mushrooms_event | 96.6 | 95.8 | 96.7 | 15.4 |
| event | three_sentries | 85.4 | 89.8 | 87.9 | 21.2 |
| hard | blue_slaver | 99.5 | 99.4 | 99.1 | 6.8 |
| hard | exordium_thugs | 97.7 | 98.9 | 98.0 | 12.8 |
| hard | exordium_wildlife | 99.1 | 99.5 | 99.0 | 8.8 |
| hard | gremlin_gang | 96.4 | 96.7 | 97.2 | 16.3 |
| hard | large_slime | 98.5 | 98.2 | 98.8 | 8.4 |
| hard | looter | 99.7 | 99.3 | 100.0 | 5.4 |
| hard | lots_of_slimes | 98.5 | 99.2 | 98.7 | 10.6 |
| hard | red_slaver | 98.8 | 98.2 | 100.0 | 7.6 |
| hard | three_louse | 99.1 | 99.8 | 99.5 | 6.7 |
| hard | two_fungi_beasts | 99.8 | 99.4 | 99.6 | 4.0 |

## Dev eval subset size

| boss | fights | decisions |
|---|---|---|
| False | 400 | 6643 |
| True | 600 | 18854 |
