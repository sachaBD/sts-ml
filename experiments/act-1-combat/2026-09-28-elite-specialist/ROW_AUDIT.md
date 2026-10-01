# Terminal-label row inventory

For baseline fights with a forced random move, only decisions **after** the last random move have usable realised terminal labels. Self-play has no forced random move. Counts include all decision rows (not child rows).

| part | encounter | fights | decisions | terminal_eligible | no_terminal_rows | fights_with_random | fights_multi_random |
| --- | --- | --- | --- | --- | --- | --- | --- |
| excluded 0-1 | gremlin_nob | 1012 | 12909 | 8840 | 23 | 539 | 0 |
| excluded 0-1 | lagavulin | 1076 | 22169 | 12881 | 12 | 876 | 0 |
| excluded 0-1 | three_sentries | 974 | 22941 | 13363 | 16 | 830 | 0 |
| final 2-3 | gremlin_nob | 987 | 12355 | 8601 | 15 | 524 | 0 |
| final 2-3 | lagavulin | 1008 | 20564 | 12242 | 14 | 816 | 0 |
| final 2-3 | three_sentries | 1074 | 25578 | 14971 | 20 | 903 | 0 |
| train baseline 4,6-9 | gremlin_nob | 2448 | 31228 | 21786 | 62 | 1283 | 0 |
| train baseline 4,6-9 | lagavulin | 2517 | 52515 | 30813 | 41 | 2021 | 0 |
| train baseline 4,6-9 | three_sentries | 2571 | 62401 | 36835 | 43 | 2253 | 0 |
| train replay bucket 4 | gremlin_nob | 330 | 4066 | 4066 | 0 | 0 | 0 |
| train replay bucket 4 | lagavulin | 330 | 6826 | 6826 | 0 | 0 | 0 |
| train replay bucket 4 | three_sentries | 330 | 8040 | 8040 | 0 | 0 | 0 |
| validation 5 | gremlin_nob | 508 | 6308 | 4496 | 13 | 258 | 0 |
| validation 5 | lagavulin | 527 | 10821 | 6344 | 11 | 421 | 0 |
| validation 5 | three_sentries | 512 | 12416 | 7309 | 8 | 441 | 0 |
