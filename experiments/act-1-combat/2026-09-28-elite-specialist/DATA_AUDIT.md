# Elite data inventory

Counts refer to fights with an opening decision row. The baseline is
`combat_v3/2026-09-26/act1-all-bosses-a20-scaled-search`; the additional self-play source is `combat_v3/2026-09-27/ab-selfplay-gen1-rest`.

| Source / partition | Encounter | Fights | Run seeds | Distinct opening card multisets | Floors ≤10 / 11–14 / ≥15 | Starting HP median | Wins |
|---|---|---:|---:|---:|---:|---:|---:|
| train: buckets 6-9 | gremlin_nob | 1934 | 1887 | 1931 | 1251 / 683 / 0 | 55 | 1797 |
| train: buckets 6-9 | lagavulin | 2033 | 1990 | 2032 | 1234 / 799 / 0 | 57 | 1815 |
| train: buckets 6-9 | three_sentries | 2071 | 2036 | 2069 | 1296 / 775 / 0 | 55 | 1751 |
| train: bucket 4 | gremlin_nob | 514 | 501 | 514 | 342 / 172 / 0 | 56 | 473 |
| train: bucket 4 | lagavulin | 484 | 470 | 484 | 297 / 187 / 0 | 56 | 436 |
| train: bucket 4 | three_sentries | 500 | 489 | 500 | 313 / 187 / 0 | 56 | 414 |
| train: gen0 replay of bucket 4 | gremlin_nob | 330 | 325 | 330 | 219 / 111 / 0 | 58 | 311 |
| train: gen0 replay of bucket 4 | lagavulin | 330 | 323 | 330 | 198 / 132 / 0 | 56 | 285 |
| train: gen0 replay of bucket 4 | three_sentries | 330 | 327 | 330 | 203 / 127 / 0 | 56 | 259 |
| validation: bucket 5 | gremlin_nob | 508 | 497 | 508 | 303 / 205 / 0 | 56 | 472 |
| validation: bucket 5 | lagavulin | 527 | 516 | 527 | 315 / 212 / 0 | 55 | 461 |
| validation: bucket 5 | three_sentries | 512 | 503 | 512 | 332 / 180 / 0 | 57 | 433 |
| final: bucket 2-3 | gremlin_nob | 987 | 958 | 987 | 622 / 365 / 0 | 56 | 931 |
| final: bucket 2-3 | lagavulin | 1008 | 989 | 1006 | 644 / 364 / 0 | 56 | 919 |
| final: bucket 2-3 | three_sentries | 1074 | 1055 | 1074 | 667 / 407 / 0 | 56 | 906 |
| old dev: excluded | gremlin_nob | 1012 | 1001 | 1012 | 656 / 356 / 0 | 56 | 947 |
| old dev: excluded | lagavulin | 1076 | 1060 | 1076 | 692 / 384 / 0 | 56 | 971 |
| old dev: excluded | three_sentries | 974 | 954 | 974 | 590 / 384 / 0 | 54 | 818 |

## Diversity checks

- validation: bucket 5, gremlin_nob: 502/508 opening card multisets not present in training (this is **not** a count of new master decks).
- validation: bucket 5, lagavulin: 521/527 opening card multisets not present in training (this is **not** a count of new master decks).
- validation: bucket 5, three_sentries: 508/512 opening card multisets not present in training (this is **not** a count of new master decks).
- final: bucket 2-3, gremlin_nob: 979/987 opening card multisets not present in training (this is **not** a count of new master decks).
- final: bucket 2-3, lagavulin: 994/1008 opening card multisets not present in training (this is **not** a count of new master decks).
- final: bucket 2-3, three_sentries: 1057/1074 opening card multisets not present in training (this is **not** a count of new master decks).

- Self-play: 990 replays from 990 distinct source fight IDs; 0 sources missing from bucket-4 baseline; 0 source IDs in held-out partitions.
- Opening card multiset is an approximate deck fingerprint: it drops order and zone and retains card ID/upgraded; it can include fight-generated cards. Exact deck diversity requires inspecting stored fight-start snapshots.

