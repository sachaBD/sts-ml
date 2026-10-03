# Combat search

Written 2026-10-03 while planning boss-combat work (The Champ). Explanatory notes, not results.

1. [Tree search, the basics](1-tree-search-basics.md): the idea, step by step, no game details.
2. [What we use today](2-today.md): reference only.
3. [AlphaZero-style search](3-alphazero.md): a network that steers (π) and scores (V), so the search can plan
   several turns ahead.

Current implementation: [PV README](../../../agents/combat/pv/README.md). It specifies the dedicated PV tree,
explicit root evaluation, template scoring/selection policies and HP-equivalent value contract. The primers use
illustrative scores and visit counts; they are not depth/performance measurements.

Older: [combat search in plain English](../combat_search_in_plain_english.md).
