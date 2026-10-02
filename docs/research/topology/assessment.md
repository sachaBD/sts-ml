# Value-net topology assessment (act 1, combat_v3)

For the **selected Act 2 overworld** network and its interactive visual report, see
[overworld.md](overworld.md). This assessment concerns the separate combat network.

Written 2026-09-26 while reviewing `python/sts_combat_rl/models/deep_sets.py` (v1),
`models/value_net.cpp`, the trainer, the `act1-gen0` checkpoint metrics, and the new dataset
`combat_v3/2026-09-26/act1-all-bosses-a20-scaled-search`. These are informed opinions, not results.
Every recommendation is still a hypothesis until an experiment confirms it. See `plan.md` for the work.

## TL;DR

- The **topology family is right**. Keep it: typed set tokens, shared per-token encoders, sum pooling, and
  explicit card×monster pair tokens.
- The **post-pool head is the bottleneck**: one 64-wide layer does all cross-set reasoning. Make it
  deeper, wider and normalised first.
- **Parameters**: 63k is small. Scale to roughly 300k–1M, and put the parameters where C++ inference
  caches them.
- **Attention**: not first. Later, try a small version over hand + monsters + a global token (~8–15
  tokens), with piles still pooled. Adopt it only if it wins at equal wall-clock search time.
- **The labels are inconsistent**, and this probably matters as much as the topology (see below).

## v1 as built (62,960 params)

| Part | Params | Notes |
|---|---:|---|
| Embeddings: card_id(512×8), move(512×8), monster(128×8), small ones | ~9.6k | Mostly unused vocab: 125 cards, 25 monsters, 66 moves seen |
| card_mlp 32→64→64 | 6.3k | Cached per card token in C++ |
| monster_mlp 25→64→64 | 5.8k | Cached per monster token |
| interaction_mlp (2·64+6)→64→64 | 12.8k | Cached per (card, monster, numeric) |
| **head** (50+8+6·64=442)→64→1, tanh | **28.4k** | Runs once per evaluation; the only place hand, piles, monsters and player meet |

Pools: 4 card-zone sums (hand, draw, discard, exhaust), a monster sum and an interaction sum. There is
no normalisation anywhere.

## Weak spots

1. **The head is too thin.** Deep Sets (sum pooling) is only expressive if the network after the sum is
   strong. Questions like "is the damage in hand enough to kill this through its block with 3 energy"
   all have to be answered by one 64-unit ReLU layer.
2. **Pooled sums aren't normalised.** Draw-pile size ranges from about 0 to 70, so the size of the summed
   vector varies about 10×. The head sees inputs whose scale swings a lot. Fix: add log1p(count)
   features and a LayerNorm at the head input.
3. **Card vectors ignore the state.** A card is encoded without energy, strength or enemy intent. That
   makes it cacheable and is fine for piles. The hand is where context would help (attention, later).
4. **Small details.** Targets lie in [0, 0.91] (0 = loss), so sigmoid fits better than tanh. 8-dim
   card-id embeddings are tight; 16–32 cost nothing because they're cached.

## Parameters: how many, and where

- `act1-gen0` (1.8M rows, 63k params, 12 epochs): validation MSE flattened around epoch 5 (~0.0045)
  while training MSE kept falling (0.0040 → 0.0027). That is **not** a pure capacity limit. The
  likely cause is limited *effective* data: rows within a fight are strongly correlated, so there are
  ~80k independent fights, not 4.5M independent rows. Inconsistent labels (below) are another candidate.
  Larger models plus regularisation still usually win here, but this has to be measured.
- **Inference cost is the real limit**: every microsecond in the net is taken from search simulations.
  - Cached parts (card, monster and interaction encoders, embeddings) are nearly free at search time,
    so make them wide.
  - The head runs once per evaluation. 3×256 residual layers are about 10× the current head's compute.
    Measure it in `apps/bench_search`.

## Attention (deferred)

- **Would help with**: interactions within the hand (Bash before attacks, energy limits on which cards
  can be played, Limit Break with strength).
- **Would not help much with**: the piles. Their order is hidden, so a sum is nearly the right summary.
- **Cost**: a full 2-layer d=64 transformer over ~20 tokens is roughly 30× the current compute, and it
  breaks the per-card cache.
- **Cheap variant to try later**: 1–2 layers, width 64–128, 4 heads, over hand cards + monsters +
  1 global token + 1 pooled token per pile. The pile tokens stay cached sums.
- **Judge it** on gameplay at a fixed time budget, not on validation MSE.

## The label issue (not yet scheduled)

`root_value` is the teacher's mean backed-up reward over all its search visits. It is consistently
**lower than the realised outcome** `terminal_value`:

| Category | Decision rows: mean(z − v) | Child rows |
|---|---:|---:|
| boss | +0.13 | +0.17 |
| elite | +0.12 | +0.16 |
| event | +0.10 | +0.15 |
| hard | +0.07 | +0.09 |
| easy | +0.06 | +0.07 |

The gap is largest early in a fight: boss turns 0–2 are +0.18, and it shrinks to about +0.08 later.

- `label = blend` gives the chosen move's rows 0.5·z + 0.5·v.
- Child rows (73% of the new data: the teacher's non-chosen actions with ≥50 visits) always get v.
- So the chosen action gets a built-in bonus of about 0.5 × 0.13 ≈ 0.065 on bosses. That's comparable
  to model RMSE (~0.07 on boss/elite in `act1-gen0`), and it distorts exactly the sibling comparisons
  search relies on.
- Options: use one target type for all rows (e.g. `label = root`), calibrate v, or train separate
  outputs per target type.
- Also: the trainer's "teacher MSE" is **not** a noise floor, because the label is partly v itself.

## New dataset facts (act1-all-bosses-a20-scaled-search)

- **Size**: 4,507,202 rows, 80,769 fights, 12,106 runs (split by `run_seed`). The `run.json` status is
  "failed" only because it ran with `forever = true` and was interrupted; the parts are complete fights.
- **Row kinds**: 1.17M decision, 46k random-move decision, 3.29M child. Every child row has
  `simulations_used = 0`.
- **Tokens per row**: on average 16.5 cards (max 78), 1.44 monsters, ~3.3 interactions (max 40).
- **Categories**: easy fights are 27% of rows and ~99.99% wins. Boss win rates: guardian 72%,
  slime 75%, hexaghost 55%.
- **Targets**: z = 0 on a loss and in [0.25, 0.91] on a win. corr(v, z) = 0.88. std ≈ 0.24.

## Other modern improvements that fit here

- **Auxiliary heads** (KataGo-style): add a `won` output (binary cross-entropy) and a `final_hp`
  output. They're cheap and usually improve data efficiency. C++ skips them at inference. Trajectory
  outcomes are only valid for on-trajectory rows after the last random move, so mask the rest.
- **GPU training** on the RTX 3060: batch 1024–4096, AdamW, warmup + cosine, a few epochs, and a
  weight average (EMA) for the evaluated or exported model.
- **Per-category reporting** and optional category weights, because easy fights dominate the rows.
