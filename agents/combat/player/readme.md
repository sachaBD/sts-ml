# Combat player

The agent that plays fights: `nn_play` (`nn_play.cpp`), launched by `play.py`. `nn_play --help` lists its
options. Its settings are each stage's `play.toml`, written by `play.py`; `play_config.hpp` owns them. It knows two
interfaces:

- `search::SearchFactory` (`search/search.hpp`): a fresh search per decision. `search::SearchConfig`
  (`search/config.hpp`) decides which search runs; the player never names it.
- `nn::NetworkEvaluator` (`nn/evaluator.hpp`): a search query becomes network tokens; a prediction becomes an
  evaluation.

`Decision` joins one search to the network. `Episodes` (`episodes.hpp`) hands fights to infra as `infra::Episode`s, so infra
batches the network calls of every fight in flight ([infra's contract](../../../infra/workers/readme.md)).

For deterministic experimental play, `[search].algorithm = "unified"` uses the Horizon teacher's
24-eval thinking output to price the opening comparison; `lambda` prices subsequent paired waves
as well. Use a checkpoint trained on the Horizon `[8, 24, 64, 160]` targets (a four-output Gumbel
checkpoint is **not** interchangeable), `gumbel_scale = 0`, and `rollout_mix = 0`. The `sims`
setting remains a safety cap on simulations, not a count of NN requests. Compare costs with the
fight result's `network_requests`, which excludes cache hits. This mode does not train a new head.

## Why a separate process

Each play stage is a fresh `nn_play` process: it gets a fresh CUDA context, never shares the GPU with the
PyTorch learner, and a crash is recovered by starting a new one. A poisoned CUDA context exits without
teardown, for the app's supervisor to handle.

## Failures and output

Failures abort collection; they never become losses. Archive an interrupted stage; never append partial
replay. Each fight's samples and result line are written under one lock, and the result records the samples'
byte range, so replay finds fights without inferring boundaries from order. The two files are not
crash-atomic: trailing sample bytes alone do not prove a finished fight.

`algorithm = "filter"` is a separate deterministic allocator using the same four Horizon targets.
It prices shared-world comparisons against the prior, plunging greedily to the next turn (or a
terminal). It can also test the policy's second action at the first interior public decision.
A scalar normal update estimates each paired difference; expected improvement in the root choice
prices a further observation. `sims` caps **new non-root NN requests** in this mode, and `particles`
bounds distinct worlds per comparison. `filter-root` disables interior comparisons; `filter-fixed`
keeps updating mean scores but freezes uncertainty for purchase pricing. These are ablations,
not additional deployment dials. This version does not retrain the heads or implement stochastic play.
