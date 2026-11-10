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

## Why a separate process

Each play stage is a fresh `nn_play` process: it gets a fresh CUDA context, never shares the GPU with the
PyTorch learner, and a crash is recovered by starting a new one. A poisoned CUDA context exits without
teardown, for the app's supervisor to handle.

## Failures and output

Failures abort collection; they never become losses. Archive an interrupted stage; never append partial
replay. Each fight's samples and result line are written under one lock, and the result records the samples'
byte range, so replay finds fights without inferring boundaries from order. The two files are not
crash-atomic: trailing sample bytes alone do not prove a finished fight.
