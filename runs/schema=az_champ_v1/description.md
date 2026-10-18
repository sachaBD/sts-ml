# az_champ_v1

Producer: `./apps/az_train/run.sh` (`apps/az_train/train.py`). Contract: [schema.py](schema.py).

One AlphaZero-style training run on the JAX simulator (sts-lightspeed `jaxsim`, Ironclad vs The Champ, A20):
self-play with Gumbel MuZero search over a start bank of random decks of jaxsim-supported cards, win-only value
targets. No Parquet tables: `metrics.jsonl` (one JSON object per iteration), `checkpoints/params-<it>.msgpack`
and `params.msgpack` (Flax-serialized parameters of `agents.combat.az.model.AZNet`, configured by `config.json`
[model]), `summary.json` (final evaluation, merged into run.json).

Evaluation win rates (`eval_win_hp<H>`) are on a fixed, seeded bank of evaluation decks at starting HP H, with
deterministic search (gumbel_scale 0). They are not comparable across different [eval] settings.
