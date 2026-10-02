# Real-run overworld learning

Run commands from the repository root. **v3.1 and v3.2 use the same train/play interface**;
playback detects the model kind from its checkpoint. No manual encoder selection is needed.
The new default architectures are untrained; the synthetic smoke-test weights are not agents.

## 1. Build and collect fresh observations

```sh
make build
.venv/bin/python apps/run_rl/play.py \
  --out runs/overworld-v3-collect --first-seed 980000000000 --seeds 100 --workers 2 \
  --policy simple --max-act 4 --target heart
```

`play.py` always uses Ironclad A20. `--max-act 4` means all three acts plus the Heart,
with actual key requirements. The rebuilt worker records graph connectivity and versioned
public observations even when SimpleAgent makes decisions. Old path-only traces cannot train v3.
Choose a fresh output directory when changing policy or goal: playback resumes completed seeds
in an existing directory, without validating that its previous configuration matches.

## 2. Train either topology

```sh
.venv/bin/python apps/run_rl/train.py \
  --data runs/overworld-v3-collect --out runs/overworld-v3.1/model.pt \
  --arch-spec agents/overworld/value/architectures/rp3.1-w64-h128-l2.toml \
  --target heart
```

For attention, replace `rp3.1` with `rp3.2` and use a separate output directory.
Named specs are hashed and frozen after their first successful training run. To continue a
checkpoint, use `--init PATH` instead of `--arch-spec`.

**Sparse-target warning:** `heart` rewards only actual Heart defeat. If collection contains no
Heart wins, start with `--target floors3` for progress pretraining and improve collection, then
fine-tune with `--init ... --target heart` when positive Heart examples exist. This is a suggested
bootstrap, not an experimentally established recipe. Graph defaults are training batch 32 /
validation batch 64; `--batch-size` and `--val-batch-size` override them.

## 3. Play a trained checkpoint

```sh
.venv/bin/python apps/run_rl/play.py \
  --out runs/overworld-v3.1-eval --first-seed 990000000000 --seeds 100 --workers 2 \
  --policy net --ckpt runs/overworld-v3.1/model.pt \
  --max-act 4 --target heart --decide rest path shop event neow boss_relic
```

`--decide` matters: without it, only card choices use the network; other decisions remain under
SimpleAgent. Combat defaults to guided-rollout MCTS. Use disjoint collection/evaluation seeds.
`--worker build/overworld-v3/run_rl_worker` selects the isolated validation build if desired;
the default is `build/main/run_rl_worker`.

Heart mode takes sapphire before its competing chest relic. Learned rest choices include recall;
SimpleAgent-controlled campfires fall back to recall at the last Act 3 campfire. Emerald still
requires reaching a burning elite. Missing keys produce `heart_locked`, not a Heart clear.
These are minimal key fallbacks, not a fully learned key-acquisition policy.

## Scope and references

The v3-ready entrypoints are **`play.py` and `train.py`**. Historical `loop.py`, `collect.py` and
`netstudy.py` retain their older experiment-specific assumptions; they are not advertised as a
one-command Heart expert-iteration workflow. v1/v2 checkpoint loading remains supported.

- [Concise map encoder explainer](../../docs/research/topology/overworld-map.md)
- [Variants, observation/privacy contract, verification and cost](../../docs/research/topology/overworld-v3.md)
