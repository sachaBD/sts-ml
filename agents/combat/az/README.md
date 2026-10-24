# az: AlphaZero-style combat agent on the JAX simulator

Ironclad vs The Champ (A20) on sts-lightspeed's `jaxsim`. Search, network and simulator run in one compiled JAX
program. The simulator is used only through `jaxsim.Env` (contract: `jaxsim/src/jaxsim/api.py` in sts-lightspeed).

| File | What |
|---|---|
| `model.py` | `AZNet`: transformer over card-slot, potion, player and enemy tokens -> per-action logits + P(win) logit |
| `search.py` | mctx Gumbel MuZero with the real simulator as the model; root particles = different keys |
| `selfplay.py` | B games x T decisions per jitted call, written straight into the device replay buffer; lane recycling; evaluation |
| `replay.py` | device replay ring + game-outcome table (value targets: win/loss; truncated and errors give none) |
| `learn.py` | policy cross-entropy + masked value BCE, AdamW, K steps per jitted call |
| `decks.py` | deck files (JSON, see the module docstring) with a rejection report; seeded random banks for tests |

**Status:** runs end to end (`apps/az_train`, smoke config on the Mac and on an RTX 3060). Not yet trained on the GPU.
RTX 3060 throughput, the search-depth bottleneck and next steps: `docs/research/az_gpu_throughput_handoff.md`.
On macOS the package sets XLA optimisation level 0 (see `__init__.py`): the default takes minutes to compile
jaxsim's batched step on the CPU.

## Setup

`make jax-env STS_LIGHTSPEED_DIR=../sts-lightspeed` (the default is `../sts_lightspeed`). On a uv-created
venv: `STS_LIGHTSPEED_DIR=../sts-lightspeed uv pip install --python .venv/bin/python -r requirements-jax.txt`.

```sh
PYTHONPATH=. .venv/bin/python -m agents.combat.az.test_az
./apps/az_train/run.sh apps/az_train/config/smoke.toml --scratch      # ~1 min
./apps/az_train/run.sh apps/az_train/config/gpu3060.toml             # the GPU box; decks in [decks]
PYTHONPATH=. .venv/bin/python -m apps.az_bench.bench           # quick, ~40 s on the Mac (compile cached)
PYTHONPATH=. .venv/bin/python -m apps.az_bench.bench --full    # more batch sizes, longer timing
```

## Throughput

`apps/az_bench`: reference deck bank, untrained 179k-parameter network, 8 particles x 64 simulations.
CPU numbers are a rough guide only; the target is a GPU. Rows up to batch 2 used the old full run.

| Date | Device | Cards | Sim decisions/s | Search simulations/s | Self-play decisions/s | Self-play fights/s |
|---|---|---|---|---|---|---|
| 2026-10-18 | MacBook Air CPU | 12 | 410k | 67k (64 games) | 65 (64 games) | 3.0 |
| 2026-10-18 | MacBook Air CPU | 12 (sts-lightspeed b648f15, rerun) | 406k / 429k | 69k | 65.9 | 2.98 |
| 2026-10-18 | MacBook Air CPU | 44 (batch 1, sts-lightspeed 3e5dfa5) | 372k / 388k | 69k | 60.9 | 2.64 |
| 2026-10-18 | MacBook Air CPU | 44 (batch 1, 3e5dfa5, rerun) | 388k / 409k | 69k | 63.9 | 2.77 |
| 2026-10-18 | MacBook Air CPU | 54 (batch 2, 0d330cc) | 327k / 326k | 63k | 59.4 | 2.46 |

Sim decisions/s is given at 1024 / 4096 games. Fight length depends on the policy (an untrained network loses
fast), so compare decisions/s. The reference bank is drawn from the supported cards, so it changes as cards are
added. Rows marked "rerun" were measured back-to-back with the row below them, under the same machine load.

RTX 3060, 128 simulations x 1 particle, mid-fight, after the search fix (23e8f04): self-play 714 / 281 / 256
decisions/s at 2048 games with a trivial / small (59k) / reference (460k) network (`apps/az_bench/probe_selfplay`).
