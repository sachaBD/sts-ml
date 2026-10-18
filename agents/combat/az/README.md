# az: AlphaZero-style combat agent on the JAX simulator

Ironclad vs The Champ (A20) on sts-lightspeed's `jaxsim`. Search, network and simulator run in one compiled JAX
program. The simulator is used only through `jaxsim.Env` (contract: `jaxsim/src/jaxsim/api.py` in sts-lightspeed).

| File | What |
|---|---|
| `model.py` | `AZNet`: integer observations -> per-slot policy logits + win-probability logit |
| `search.py` | mctx Gumbel MuZero with the real simulator as the model; root particles = different keys |
| `selfplay.py` | B games x T decisions per jitted call, reset from a start bank; evaluation play |
| `replay.py` | host replay buffer; win-only value targets attached when a game ends |
| `learn.py` | policy cross-entropy + value BCE, AdamW |
| `decks.py` | random start banks over jaxsim-supported cards (until human decks are supported) |

**Status:** the plumbing works end to end (`apps/az_train`). It is not tuned for learning: training waits
until jaxsim supports enough cards for real decks. The tracked metric for now is throughput (`apps/az_bench`).

## Setup

`make jax-env STS_LIGHTSPEED_DIR=../sts-lightspeed` (the default is `../sts_lightspeed`). On a uv-created
venv: `STS_LIGHTSPEED_DIR=../sts-lightspeed uv pip install --python .venv/bin/python -r requirements-jax.txt`.

```sh
PYTHONPATH=. .venv/bin/python -m agents.combat.az.test_az
./apps/az_train/run.sh apps/az_train/config/smoke.toml --scratch
PYTHONPATH=. .venv/bin/python -m apps.az_bench.bench --games 64 256 --json bench.jsonl
```

## Throughput

`apps/az_bench`: reference deck bank, untrained 179k-parameter network, 8 particles x 64 simulations.

| Date | Device | Cards | Sim decisions/s | Search simulations/s | Self-play decisions/s | Self-play fights/s |
|---|---|---|---|---|---|---|
| 2026-10-18 | MacBook Air CPU | 12 | 410k | 67k (64 games) | 65 (64 games) | 3.0 |

Fight length depends on the policy (an untrained network loses fast), so compare decisions/s.
