# Apps

Runnable workflows live here. Each launchable app keeps its example TOML files in
`apps/<app>/config/` and is run from the repository root:

```sh
./apps/<app>/run.sh apps/<app>/config/<preset>.toml [--scratch]
```

`run.sh` records a managed run under `runs/`; see [`../runs/README.md`](../runs/README.md).
Apps without `run.sh` are direct tools or research drivers; their module docstring is their
entry-point reference.

| App | Purpose | Output / entry point |
|---|---|---|
| `bootstrap` | Search-teacher Act 1 self-play | `combat_v3` |
| `combat_transition` | Replay combat rows into pre/post persistent states | `combat_transition_v1` |
| `value_train` | Train a combat value model | `value_net_v1` |
| `value_play` | Play combat with a value model | `combat_v3` |
| `dagger` | Collect learner-played, teacher-labelled combat data | `combat_v3` |
| `fight_resample` | Re-evaluate stored fights | `combat_v3` |
| `compare_fights` | Compare two combat policies on matched fights | `fight_comparison_v1` |
| `card_marginals` | Generate paired card-addition combat outcomes | `card_marginals_v1` |
| `gauntlet` | Evaluate card rewards on downstream elite/boss fights | `gauntlet_v1` |
| `card_search` | Prototype model-based card-pick search | `search.py` |
| `run_rl` | Research loop for real-run card-choice RL | `loop.py` |
| `topology_report` | Offline selected-overworld architecture and interaction visualizations | `python -m apps.topology_report.generate` |
| `export_save` | Export a stored fight as a Slay the Spire save | `export_save.py` |
| `common` | Shared launcher, configuration, worker, and replay support | library only |

Experiment-specific configurations and analysis remain with their experiment in `experiments/`.
