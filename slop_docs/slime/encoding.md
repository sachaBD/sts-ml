# Combat NN encoding v3

Source of truth: `combat/encoding.hpp` and `CombatEnvironment::decision()` in `combat/environment.cpp`. Any change to feature order or normalisation must increment `combat_encoding_schema_version`.

Scope: Ironclad cards covered by the encoder's card table (`card_meta`). Unsupported cards throw rather than encode silently. Relics and potions are not encoded; the entry projection strips them.

The encoding is public information only: no draw order, no RNG, no card instance IDs. Cards, monsters and card-monster pairs are variable-length sets, pooled by summation.

## Global features

Categorical: `input_state`, `card_selection_task`.

Numeric (50):

| Index | Feature | Scale |
|---:|---|---|
| 0 | turn | /20 |
| 1–2 | player HP, HP fraction | /100, hp/max |
| 3–5 | block, energy, energy per turn | /100, /10, /10 |
| 6–10 | strength, dexterity, weak, vulnerable, frail | /10 |
| 11–15 | cards played this turn, hand, draw, discard, exhaust sizes | /20, /10, /64, /64, /64 |
| 16–17 | incoming attack damage, HP loss after block | /100 |
| 18–21 | Combust, Combust HP loss, Flame Barrier, NO_DRAW | /10, /10, /50, 0/1 |
| 22–23 | intangible, artifact | /10 |
| 24–25 | Barricade, Corruption | 0/1 |
| 26–34 | Brutality, Demon Form, Dark Embrace, Evolve, Feel No Pain, Metallicize, Rage, Double Tap, Vigor | /10 |
| 35–37 | The Bomb timers 1–3 | /50 |
| 38–47 | no block, lose strength, lose dexterity, energized, Fire Breathing, Juggernaut, Rupture, Magnetism, Mayhem, Panache | /10 |
| 48 | Panache counter | /20 |
| 49 | Sadistic | /10 |

## Card tokens

One token per card in hand, draw, discard or exhaust.

- **Categorical:** card ID, zone, card type, target type.
- **Numeric (14):** upgraded, base cost, current cost, damage, hits, base block, effective block, draws, special counter, free to play, exhausts, ethereal, retained, playable now.

Damage and block come from `card_meta`, which accounts for state-dependent cards (e.g. Body Slam, Heavy Blade, Perfected Strike).

## Monster tokens

One token per living monster.

- **Categorical:** monster ID, current move.
- **Numeric (9):** HP, HP fraction, block, intended total damage, hit count, strength, weak, vulnerable, targetable.

## Card-monster interaction tokens

One token per damaging hand card × targetable monster (random-target cards excluded).

- **References:** card-token index and monster-token index, after canonical sorting.
- **Numeric (6):** hits, total damage, HP damage after block, target HP after hit, target HP fraction after hit, playable on this target.

## Actions

`ActionToken`s (kind, source card, target monster, potion, selection task) are built for every legal action. They are not stored in Parquet and the value net does not use them. They are reserved for a policy head.

## Network

```text
card_h        = card_mlp(card embeddings, card numeric)
card_pool[z]  = Σ card_h over zone z          (hand, draw, discard, exhaust)
monster_pool  = Σ monster_mlp(monster embeddings, monster numeric)
interaction   = Σ interaction_mlp(card_h[c], monster_h[m], interaction numeric)
value         = tanh(head([global numeric, global embeddings, card pools, monster pool, interaction pool]))
```

Implementations: PyTorch `python/sts_combat_rl/models/deep_sets.py` (training) and native C++ `models/value_net.{hpp,cpp}` (play). A parity test keeps them in sync.

## Value meaning

Predicted final combat score, on the teacher's scale: win = (35 + hp + 4·potions)/(55 + max_hp), loss = 0. Labels are built at training time (`training/data.py::assign_targets`). Stored columns are described in `data/combat/README.md`.
