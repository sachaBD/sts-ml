# Combat NN encoding v2

Scope: Ironclad starter deck versus Slime Boss. Supported cards are Strike, Defend, Bash, Slimed, Flame Barrier, Combust, Hemokinesis, and Battle Trance. Encoding fails on unsupported cards rather than silently emitting incomplete features.

Schema version: `2`. The earlier minimal semantic encoding was version `0`.

The scenario must contain no usable potions or unencoded combat-affecting relics. Broader scenarios require an encoding-version change or explicit feature support.

The encoding exposes public combat quantities after engine modifiers so the network learns strategy rather than arithmetic. Cards, monsters, and card–monster interactions are variable-length sets pooled by summation.

## Global features

### Categorical

- `input_state`
- `card_selection_task`

### Numeric vector

| Index | Feature | Model input |
|---:|---|---|
| 0 | turn | `turn / 20` |
| 1 | player current HP | `hp / 100` |
| 2 | player HP fraction | `hp / max_hp` |
| 3 | block | `block / 100` |
| 4 | energy | `energy / 10` |
| 5 | energy per turn | `energy_per_turn / 10` |
| 6 | strength | `strength / 10` |
| 7 | dexterity | `dexterity / 10` |
| 8 | weak duration | `weak / 10` |
| 9 | vulnerable duration | `vulnerable / 10` |
| 10 | frail duration | `frail / 10` |
| 11 | cards played this turn | `count / 20` |
| 12 | hand size | `count / 10` |
| 13 | draw size | `count / 64` |
| 14 | discard size | `count / 64` |
| 15 | exhaust size | `count / 64` |
| 16 | total incoming attack damage | `damage / 100` |
| 17 | incoming HP loss after current block | `max(0, damage - block) / 100` |
| 18 | active Combust damage | `damage / 10` |
| 19 | Combust HP loss | `loss / 10` |
| 20 | active Flame Barrier retaliation | `damage / 50` |
| 21 | NO_DRAW active | 0 or 1 |

Incoming damage is calculated from public intents after monster strength/weak and player vulnerable. Non-damage intent effects remain represented by move ID. The aggregate incoming fields intentionally duplicate information available in monster tokens: the derived channels avoid requiring the network to learn summation and block subtraction before it can learn strategy.

Maximum HP is not a separate input. Current HP and HP fraction provide the useful absolute and relative views.

## Card tokens

One token per observable card in hand, draw, discard, or exhaust.

### Categorical

- card ID;
- zone: hand, draw, discard, exhaust;
- card type: attack, skill, power, status, curse;
- target type: none, one enemy, all enemies, random enemy.

### Numeric vector

| Index | Feature | Model input |
|---:|---|---|
| 0 | upgraded | 0 or 1 |
| 1 | base cost | `cost / 3` |
| 2 | current cost | `cost_for_turn / 3` |
| 3 | base damage per hit | `damage / 50` |
| 4 | hit count | `hits / 10` |
| 5 | base block | `block / 50` |
| 6 | effective block now | engine result after dexterity/frail, `/ 50` |
| 7 | cards drawn | `count / 10` |
| 8 | card-specific counter | `special_data / 10` |
| 9 | free to play | 0 or 1 |
| 10 | exhausts | 0 or 1 |
| 11 | ethereal | 0 or 1 |
| 12 | retained | 0 or 1 |
| 13 | playable now | 0 or 1 |

Static damage, hits, block, and draw come from a project-owned table covering exactly Strike, Defend, Bash, and Slimed. Effective damage uses `BattleContext::calculateCardDamage`; effective block uses `BattleContext::calculateCardBlock`. Tests compare engineered results with executing each supported card. Card ID remains present for Bash's vulnerable effect and Slimed's exhaust effect.

Do not encode card instance ID or unknown draw-pile order.

## Monster tokens

One token per relevant monster. Slime Boss combat reaches at most four living enemies, but this is not a tensor bound.

### Categorical

- monster ID;
- current move/intent.

### Numeric vector

| Index | Feature | Model input |
|---:|---|---|
| 0 | current HP | `hp / 100` |
| 1 | HP fraction | `hp / max_hp` |
| 2 | block | `block / 100` |
| 3 | total intended damage | `(damage_per_hit * hits) / 100` |
| 4 | intended hit count | `hits / 10` |
| 5 | strength | `strength / 10` |
| 6 | weak duration | `weak / 10` |
| 7 | vulnerable duration | `vulnerable / 10` |
| 8 | targetable | 0 or 1 |

No `below_half`, `will_split`, or distance-to-split feature is supplied. Monster ID and HP fraction expose the facts needed to learn phase thresholds.

## Card–monster interaction tokens

For each hand card that deals direct damage and each affected target, emit one interaction token.

### References

- semantic card-token index;
- semantic monster-token index.

These references are local to the encoded state after canonical token sorting. They are not simulator hand or monster-slot indices.

### Numeric vector

| Index | Feature | Model input |
|---:|---|---|
| 0 | hit count | `hits / 10` |
| 1 | total effective damage | after strength, weak, vulnerable, and hit count, `/ 100` |
| 2 | HP damage after target block | `/ 100` |
| 3 | target HP after immediate damage | `/ 100` |
| 4 | target HP fraction after immediate damage | `remaining_hp / max_hp` |
| 5 | playable on this target now | 0 or 1 |

The resulting-HP fields are intentional deterministic one-step action features computed only from public information. They expose arithmetic, not encounter rules: the encoder never says whether the resulting HP triggers a split or what a split produces.

For all-enemy attacks, emit an interaction for every affected monster. Cards without direct damage have no card–monster interaction token; their block and other basic effects remain on the card token.

## Network

```text
card_h[i] = card_mlp(card categorical embeddings, card numeric features)
card_pool[z] = sum(card_h[i] for cards in zone z)

monster_h[j] = monster_mlp(monster categorical embeddings, monster numeric features)
monster_pool = sum(monster_h[j])

interaction_h[k] = interaction_mlp(
    card_h[interaction.card],
    monster_h[interaction.monster],
    interaction numeric features)
interaction_pool = sum(interaction_h[k])

context = concat(
    global numeric features,
    global categorical embeddings,
    card_pool[hand],
    card_pool[draw],
    card_pool[discard],
    card_pool[exhaust],
    monster_pool,
    interaction_pool)

value = tanh(value_mlp(context))
```

No positional encoding is used. For batching, concatenate tokens and carry state indices for segmented sums.

## Value meaning

```text
loss: -1
win:  0.5 + 0.5 * final_player_hp / player_max_hp
```

The network predicts expected final combat return from the current public-information state. Initial targets may use MCTS root estimates. Store terminal results separately.

## Parquet row

One row represents one player decision. Columns have a fixed schema; token columns contain variable-length lists.

```text
encoding_version
episode_id
seed
decision_index

global_numeric                 fixed_size_list<float32>[22]
input_state                    int16
card_selection_task            int16

cards                          list<struct<
    card_id: int16,
    zone: int8,
    card_type: int8,
    target_type: int8,
    numeric: fixed_size_list<float32>[14]
>>

monsters                       list<struct<
    monster_id: int16,
    move_id: int16,
    numeric: fixed_size_list<float32>[9]
>>

card_monster_interactions      list<struct<
    card_index: int16,
    monster_index: int8,
    numeric: fixed_size_list<float32>[6]
>>

mcts_value                     float32
terminal_outcome               int8
final_player_hp                int16
```

Parquet stores deterministic model inputs, not learned embeddings or pooled activations. PyArrow writes these columns directly; JSON is not part of training.

## Invariants

- Card ordering within a zone does not change the value.
- Monster ordering does not change the value.
- Temporary hand indices and monster slots may be used during construction only.
- After canonical card and monster sorting, interaction references are remapped to token indices.
- Interaction tokens are then sorted by referenced semantic card key, semantic monster key, and numeric fields; temporary routing identity is not serialized.
- Hidden RNG and unknown draw order do not change the encoding.
- Card instance ID, simulator monster slot, and action execution index are never model inputs.
- Feature-order or normalization changes increment `encoding_version`.

## Deferred from v1

- policy/action tensors;
- potions and relics;
- complete semantic features for every card in the game;
- powers not needed by the initial scenario;
- known top-card position;
- attention layers.
