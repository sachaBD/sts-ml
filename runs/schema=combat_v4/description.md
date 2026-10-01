# `combat_v4`

## Goal

Record vanilla Slay the Spire combats losslessly without storing a complete state at every decision.

## Tables

### `fights`

One record is the complete simulator state immediately before the first ordinary player action of one combat branch. It includes all simulator state required to restore that boundary, including RNG.

**Key fields:** `fight_id` is an opaque globally unique combat-branch ID.

### `steps`

One record is one simulator action in a fight's replay trace. Records are ordered by `fight_id, step_index`. `action_bits` is the exact Lightspeed action representation executed at that point; it includes ordinary actions and any subsequent in-combat selection actions.

**Key fields:** `(fight_id, step_index)` is unique. `step_index` is zero-based.

## Reconstruction

Restore `fights.initial_state`, enumerate and execute each ordered `steps.action_bits`, and obtain each later combat state deterministically. This includes states at card-selection prompts without storing a separate snapshot for them.

## Joinable Fields

| field | target | scope | cardinality |
|---|---|---|---|
| `fights.fight_id` | `steps.fight_id` | one combat branch | one fight to many steps |
| `fight_id` | future combat result schemas | one combat branch | one-to-one |
| `(fight_id, step_index)` | future derived snapshot/action-label schemas | one action boundary | one-to-many |
