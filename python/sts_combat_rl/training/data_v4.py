"""Encoding v4 columns of combat_v3 rows (runs/README.md; combat/encoding.hpp v4_columns) for deep_sets_v3.

data.training_rows(..., v4=True) packs them next to the v3 columns (pack); Rows.collate adds the model inputs
(collate): the v4 state inputs, the legal-action tokens (policy inputs) and the targets
  policy_target  per action: the root visit share, identical tokens of a row merged (their visits summed, one
                 token kept), so the policy scores unique moves (the search merges identical cards too). Rows
                 without visits (child rows, forced moves the search didn't record, oracle rows) have no policy
                 target: policy_mask 0 and every action target 0.
  aux_keep       potions left at the end / potions held now (clipped to [0, 1]); aux_keep_mask: won decision rows
                 (aux_hp_mask) holding a potion.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch

VERSION = 4
COLUMNS = ("v4_encoding_version", "player_numeric", "max_hp", "potions", "oracle", "monster_v4", "potion_tokens",
           "relic_tokens", "legal_actions", "actions")
# token groups added by v4 (their <group>.count per row and per-token fields, like data.TOKENS)
GROUPS = ("potion_tokens", "relic_tokens", "legal_actions")
BOUNDS = {"potion_id": 64, "relic_id": 192, "move_id": 512, "kind": 8, "card_selection_task": 32, "card_id": 512,
          "zone": 5, "card_type": 5, "target_type": 4, "monster_id": 128}
_HASH = np.random.default_rng(0x5eed).integers(1, 2**63, size=256, dtype=np.uint64) | np.uint64(1)


def _check(ok, rows, message):
    if not np.all(ok):
        raise ValueError(f"row {np.asarray(rows)[np.argmin(ok)]}: {message}")


def _numpy(column):
    return column.to_numpy(zero_copy_only=False)


def _lengths(column):
    return pc.fill_null(pc.list_value_length(column), 0).to_numpy(zero_copy_only=False).astype(np.int64)


def _fixed(values, width, rows, what):
    """A fixed-width list array (nulls -> zeros) as float32 [len, width]."""
    valid = _numpy(values.is_valid()) if len(values) else np.zeros(0, bool)
    lengths = _lengths(values)
    _check(lengths[valid] == width, rows[valid], f"{what} must have width {width}")
    out = np.zeros((len(values), width), np.float32)
    if valid.any():
        out[valid] = _numpy(pc.list_flatten(values.filter(pa.array(valid)))).astype(np.float32).reshape(-1, width)
    return out, valid.astype(np.float32)


def _ints(values, field, rows, what):
    out = _numpy(pc.fill_null(values, 0)).astype(np.int64)
    if field in BOUNDS:
        _check((out >= 0) & (out < BOUNDS[field]), rows, f"{what} {field} outside model bounds [0,{BOUNDS[field]})")
    return out.astype(np.int16)


def pack(batch: pa.RecordBatch, first_row: int, out: dict[str, np.ndarray]) -> None:
    """Adds the v4 columns of one record batch to data._pack's `out` (which has monsters.count)."""
    n = batch.num_rows
    rows = np.arange(first_row, first_row + n)
    version = _numpy(pc.fill_null(batch.column("v4_encoding_version"), 0))
    _check(version == VERSION, rows, f"v4_encoding_version must be {VERSION} (rows written before encoding v4 "
                                     "have no v4 columns)")
    out["player_numeric"], _ = _fixed(batch.column("player_numeric"), 12, rows, "player_numeric")
    out["max_hp"] = _numpy(batch.column("max_hp")).astype(np.float32)
    out["potions"] = _numpy(batch.column("potions")).astype(np.float32)
    out["oracle"] = _numpy(pc.fill_null(batch.column("oracle"), False)).astype(bool)

    counts = _lengths(batch.column("monster_v4"))
    _check(counts == out["monsters.count"], rows, "monster_v4 must align with monsters")
    tokens = pc.list_flatten(batch.column("monster_v4"))
    owners = np.repeat(rows, counts)
    out["monsters.previous_move_id"] = _ints(pc.struct_field(tokens, "previous_move_id"), "move_id", owners, "monster")
    out["monsters.status"], _ = _fixed(pc.struct_field(tokens, "status"), 15, owners, "monster status")

    for group, field, width in (("potion_tokens", "potion_id", 18), ("relic_tokens", "relic_id", 3)):
        counts = _lengths(batch.column(group))
        tokens = pc.list_flatten(batch.column(group))
        owners = np.repeat(rows, counts)
        out[group + ".count"] = counts
        out[f"{group}.{field}"] = _ints(pc.struct_field(tokens, field), field, owners, group)
        out[group + ".numeric"], _ = _fixed(pc.struct_field(tokens, "numeric"), width, owners, group)

    _pack_actions(batch, rows, out)


def _pack_actions(batch, rows, out):
    column = batch.column("legal_actions")
    counts = _lengths(column)
    a = pc.list_flatten(column)
    owners = np.repeat(np.arange(len(rows)), counts)
    f = {}
    for field in ("action", "kind", "card_selection_task"):
        f[field] = _ints(pc.struct_field(a, field), field, rows[owners], "legal action")
    f["skips_selection"] = _numpy(pc.fill_null(pc.struct_field(a, "skips_selection"), False)).astype(np.float32)
    f["discards_potion"] = _numpy(pc.fill_null(pc.struct_field(a, "discards_potion"), False)).astype(np.float32)
    card, monster, potion = (pc.struct_field(a, x) for x in ("card", "monster", "potion"))
    f["card_mask"] = _numpy(card.is_valid()).astype(np.float32)
    for field, name in (("card_id", "card_id"), ("zone", "card_zone"), ("card_type", "card_type"),
                        ("target_type", "target_type")):
        f[name] = _ints(pc.struct_field(card, field), field, rows[owners], "action card")
    f["card_numeric"], _ = _fixed(pc.struct_field(card, "numeric"), 14, rows[owners], "action card numeric")
    f["monster_mask"] = _numpy(monster.is_valid()).astype(np.float32)
    f["monster_id"] = _ints(pc.struct_field(monster, "monster_id"), "monster_id", rows[owners], "action monster")
    f["move_id"] = _ints(pc.struct_field(monster, "move_id"), "move_id", rows[owners], "action monster")
    f["previous_move_id"] = _ints(pc.struct_field(monster, "previous_move_id"), "move_id", rows[owners],
                                  "action monster")
    f["monster_numeric"], _ = _fixed(pc.struct_field(monster, "numeric"), 9, rows[owners], "action monster numeric")
    f["monster_status"], _ = _fixed(pc.struct_field(monster, "status"), 15, rows[owners], "action monster status")
    f["potion_mask"] = _numpy(potion.is_valid()).astype(np.float32)
    f["potion_id"] = _ints(pc.struct_field(potion, "potion_id"), "potion_id", rows[owners], "action potion")
    f["potion_numeric"], _ = _fixed(pc.struct_field(potion, "numeric"), 18, rows[owners], "action potion numeric")
    f["interaction_numeric"], f["interaction_mask"] = _fixed(pc.struct_field(a, "interaction"), 6, rows[owners],
                                                             "action interaction")

    # root visits per legal action (the actions column holds the tried moves by the same index)
    tried = batch.column("actions")
    tried_counts = _lengths(tried)
    t = pc.list_flatten(tried)
    t_owner = np.repeat(np.arange(len(rows)), tried_counts)
    t_key = t_owner.astype(np.int64) * 65536 + _numpy(pc.struct_field(t, "action")).astype(np.int64)
    t_visits = _numpy(pc.struct_field(t, "visits")).astype(np.float64)
    a_key = owners.astype(np.int64) * 65536 + f["action"].astype(np.int64)
    order = np.argsort(t_key)
    pos = np.searchsorted(t_key[order], a_key)
    pos = np.minimum(pos, max(len(order) - 1, 0))
    found = (t_key[order][pos] == a_key) if len(order) else np.zeros(len(a_key), bool)
    visits = np.where(found, t_visits[order][pos] if len(order) else 0.0, 0.0)
    # every tried move must be a legal action of its row
    _check(np.isin(t_key, a_key) | (counts[t_owner] == 0), rows[t_owner],
           "a tried action is not among the row's legal_actions")

    # merge identical tokens within a row: hash every field except the index
    parts = [f[k].reshape(len(owners), -1).astype(np.float32) for k in sorted(f) if k != "action"]
    features = np.concatenate(parts, 1) if parts else np.zeros((0, 0), np.float32)
    bits = np.ascontiguousarray(features).view(np.uint32).astype(np.uint64)
    hashes = (bits * _HASH[: bits.shape[1]]).sum(axis=1) if len(bits) else np.zeros(0, np.uint64)
    order = np.lexsort((np.arange(len(owners)), hashes, owners))
    first = np.ones(len(order), bool)
    first[1:] = (owners[order][1:] != owners[order][:-1]) | (hashes[order][1:] != hashes[order][:-1])
    group = np.cumsum(first) - 1
    merged = np.zeros(int(first.sum()), np.float64)
    np.add.at(merged, group, visits[order])
    keep = np.sort(order[first])  # the first token of each identical group, in the row's order
    keep_visits = np.zeros(len(owners))
    keep_visits[order[first]] = merged
    keep_owner = owners[keep]

    total = np.zeros(len(rows))
    np.add.at(total, keep_owner, keep_visits[keep])
    policy_mask = (total > 0) & ~out["oracle"]
    target = np.where(policy_mask[keep_owner], keep_visits[keep] / np.maximum(total[keep_owner], 1e-12), 0.0)
    out["legal_actions.count"] = np.bincount(keep_owner, minlength=len(rows)).astype(np.int64)
    for k, v in f.items():
        out["legal_actions." + k] = v[keep]
    out["legal_actions.policy_target"] = target.astype(np.float32)
    out["policy_mask"] = policy_mask.astype(np.float32)


def assign_keep_targets(columns: dict[str, np.ndarray]) -> None:
    """aux_keep / aux_keep_mask (needs data.assign_aux_targets' aux_hp_mask)."""
    held = columns["potion_tokens.count"].astype(np.float32)
    columns["aux_keep"] = np.clip(columns["potions"] / np.maximum(held, 1.0), 0.0, 1.0).astype(np.float32)
    columns["aux_keep_mask"] = (columns["aux_hp_mask"] * (held > 0)).astype(np.float32)


def collate(c: dict[str, np.ndarray], starts: dict[str, np.ndarray], index: np.ndarray,
            monster_tokens: np.ndarray) -> dict[str, torch.Tensor]:
    """The v4 inputs of rows `index` (data.Rows.collate; monster_tokens: the batch's monster token positions)."""
    long = lambda x: torch.from_numpy(x.astype(np.int64))
    floats = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    out = {"player_numeric": floats(c["player_numeric"][index]), "max_hp": floats(c["max_hp"][index]),
           "previous_move_ids": long(c["monsters.previous_move_id"][monster_tokens]),
           "monster_status": floats(c["monsters.status"][monster_tokens])}
    take = {}
    for group in GROUPS:
        counts = c[group + ".count"][index]
        owner = np.repeat(np.arange(len(index)), counts)
        first = np.cumsum(counts) - counts
        take[group] = starts[group][index][owner] + np.arange(len(owner)) - first[owner], owner
    for group, field, name in (("potion_tokens", "potion_id", "potion"), ("relic_tokens", "relic_id", "relic")):
        tokens, owner = take[group]
        out.update({f"{name}_ids": long(c[f"{group}.{field}"][tokens]),
                    f"{name}_numeric": floats(c[group + ".numeric"][tokens]),
                    f"{name}_state_indices": long(owner)})
    tokens, owner = take["legal_actions"]
    g = lambda k: c["legal_actions." + k][tokens]
    out.update(
        action_state_indices=long(owner), action_kinds=long(g("kind")), action_tasks=long(g("card_selection_task")),
        action_skips=floats(g("skips_selection")), action_discards=floats(g("discards_potion")), action_card_mask=floats(g("card_mask")),
        action_card_ids=long(g("card_id")), action_card_zones=long(g("card_zone")),
        action_card_types=long(g("card_type")), action_target_types=long(g("target_type")),
        action_card_numeric=floats(g("card_numeric")), action_monster_mask=floats(g("monster_mask")),
        action_monster_ids=long(g("monster_id")), action_move_ids=long(g("move_id")),
        action_previous_move_ids=long(g("previous_move_id")), action_monster_numeric=floats(g("monster_numeric")),
        action_monster_status=floats(g("monster_status")), action_potion_mask=floats(g("potion_mask")),
        action_potion_ids=long(g("potion_id")), action_potion_numeric=floats(g("potion_numeric")),
        action_interaction_mask=floats(g("interaction_mask")),
        action_interaction_numeric=floats(g("interaction_numeric")),
        policy_target=floats(g("policy_target")), policy_mask=floats(c["policy_mask"][index]),
        aux_keep=floats(c["aux_keep"][index]), aux_keep_mask=floats(c["aux_keep_mask"][index]),
    )
    return out
