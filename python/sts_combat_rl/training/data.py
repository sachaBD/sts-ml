"""Validated combat_v3 Parquet rows and batching for value training."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

from .. import query
from ..run import run_parquet


def validation_run_seeds(run_seeds, validation_fraction: float, seed: int) -> set[int]:
    """The run seeds episode_split puts on the validation side."""
    groups = sorted(run_seeds)
    shuffled = groups[:]
    random.Random(seed).shuffle(shuffled)
    validation_count = (
        max(1, round(len(groups) * validation_fraction)) if len(groups) > 1 else 0
    )
    return set(shuffled[:validation_count])


def episode_split(
    rows: list[dict[str, Any]], validation_fraction: float = 0.2, seed: int = 0
):
    """Split by run_seed so one run's fights never land on both sides.

    Returns train rows, validation rows, train episode_ids, validation episode_ids.
    """
    group = lambda r: r["run_seed"]
    validation_groups = validation_run_seeds({group(row) for row in rows}, validation_fraction, seed)
    train = [r for r in rows if group(r) not in validation_groups]
    valid = [r for r in rows if group(r) in validation_groups]
    return (
        train,
        valid,
        sorted({r["episode_id"] for r in train}),
        sorted({r["episode_id"] for r in valid}),
    )


def collate_states(rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    output = {
        "global_numeric": torch.tensor(
            [r["global_numeric"] for r in rows], dtype=torch.float32
        ),
        "input_state": torch.tensor([r["input_state"] for r in rows]),
        "card_selection_task": torch.tensor([r["card_selection_task"] for r in rows]),
        "target": torch.tensor([r.get("target", 0.0) for r in rows], dtype=torch.float32),
    }
    cards = []
    monsters = []
    interactions = []
    card_owners = []
    monster_owners = []
    interaction_owners = []
    card_offset = monster_offset = 0
    for owner, row in enumerate(rows):
        cards.extend(row["cards"])
        monsters.extend(row["monsters"])
        card_owners += [owner] * len(row["cards"])
        monster_owners += [owner] * len(row["monsters"])
        interactions.extend(
            [(x, card_offset, monster_offset) for x in row["card_monster_interactions"]]
        )
        interaction_owners += [owner] * len(row["card_monster_interactions"])
        card_offset += len(row["cards"])
        monster_offset += len(row["monsters"])
    long = lambda xs, key: torch.tensor([x[key] for x in xs], dtype=torch.long)
    output.update(
        card_ids=long(cards, "card_id"),
        card_zones=long(cards, "zone"),
        card_types=long(cards, "card_type"),
        target_types=long(cards, "target_type"),
        card_numeric=torch.tensor(
            [x["numeric"] for x in cards], dtype=torch.float32
        ).reshape(-1, 14),
        monster_ids=long(monsters, "monster_id"),
        move_ids=long(monsters, "move_id"),
        monster_numeric=torch.tensor(
            [x["numeric"] for x in monsters], dtype=torch.float32
        ).reshape(-1, 9),
        interaction_cards=torch.tensor(
            [x[0]["card_index"] + x[1] for x in interactions]
        ),
        interaction_monsters=torch.tensor(
            [x[0]["monster_index"] + x[2] for x in interactions]
        ),
        interaction_numeric=torch.tensor(
            [x[0]["numeric"] for x in interactions], dtype=torch.float32
        ).reshape(-1, 6),
        card_state_indices=torch.tensor(card_owners),
        monster_state_indices=torch.tensor(monster_owners),
        interaction_state_indices=torch.tensor(interaction_owners),
    )
    return output


# Columnar rows for training: an Arrow stream from the SQL query packed into numpy arrays (a few KB per row,
# not tens of KB as Python dicts). Same content as the dict rows collate_states takes.
META = ("run_id", "run_seed", "episode_id", "decision_index", "row_kind", "was_random", "root_value",
        "terminal_value", "encounter", "category")
STATE = ("encoding_version", "input_state", "card_selection_task", "global_numeric")
TOKENS = {  # token group: (categorical fields, numeric width)
    "cards": (("card_id", "zone", "card_type", "target_type"), 14),
    "monsters": (("monster_id", "move_id"), 9),
    "card_monster_interactions": (("card_index", "monster_index"), 6),
}
BOUNDS = {"input_state": 64, "card_selection_task": 32, "card_id": 512, "zone": 5, "card_type": 5,
          "target_type": 4, "monster_id": 128, "move_id": 512}
STRINGS = ("run_id", "row_kind", "encounter", "category")


def _check(ok: np.ndarray, row_numbers: np.ndarray, message: str) -> None:
    if not ok.all():
        raise ValueError(f"row {row_numbers[np.argmin(ok)]}: {message}")


def _list(column: pa.Array) -> tuple[np.ndarray, pa.Array]:
    """A list column's per-row lengths and its flattened values."""
    return pc.list_value_length(column).to_numpy(zero_copy_only=False), column.flatten()


def _numpy(column: pa.Array) -> np.ndarray:
    return column.to_numpy(zero_copy_only=False)


def _pack(batch: pa.RecordBatch, first_row: int) -> dict[str, np.ndarray]:
    """One record batch as numpy columns, validated (encoding widths, categorical bounds, interaction refs)."""
    n = batch.num_rows
    rows = np.arange(first_row, first_row + n)
    out = {}
    for name in META + STATE[:-1]:
        column = batch.column(name)
        if name in STRINGS:  # dictionary-encoded so equal strings share one object
            encoded = pc.dictionary_encode(column)
            out[name] = _numpy(encoded.dictionary).astype(object)[_numpy(encoded.indices)]
        else:
            out[name] = _numpy(column)
    _check(out["encoding_version"] == 3, rows, "encoding_version must be 3")
    for name in ("input_state", "card_selection_task"):
        _check((out[name] >= 0) & (out[name] < BOUNDS[name]), rows, f"{name} outside model bounds [0,{BOUNDS[name]})")
    lengths, values = _list(batch.column("global_numeric"))
    _check(lengths == 50, rows, "global_numeric must have width 50")
    out["global_numeric"] = _numpy(values).astype(np.float32).reshape(n, 50)
    for group, (fields, width) in TOKENS.items():
        counts, tokens = _list(batch.column(group))
        owners = np.repeat(rows, counts)
        out[group + ".count"] = counts.astype(np.int64)
        for field in fields:
            values = _numpy(pc.struct_field(tokens, field)).astype(np.int16)
            if field in BOUNDS:
                _check((values >= 0) & (values < BOUNDS[field]), owners,
                       f"{group} {field} outside model bounds [0,{BOUNDS[field]})")
            out[f"{group}.{field}"] = values
        lengths, values = _list(pc.struct_field(tokens, "numeric"))
        _check(lengths == width, owners, f"{group} numeric must have width {width}")
        out[group + ".numeric"] = _numpy(values).astype(np.float32).reshape(-1, width)
    owners = np.repeat(np.arange(n), out["card_monster_interactions.count"])
    for field, group in (("card_index", "cards"), ("monster_index", "monsters")):
        index = out["card_monster_interactions." + field]
        _check((index >= 0) & (index < out[group + ".count"][owners]), owners + first_row,
               f"interaction {field} out of bounds")
    return out


class Rows:
    """combat_v3 rows column-wise. Per row: META and STATE columns, `target`, `weight`; per token group:
    `<group>.count` per row, and `<group>.<field>` / `<group>.numeric` per token (rows' tokens in row order)."""

    def __init__(self, columns: dict[str, np.ndarray]):
        self.columns = columns
        columns.setdefault("target", np.zeros(len(columns["episode_id"])))
        columns.setdefault("weight", np.ones(len(columns["episode_id"])))
        self.starts = {group: np.cumsum(columns[group + ".count"]) - columns[group + ".count"] for group in TOKENS}

    def __len__(self) -> int:
        return len(self.columns["episode_id"])

    def __getitem__(self, name: str) -> np.ndarray:
        return self.columns[name]

    @staticmethod
    def concat(parts: list[Rows]) -> Rows:
        return Rows({key: np.concatenate([p.columns[key] for p in parts]) for key in parts[0].columns})

    def collate(self, index, weight: bool = False) -> dict[str, torch.Tensor]:
        """collate_states of rows `index` (plus loss weights if `weight`)."""
        index = np.asarray(index, dtype=np.int64)
        c = self.columns
        long = lambda a: torch.from_numpy(a.astype(np.int64))
        out = {
            "global_numeric": torch.from_numpy(c["global_numeric"][index]),
            "input_state": long(c["input_state"][index]),
            "card_selection_task": long(c["card_selection_task"][index]),
            "target": torch.from_numpy(c["target"][index].astype(np.float32)),
        }
        if weight:
            out["weight"] = torch.from_numpy(c["weight"][index].astype(np.float32))
        tokens, owners, firsts = {}, {}, {}
        for group in TOKENS:
            counts = c[group + ".count"][index]
            owner = np.repeat(np.arange(len(index)), counts)
            first = np.cumsum(counts) - counts  # batch position of each row's first token
            tokens[group] = self.starts[group][index][owner] + np.arange(len(owner)) - first[owner]
            owners[group], firsts[group] = owner, first
        take = lambda group, field: c[f"{group}.{field}"][tokens[group]]
        inter, owner = "card_monster_interactions", owners["card_monster_interactions"]
        out.update(
            card_ids=long(take("cards", "card_id")),
            card_zones=long(take("cards", "zone")),
            card_types=long(take("cards", "card_type")),
            target_types=long(take("cards", "target_type")),
            card_numeric=torch.from_numpy(take("cards", "numeric")),
            monster_ids=long(take("monsters", "monster_id")),
            move_ids=long(take("monsters", "move_id")),
            monster_numeric=torch.from_numpy(take("monsters", "numeric")),
            interaction_cards=long(take(inter, "card_index") + firsts["cards"][owner]),
            interaction_monsters=long(take(inter, "monster_index") + firsts["monsters"][owner]),
            interaction_numeric=torch.from_numpy(take(inter, "numeric")),
            card_state_indices=long(owners["cards"]),
            monster_state_indices=long(owners["monsters"]),
            interaction_state_indices=long(owner),
        )
        return out


class RowLoader(DataLoader):
    """Batches of Rows `index` (a DataLoader over row positions, so shuffling matches a list-of-rows loader)."""

    def __init__(self, rows: Rows, index: np.ndarray, batch_size: int, shuffle: bool = False,
                 weight: bool = False, generator=None):
        super().__init__(index, batch_size=batch_size, shuffle=shuffle, num_workers=0, generator=generator,
                         collate_fn=lambda positions: rows.collate(positions, weight))


def training_rows(sql: str, corrective: bool = False, oracle: bool = False) -> Rows:
    """The combat_v3 rows of `sql` (sts_combat_rl.query) to train on, validated.

    DAgger runs (parts tagged collection_method=dagger: teacher-root-only targets, learner outcomes; the blend
    would misuse them) are only accepted as corrections, and corrections only from them. Oracle rows raise
    unless oracle=True.
    """
    parts, count = [], 0
    for batch in query.batches(sql, META + STATE + tuple(TOKENS), oracle=oracle):
        parts.append(_pack(batch, count))
        count += batch.num_rows
    if not count:
        raise ValueError(f"no rows: {sql}")
    columns = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    del parts
    for run_id in sorted(set(columns["run_id"])):
        dagger = {(pq.read_schema(part).metadata or {}).get(b"collection_method") == b"dagger"
                  for part in run_parquet(run_id)}
        if dagger != {corrective}:
            raise ValueError(f"{run_id}: " + ("not a DAgger run; corrections come only from DAgger runs" if corrective
                                              else "DAgger rows; use them as corrections, not ordinary data"))
    return Rows(columns)


def split_rows(rows: Rows, validation_fraction: float = 0.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """episode_split of Rows: train and validation row indices."""
    seeds = rows["run_seed"]
    validation = np.isin(seeds, list(validation_run_seeds(np.unique(seeds).tolist(), validation_fraction, seed)))
    return np.flatnonzero(~validation), np.flatnonzero(validation)


def assign_targets(rows: Rows, label: str, blend: float = 0.5) -> None:
    """Set rows["target"].

    root:     teacher search estimate v = root_value
    terminal: fight outcome z = terminal_value
    blend:    blend*z + (1-blend)*v, except rows at or before the fight's random move
              use v only, because their z includes a move the teacher did not choose.
    Child rows (off-trajectory positions: the fight outcome is not their outcome) always use v.
    """
    v = rows["root_value"].astype(np.float64)
    z = rows["terminal_value"].astype(np.float64)
    if label == "root":
        target = v
    elif label == "terminal":
        target = z
    elif label == "blend":
        episodes, episode = np.unique(rows["episode_id"], return_inverse=True)
        random_at = np.full(len(episodes), -1, dtype=np.int64)  # per fight: last random decision_index
        was_random = rows["was_random"].astype(bool)
        np.maximum.at(random_at, episode[was_random], rows["decision_index"][was_random])
        target = np.where(rows["decision_index"] <= random_at[episode], v, blend * z + (1 - blend) * v)
    else:
        raise ValueError(f"unknown label {label!r}")
    rows.columns["target"] = np.where(rows["row_kind"] == "child", v, target)
