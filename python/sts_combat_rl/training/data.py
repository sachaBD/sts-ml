"""Validated combat_v3 Parquet rows and batching for value training."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pyarrow.dataset as ds
import torch
from torch.utils.data import Dataset

from ..schemas import combat_v3


def validate_rows(rows: list[dict[str, Any]]) -> None:
    """Check encoding widths, categorical bounds, and local interaction references."""
    bounds = {"input_state": 64, "card_selection_task": 32}
    card_bounds = {"card_id": 512, "zone": 5, "card_type": 5, "target_type": 4}
    monster_bounds = {"monster_id": 128, "move_id": 512}
    for row_number, row in enumerate(rows):
        if row.get("encoding_version") != 3:
            raise ValueError(f"row {row_number}: encoding_version must be 3")
        if len(row.get("global_numeric", [])) != 50:
            raise ValueError(f"row {row_number}: global_numeric must have width 50")
        for name, limit in bounds.items():
            if not 0 <= row[name] < limit:
                raise ValueError(
                    f"row {row_number}: {name} outside model bounds [0,{limit})"
                )
        for group, group_bounds, width in (
            ("cards", card_bounds, 14),
            ("monsters", monster_bounds, 9),
        ):
            for index, token in enumerate(row[group]):
                if len(token.get("numeric", [])) != width:
                    raise ValueError(
                        f"row {row_number} {group}[{index}]: numeric must have width {width}"
                    )
                for name, limit in group_bounds.items():
                    if not 0 <= token[name] < limit:
                        raise ValueError(
                            f"row {row_number} {group}[{index}]: {name} outside model bounds [0,{limit})"
                        )
        for index, interaction in enumerate(row["card_monster_interactions"]):
            if len(interaction.get("numeric", [])) != 6:
                raise ValueError(
                    f"row {row_number} interaction[{index}]: numeric must have width 6"
                )
            if not 0 <= interaction["card_index"] < len(row["cards"]):
                raise ValueError(
                    f"row {row_number} interaction[{index}]: card_index out of bounds"
                )
            if not 0 <= interaction["monster_index"] < len(row["monsters"]):
                raise ValueError(
                    f"row {row_number} interaction[{index}]: monster_index out of bounds"
                )


def episode_split(
    rows: list[dict[str, Any]], validation_fraction: float = 0.2, seed: int = 0
):
    """Split by run_seed so one run's fights never land on both sides.

    Returns train rows, validation rows, train episode_ids, validation episode_ids.
    """
    group = lambda r: r["run_seed"]
    groups = sorted({group(row) for row in rows})
    shuffled = groups[:]
    random.Random(seed).shuffle(shuffled)
    validation_count = (
        max(1, round(len(groups) * validation_fraction)) if len(groups) > 1 else 0
    )
    validation_groups = set(shuffled[:validation_count])
    train = [r for r in rows if group(r) not in validation_groups]
    valid = [r for r in rows if group(r) in validation_groups]
    return (
        train,
        valid,
        sorted({r["episode_id"] for r in train}),
        sorted({r["episode_id"] for r in valid}),
    )


def load_rows(
    path: str | Path,
    limit: int | None = None,
    categories: list[str] | None = None,
    encounters: list[str] | None = None,
    corrective: bool = False,
) -> list[dict[str, Any]]:
    """Load one combat_v3 Parquet shard, or every part in a run's out/ directory.

    categories / encounters keep only matching rows, e.g. ["boss"] / ["slime_boss"].
    Raises unless every part is tagged schema=combat_v3 in its parquet metadata. Ordinary loads reject
    DAgger parts; corrective=True accepts only them (collection_method=dagger, training_target=teacher_root_only).
    """
    dataset = ds.dataset(path, format="parquet", exclude_invalid_files=True)
    fragments = list(dataset.get_fragments())
    if not fragments:
        raise ValueError(f"{path}: no parquet parts")
    for fragment in fragments:
        schema = (fragment.physical_schema.metadata or {}).get(b"schema", b"").decode()
        if schema != combat_v3.NAME:
            raise ValueError(f"{fragment.path}: schema {schema or 'untagged'!r}, expected {combat_v3.NAME!r}")
        # DAgger parts (apps/dagger): teacher-root-only targets, learner outcomes; the blend would misuse them.
        metadata = fragment.physical_schema.metadata or {}
        if corrective:
            if metadata.get(b"collection_method") != b"dagger" or metadata.get(b"training_target") != b"teacher_root_only":
                raise ValueError(f"{fragment.path}: not a DAgger teacher_root_only part")
        elif b"training_target" in metadata:
            raise ValueError(f"{fragment.path}: corrective (DAgger) source; use [data].corrections, not an ordinary input")
    keep = None
    for column, values in (("category", categories), ("encounter", encounters)):
        if values:
            match = ds.field(column).isin(values)
            keep = match if keep is None else keep & match
    table = dataset.to_table(filter=keep)
    rows = table.slice(0, limit).to_pylist() if limit else table.to_pylist()
    validate_rows(rows)
    return rows


class ValueDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


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


def assign_targets(rows: list[dict[str, Any]], label: str, blend: float = 0.5) -> None:
    """Set row["target"].

    root:     teacher search estimate v = root_value
    terminal: fight outcome z = terminal_value
    blend:    blend*z + (1-blend)*v, except rows at or before the fight's random move
              use v only, because their z includes a move the teacher did not choose.
    """
    random_at: dict[int, int] = {}  # episode_id (one fight) -> last random decision_index
    for r in rows:
        if r["was_random"]:
            key = r["episode_id"]
            random_at[key] = max(random_at.get(key, -1), r["decision_index"])
    for r in rows:
        v, z = r["root_value"], r["terminal_value"]
        if r["row_kind"] == "child":  # off-trajectory position: the fight outcome is not its outcome
            r["target"] = v
        elif label == "root":
            r["target"] = v
        elif label == "terminal":
            r["target"] = z
        elif label == "blend":
            k = random_at.get(r["episode_id"], -1)
            r["target"] = v if r["decision_index"] <= k else blend * z + (1 - blend) * v
        else:
            raise ValueError(f"unknown label {label!r}")
