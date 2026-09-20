"""Validated Parquet rows and batching for schema-v2 value training."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


def validate_rows(rows: list[dict[str, Any]]) -> None:
    """Check encoding widths, categorical bounds, and local interaction references."""
    bounds = {"input_state": 64, "card_selection_task": 32}
    card_bounds = {"card_id": 512, "zone": 5, "card_type": 5, "target_type": 4}
    monster_bounds = {"monster_id": 128, "move_id": 512}
    for row_number, row in enumerate(rows):
        if row.get("encoding_version") != 2:
            raise ValueError(f"row {row_number}: encoding_version must be 2")
        if len(row.get("global_numeric", [])) != 22:
            raise ValueError(f"row {row_number}: global_numeric must have width 22")
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
    episode_ids = sorted({row["episode_id"] for row in rows})
    shuffled = episode_ids[:]
    random.Random(seed).shuffle(shuffled)
    validation_count = (
        max(1, round(len(episode_ids) * validation_fraction))
        if len(episode_ids) > 1
        else 0
    )
    validation_ids = set(shuffled[:validation_count])
    return (
        [r for r in rows if r["episode_id"] not in validation_ids],
        [r for r in rows if r["episode_id"] in validation_ids],
        sorted(set(episode_ids) - validation_ids),
        sorted(validation_ids),
    )


def load_rows(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    table = pq.read_table(path)
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
        "target": torch.tensor(
            [r.get("mcts_value", 0.0) for r in rows], dtype=torch.float32
        ),
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
