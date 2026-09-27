"""Batches of encoding v4 states (combat/encoding.hpp to_json_v4 dicts) as DeepSetsV3 keyword inputs.

A starting point for the future combat_v4 table loader; today used by the Python / C++ parity test.
"""

from __future__ import annotations

from typing import Any

import torch

VERSION = 4


def collate_v4(states: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """model(**collate_v4(states)) for a DeepSetsV3."""
    cards, monsters, interactions, potions, relics = [], [], [], [], []
    owners = {k: [] for k in ("card", "monster", "interaction", "potion", "relic")}
    card_offset = monster_offset = 0
    for owner, s in enumerate(states):
        if s["encoding_version"] != VERSION:
            raise ValueError(f"state is encoding v{s['encoding_version']}, not v{VERSION}")
        for x in s["card_monster_interactions"]:
            interactions.append((x, card_offset, monster_offset))
        for name, items, bucket in (("card", s["cards"], cards), ("monster", s["monsters"], monsters),
                                    ("interaction", s["card_monster_interactions"], None),
                                    ("potion", s["potions"], potions), ("relic", s["relics"], relics)):
            if bucket is not None:
                bucket.extend(items)
            owners[name] += [owner] * len(items)
        card_offset += len(s["cards"])
        monster_offset += len(s["monsters"])

    def long(items, key):
        return torch.tensor([x[key] for x in items], dtype=torch.long)

    def floats(items, key, width):
        return torch.tensor([x[key] for x in items], dtype=torch.float32).reshape(len(items), width)

    return dict(
        global_numeric=torch.tensor([s["global_numeric"] for s in states], dtype=torch.float32),
        player_numeric=torch.tensor([s["player_numeric"] for s in states], dtype=torch.float32),
        max_hp=torch.tensor([s["max_hp"] for s in states], dtype=torch.float32),
        input_state=torch.tensor([s["input_state"] for s in states], dtype=torch.long),
        card_selection_task=torch.tensor([s["card_selection_task"] for s in states], dtype=torch.long),
        card_ids=long(cards, "card_id"),
        card_zones=long(cards, "zone"),
        card_types=long(cards, "card_type"),
        target_types=long(cards, "target_type"),
        card_numeric=floats(cards, "numeric", 14),
        monster_ids=long(monsters, "monster_id"),
        move_ids=long(monsters, "move_id"),
        previous_move_ids=long(monsters, "previous_move_id"),
        monster_numeric=floats(monsters, "numeric", 9),
        monster_status=floats(monsters, "status", 15),
        interaction_cards=torch.tensor([x["card_index"] + c for x, c, _ in interactions], dtype=torch.long),
        interaction_monsters=torch.tensor([x["monster_index"] + m for x, _, m in interactions], dtype=torch.long),
        interaction_numeric=floats([x for x, _, _ in interactions], "numeric", 6),
        potion_ids=long(potions, "potion_id"),
        potion_numeric=floats(potions, "numeric", 18),
        relic_ids=long(relics, "relic_id"),
        relic_numeric=floats(relics, "numeric", 3),
        card_state_indices=torch.tensor(owners["card"], dtype=torch.long),
        monster_state_indices=torch.tensor(owners["monster"], dtype=torch.long),
        interaction_state_indices=torch.tensor(owners["interaction"], dtype=torch.long),
        potion_state_indices=torch.tensor(owners["potion"], dtype=torch.long),
        relic_state_indices=torch.tensor(owners["relic"], dtype=torch.long),
    )
