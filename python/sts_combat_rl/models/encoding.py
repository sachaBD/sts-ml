import torch


def value_tensors(state: dict) -> dict[str, torch.Tensor]:
    version = state.get("encoding_version") if "encoding_version" in state else state.get("version")
    if version != 2:
        raise ValueError(f"unsupported encoding version: {version!r} (expected 2)")
    cards, monsters, interactions = state["cards"], state["monsters"], state["card_monster_interactions"]
    return {
        "global_numeric": torch.tensor(state["global_numeric"], dtype=torch.float32),
        "input_state": torch.tensor(state["input_state"], dtype=torch.long),
        "card_selection_task": torch.tensor(state["card_selection_task"], dtype=torch.long),
        "card_ids": torch.tensor([x["card_id"] for x in cards], dtype=torch.long),
        "card_zones": torch.tensor([x["zone"] for x in cards], dtype=torch.long),
        "card_types": torch.tensor([x["card_type"] for x in cards], dtype=torch.long),
        "target_types": torch.tensor([x["target_type"] for x in cards], dtype=torch.long),
        "card_numeric": torch.tensor([x["numeric"] for x in cards], dtype=torch.float32),
        "monster_ids": torch.tensor([x["monster_id"] for x in monsters], dtype=torch.long),
        "move_ids": torch.tensor([x["move_id"] for x in monsters], dtype=torch.long),
        "monster_numeric": torch.tensor([x["numeric"] for x in monsters], dtype=torch.float32),
        "interaction_cards": torch.tensor([x["card_index"] for x in interactions], dtype=torch.long),
        "interaction_monsters": torch.tensor([x["monster_index"] for x in interactions], dtype=torch.long),
        "interaction_numeric": torch.tensor([x["numeric"] for x in interactions], dtype=torch.float32).reshape(-1, 6),
    }
