"""Start banks: decks from a JSON file (training and evaluation), or seeded random decks (tests, benchmark).

Deck file (JSON):
  {"decks": [{"name": "my deck", "hp": 62, "max_hp": 80,
              "cards": ["BASH", "STRIKE_RED+", ["SEARING_BLOW", 3], ["RITUAL_DAGGER", 0, 25]],
              "relics": ["BURNING_BLOOD", ["PEN_NIB", 4]],
              "potions": ["FIRE_POTION"], "potion_capacity": 2}]}
Cards: jaxsim names; "NAME+" = upgraded; [name, upgrades] or [name, upgrades, misc]. Relics: name or
[name, run counter]. max_hp defaults to hp, potions to none, potion_capacity to 2 (4 with Potion Belt).
Decks jaxsim can't play are rejected with the reason (load_file's report), not silently dropped.
"""
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import jaxsim

STARTER = ["STRIKE_RED"] * 5 + ["DEFEND_RED"] * 4 + ["BASH", "ASCENDERS_BANE"]
NOT_DRAFTABLE = {"WOUND", "DAZED", "BURN", "ASCENDERS_BANE"}


def _card(c):
    if isinstance(c, str):
        return (c[:-1], 1) if c.endswith("+") else (c, 0)
    return tuple(c)


def _relic(r):
    return r if isinstance(r, str) else (r[0], int(r[1]))


def start_of(deck: dict, spec: jaxsim.EnvSpec) -> jaxsim.StartSpec:
    return jaxsim.make_start([_card(c) for c in deck["cards"]], hp=int(deck["hp"]),
                             max_hp=int(deck.get("max_hp", deck["hp"])),
                             relics=[_relic(r) for r in deck.get("relics", [])], potions=deck.get("potions", []),
                             potion_capacity=deck.get("potion_capacity"), spec=spec)


@dataclass
class Loaded:
    names: list[str]              # accepted decks, in bank order
    bank: jaxsim.StartSpec | None  # stacked (None if nothing was accepted)
    rejected: list[tuple[str, str]]  # (name, reason)

    def report(self) -> str:
        lines = [f"{len(self.names)} decks accepted, {len(self.rejected)} rejected"]
        reasons = Counter(r for _, r in self.rejected)
        lines += [f"  {n} x {r}" for r, n in reasons.most_common(20)]
        return "\n".join(lines)


def load_file(path: str | Path, spec: jaxsim.EnvSpec) -> Loaded:
    data = json.loads(Path(path).read_text())
    names, starts, rejected = [], [], []
    for i, deck in enumerate(data["decks"]):
        name = deck.get("name", f"deck {i}")
        try:
            starts.append(start_of(deck, spec))
            names.append(name)
        except (ValueError, KeyError, TypeError) as e:
            rejected.append((name, str(e)))
    return Loaded(names, jaxsim.stack_starts(starts) if starts else None, rejected)


@dataclass(frozen=True)
class RandomDecks:
    count: int            # decks in the bank
    extra_min: int        # cards added to the starter deck
    extra_max: int
    upgrade_prob: float
    hp_min: int           # starting hp (max hp = max(hp, 80))
    hp_max: int
    seed: int


def random_bank(cfg: RandomDecks, spec: jaxsim.EnvSpec) -> jaxsim.StartSpec:
    rng = np.random.default_rng(cfg.seed)
    pool = sorted(set(jaxsim.supported_cards()) - NOT_DRAFTABLE)
    starts = []
    for _ in range(cfg.count):
        deck = list(STARTER)
        for _ in range(rng.integers(cfg.extra_min, cfg.extra_max + 1)):
            deck.append((pool[rng.integers(len(pool))], int(rng.random() < cfg.upgrade_prob)))
        hp = int(rng.integers(cfg.hp_min, cfg.hp_max + 1))
        starts.append(jaxsim.make_start(deck, hp=hp, max_hp=max(hp, 80), spec=spec))
    return jaxsim.stack_starts(starts)
