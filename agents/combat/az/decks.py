"""Start banks: random A20 Ironclad decks over jaxsim's supported cards, for training before human decks."""
from dataclasses import dataclass

import numpy as np

import jaxsim

STARTER = ["STRIKE_RED"] * 5 + ["DEFEND_RED"] * 4 + ["BASH", "ASCENDERS_BANE"]
NOT_DRAFTABLE = {"WOUND", "ASCENDERS_BANE"}


@dataclass(frozen=True)
class DeckConfig:
    count: int            # decks in the bank
    extra_min: int        # cards added to the starter deck
    extra_max: int
    upgrade_prob: float
    hp_min: int           # starting hp (max hp = max(hp, 80))
    hp_max: int
    seed: int


def random_deck(rng: np.random.Generator, cfg: DeckConfig):
    pool = sorted(set(jaxsim.supported_cards()) - NOT_DRAFTABLE)
    deck = list(STARTER)
    for _ in range(rng.integers(cfg.extra_min, cfg.extra_max + 1)):
        deck.append((pool[rng.integers(len(pool))], int(rng.random() < cfg.upgrade_prob)))
    return deck


def make_bank(cfg: DeckConfig, spec: jaxsim.EnvSpec):
    rng = np.random.default_rng(cfg.seed)
    starts = []
    for _ in range(cfg.count):
        hp = int(rng.integers(cfg.hp_min, cfg.hp_max + 1))
        starts.append(jaxsim.make_start(random_deck(rng, cfg), hp=hp, max_hp=max(hp, 80), spec=spec))
    return jaxsim.stack_starts(starts)
