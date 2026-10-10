"""Pure, deterministic family-balanced replay planning for the Champ corpus pilot.

Inputs are lightweight fight IDs and encoded-row counts.  Loading shards and tensors is
intentionally trainer work, not part of this protocol planner.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from typing import Iterable

import numpy as np

SLOTS_PER_EPOCH = 3000
STATES_PER_SLOT = 64


@dataclass(frozen=True)
class Fight:
    fight_id: str
    family_id: str
    source: str                 # "teacher" or "learner"
    collection_round: int       # 0 for eligible legacy data; 1..4 for pilot collection
    row_count: int


@dataclass(frozen=True)
class Slot:
    family_id: str
    source: str                 # teacher, current_learner, historical_learner
    fight_id: str
    state_rows: tuple[int, ...]


@dataclass(frozen=True)
class EpochPlan:
    round_number: int
    seed: int
    slots: tuple[Slot, ...]

    @property
    def state_count(self):
        return len(self.slots) * STATES_PER_SLOT

    def counts(self):
        return Counter((x.family_id, x.source) for x in self.slots)

    def mixed_state_draws(self):
        """State-level shuffled draws for training; consumers must not batch raw slots."""
        draws = [(slot.family_id, slot.source, slot.fight_id, row) for slot in self.slots for row in slot.state_rows]
        order = _rng(self.seed, self.round_number, "state-mix").permutation(len(draws))
        return tuple(draws[i] for i in order)


def _rng(seed: int, *parts: object):
    payload = json.dumps([seed, *parts], separators=(",", ":")).encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(payload).digest()[:16], "little"))


def _families(families: Iterable[object], round_number: int):
    """Accept immutable manifest Family objects without importing the app layer."""
    selected = []
    for family in families:
        role, wave = family.role, family.wave
        if role == "original": selected.append(family.family_id)
        elif role == "pilot" and wave <= round_number: selected.append(family.family_id)
        elif role in {"development", "final"}:
            continue
        elif role == "pilot":
            continue
        else:
            raise ValueError(f"unknown family role {role!r}")
    expected = 10 + round_number * 10
    if len(selected) != expected or len(set(selected)) != len(selected):
        raise ValueError(f"round {round_number} requires {expected} unique admitted families")
    return tuple(sorted(selected))


def plan_epoch(families: Iterable[object], fights: Iterable[Fight], round_number: int, seed: int) -> EpochPlan:
    """Draw exactly 3,000 family-first slots and 64 uniform rows per slot.

    Teacher pools use all admitted teacher fights.  Current/history learner pools are
    separated by collection round; future fights, original fresh learner data, held-out
    fights, duplicate trajectory IDs, and missing required pools fail closed.
    """
    if type(round_number) is not int or round_number not in range(1, 5): raise ValueError("round_number must be 1..4")
    if type(seed) is not int: raise ValueError("seed must be an integer")
    families = tuple(families)
    family_ids = _families(families, round_number)
    roles = {f.family_id: f.role for f in families}
    waves = {f.family_id: f.wave for f in families}
    grouped = {(f, s): [] for f in family_ids for s in ("teacher", "current_learner", "historical_learner")}
    seen = set()
    for fight in fights:
        if (not isinstance(fight, Fight) or not isinstance(fight.fight_id, str) or not fight.fight_id
                or type(fight.row_count) is not int or fight.row_count < 1
                or type(fight.collection_round) is not int):
            raise ValueError("fights require nonempty string ID and positive integer row_count/round")
        if fight.fight_id in seen: raise ValueError(f"duplicate trajectory ID: {fight.fight_id}")
        seen.add(fight.fight_id)
        if fight.family_id not in roles: raise ValueError(f"fight has unknown family {fight.family_id}")
        if fight.family_id not in family_ids:
            raise ValueError(f"held-out or not-yet-admitted family sampled: {fight.family_id}")
        if fight.collection_round > round_number or fight.collection_round < 0:
            raise ValueError(f"future/invalid collection round for {fight.fight_id}")
        if fight.source not in {"teacher", "learner"}: raise ValueError(f"unknown source {fight.source}")
        if roles[fight.family_id] == "original" and fight.collection_round != 0:
            raise ValueError(f"original replay-only family has fresh fight: {fight.family_id}")
        if roles[fight.family_id] == "pilot" and fight.collection_round < waves[fight.family_id]:
            raise ValueError(f"fight predates pilot family admission: {fight.fight_id}")
        if fight.source == "teacher": grouped[fight.family_id, "teacher"].append(fight)
        elif fight.collection_round == round_number and roles[fight.family_id] != "original":
            grouped[fight.family_id, "current_learner"].append(fight)
        else: grouped[fight.family_id, "historical_learner"].append(fight)
    for pool in grouped.values():
        pool.sort(key=lambda fight: fight.fight_id)
    slots_per_family, rem = divmod(SLOTS_PER_EPOCH, len(family_ids))
    if rem: raise ValueError("admitted family count must divide 3000")
    slots = []
    for family_id in family_ids:
        pools = {key: grouped[family_id, key] for key in ("teacher", "current_learner", "historical_learner")}
        if not pools["teacher"]: raise ValueError(f"missing teacher pool: {family_id}")
        current, history = pools["current_learner"], pools["historical_learner"]
        if not current and not history: raise ValueError(f"missing both learner pools: {family_id}")
        source_counts = {"teacher": slots_per_family // 5}
        learners = slots_per_family - source_counts["teacher"]
        if current and history:
            source_counts.update(current_learner=learners // 2, historical_learner=learners // 2)
        elif current: source_counts.update(current_learner=learners, historical_learner=0)
        else: source_counts.update(current_learner=0, historical_learner=learners)
        kinds = [kind for kind, count in source_counts.items() for _ in range(count)]
        kinds = _rng(seed, round_number, family_id, "sources").permutation(kinds).tolist()
        for slot_index, kind in enumerate(kinds):
            pool = pools[kind]
            draw = _rng(seed, round_number, family_id, slot_index, kind)
            fight = pool[int(draw.integers(len(pool)))]
            rows = tuple(int(x) for x in draw.integers(fight.row_count, size=STATES_PER_SLOT))
            slots.append(Slot(family_id, kind, fight.fight_id, rows))
    slots = tuple(_rng(seed, round_number, "mix").permutation(slots).tolist())
    result = EpochPlan(round_number, seed, slots)
    if len(result.slots) != SLOTS_PER_EPOCH or result.state_count != 192000: raise AssertionError("replay size invariant")
    return result

