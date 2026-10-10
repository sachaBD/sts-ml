"""Input-only Champ corpus manifest validation and collection planning.

This checks declarations, not their completeness: a separately reviewed audit must
establish that provenance aliases and historical exposure are complete.  Manifest SHA
is an identity, not verification of the referenced artifact bytes.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "champ-corpus-pilot-v1"
TEN_DECK_SOURCE_RUN = "champ-ten-rollout-v1"
TEN_DECK_LOCATOR = "runs/schema=combat_v4/date=2026-10-09/id=champ-ten-rollout-v1/out"
LEGACY_ELIGIBLE = frozenset({"bootstrap", *(f"learner_update{i:03d}" for i in range(1, 7))})
ROLES = frozenset({"original", "pilot", "development", "final"})


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


@dataclass(frozen=True)
class Family:
    family_id: str; role: str; wave: int | None; provenance: tuple[str, ...]


@dataclass(frozen=True)
class LegacySource:
    source_id: str; family_id: str; category: str; used_for_training: bool


@dataclass(frozen=True)
class CorpusManifest:
    families: tuple[Family, ...]
    historical_exposure: tuple[LegacySource, ...]
    replay_source_ids: tuple[str, ...]
    legacy_run: dict[str, Any]
    sha256: str

    @property
    def by_id(self): return {f.family_id: f for f in self.families}


def _strings(value, name):
    if not isinstance(value, list) or not value or any(not isinstance(x, str) or not x for x in value):
        raise ValueError(f"{name} must be a nonempty list of strings")
    return tuple(value)


def parse_manifest(value: dict[str, Any]) -> CorpusManifest:
    """Validate an explicit audit/replay manifest without inspecting the filesystem."""
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError(f"manifest schema must be {SCHEMA!r}")
    allowed = {"schema", "legacy_run", "families", "historical_exposure", "replay_source_ids"}
    if set(value) - allowed: raise ValueError(f"unknown manifest keys: {sorted(set(value) - allowed)}")
    run = value.get("legacy_run")
    if not isinstance(run, dict) or run.get("run_id") != TEN_DECK_SOURCE_RUN or not isinstance(run.get("locator"), str):
        raise ValueError(f"legacy_run must declare ten-deck run {TEN_DECK_SOURCE_RUN!r} and locator")
    rows = value.get("families")
    if not isinstance(rows, list): raise ValueError("families must be a list")
    families, aliases = [], {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"family_id", "role", "wave", "provenance"}:
            raise ValueError("each family requires exactly family_id, role, wave, provenance")
        fid, role, wave = row["family_id"], row["role"], row["wave"]
        if not isinstance(fid, str) or not fid or not isinstance(role, str) or role not in ROLES: raise ValueError("family_id and supported role required")
        if role == "pilot":
            if type(wave) is not int or wave not in range(1, 5): raise ValueError(f"pilot family {fid}: wave must be 1..4")
        elif wave is not None: raise ValueError(f"non-pilot family {fid}: wave must be null")
        provenance = _strings(row["provenance"], f"{fid}.provenance")
        for key in provenance:
            if key in aliases: raise ValueError(f"provenance {key!r} belongs to both {aliases[key]} and {fid}")
            aliases[key] = fid
        families.append(Family(fid, role, wave, provenance))
    if len({f.family_id for f in families}) != len(families): raise ValueError("duplicate family_id")
    if sum(f.role == "original" for f in families) != 10: raise ValueError("exactly ten original replay-only families required")
    pilots = [f for f in families if f.role == "pilot"]
    if len(pilots) != 40 or any(sum(f.wave == w for f in pilots) != 10 for w in range(1, 5)):
        raise ValueError("exactly ten pilot families in each wave 1..4 required")
    raw = value.get("historical_exposure", [])
    if not isinstance(raw, list): raise ValueError("historical_exposure must be a list")
    family_ids = {f.family_id for f in families}; exposure = []
    for row in raw:
        if not isinstance(row, dict) or set(row) != {"source_id", "family_id", "category", "used_for_training"}:
            raise ValueError("each exposure requires source_id, family_id, category, used_for_training")
        if not isinstance(row["source_id"], str) or not row["source_id"] or row["family_id"] not in family_ids:
            raise ValueError("exposure has unknown family or invalid source_id")
        if not isinstance(row["category"], str) or not isinstance(row["used_for_training"], bool):
            raise ValueError("exposure category and used_for_training must be string/bool")
        exposure.append(LegacySource(**row))
    by_source = {s.source_id: s for s in exposure}
    if len(by_source) != len(exposure): raise ValueError("duplicate exposure source_id")
    replay = _strings(value.get("replay_source_ids"), "replay_source_ids")
    if len(set(replay)) != len(replay): raise ValueError("duplicate replay source ID")
    by_family = {f.family_id: f for f in families}
    for source in exposure:
        if source.used_for_training and by_family[source.family_id].role != "original":
            raise ValueError(f"non-original family has historical training exposure: {source.family_id}")
    for source_id in replay:
        source = by_source.get(source_id)
        if source is None: raise ValueError(f"replay source not in historical exposure: {source_id}")
        if not source.used_for_training or source.category not in LEGACY_ELIGIBLE:
            raise ValueError(f"ineligible legacy replay source {source_id}: {source.category}")
        if by_family[source.family_id].role != "original":
            raise ValueError(f"legacy replay source must belong to original family: {source.family_id}")
    return CorpusManifest(tuple(families), tuple(exposure), replay, run, hashlib.sha256(_canonical(value)).hexdigest())


def validate_production_freeze(manifest: CorpusManifest) -> None:
    """Extra requirements for the real frozen manifest, intentionally not unit fixtures."""
    if sum(f.role == "development" for f in manifest.families) != 20: raise ValueError("production freeze requires 20 development families")
    if sum(f.role == "final" for f in manifest.families) != 100: raise ValueError("production freeze requires 100 final families")
    if manifest.legacy_run.get("locator") != TEN_DECK_LOCATOR: raise ValueError("noncanonical legacy run locator")
    hashes = manifest.legacy_run.get("artifact_sha256")
    if not isinstance(hashes, dict) or not hashes or any(not isinstance(k, str) or not isinstance(v, str) or len(v) != 64 for k, v in hashes.items()):
        raise ValueError("production freeze requires declared artifact SHA-256 hashes")


def load_manifest(path: Path) -> CorpusManifest: return parse_manifest(json.loads(Path(path).read_text()))


def eligible_legacy_sources(manifest: CorpusManifest) -> tuple[LegacySource, ...]:
    selected = set(manifest.replay_source_ids)
    return tuple(s for s in manifest.historical_exposure if s.source_id in selected)


def collection_plan(manifest: CorpusManifest, round_number: int, seed: int = 0) -> tuple[dict[str, Any], ...]:
    if type(round_number) is not int or round_number not in range(1, 5): raise ValueError("round_number must be 1..4")
    if type(seed) is not int: raise ValueError("seed must be an integer")
    pilot = [f for f in manifest.families if f.role == "pilot"]
    current = sorted((f for f in pilot if f.wave == round_number), key=lambda f: f.family_id)
    earlier = sorted((f for f in pilot if f.wave < round_number), key=lambda f: f.family_id)
    jobs = [dict(family_id=f.family_id, source="teacher", games=20) for f in current]
    jobs += [dict(family_id=f.family_id, source="learner", games=80) for f in current]
    if earlier:
        order = sorted(earlier, key=lambda f: hashlib.sha256(f"{seed}:revisit:{round_number}:{f.family_id}".encode()).digest())
        q, r = divmod(500, len(order)); jobs += [dict(family_id=f.family_id, source="revisit_learner", games=q + (i < r)) for i, f in enumerate(order)]
    assert sum(j["games"] for j in jobs) == (1000 if round_number == 1 else 1500)
    return tuple(jobs)
