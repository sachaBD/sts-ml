"""Champ boss fights from the act3-heart combat_v4 recordings -> champ_fights.parquet (one row per fight).
Deck/relics/potions decoded to sts_lightspeed enum names. Usage: .venv/bin/python experiments/act2/champ/load.py"""
import re
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
SIM = ROOT.parent / "sts_lightspeed" / "include" / "constants"
OUT = Path(__file__).parent / "champ_fights.parquet"


def enum(header, name):
    text = (SIM / header).read_text()
    body = re.search(name + r"\s*\[\]\s*(?:=\s*)?\{(.*?)\}", text, re.S).group(1)
    return re.findall(r'"([^"]*)"', body)


CARDS, ENC = enum("Cards.h", "cardEnumStrings"), enum("MonsterEncounters.h", "monsterEncounterEnumNames")
RELICS, POTIONS = enum("Relics.h", "relicEnumNames"), enum("Potions.h", "potionEnumNames")
CHAMP = ENC.index("CHAMP")

files = sorted(ROOT.glob("runs/schema=overworld_v1/*/id=ah-*/out/combat/fights-*.parquet"))
df = duckdb.sql(f"""select regexp_extract(filename, 'id=([^/]+)/out', 1) as run, fight_id, agent, won, final_hp,
                    start.seed as seed, start.act as act, start.floor as floor, start.hp as hp, start.max_hp as max_hp,
                    start.deck as deck, start.relics as relics, start.potions as potions, len(actions) as decisions
                    from read_parquet({[str(f) for f in files]}, filename = true)
                    where start.encounter = {CHAMP}""").to_arrow_table().to_pylist()
for r in df:
    r["deck"] = sorted(CARDS[c["id"]].lower() + ("+" if c["upgraded"] else "") for c in r["deck"])
    r["relics"] = sorted(RELICS[x["id"]].lower() for x in r["relics"])
    r["potions"] = [POTIONS[x].lower() for x in r["potions"] if POTIONS[x] not in ("INVALID", "EMPTY_POTION_SLOT")]
pq.write_table(pa.Table.from_pylist(df), OUT)
print(len(files), "files;", len(df), "champ fights; win", round(sum(r["won"] for r in df) / len(df), 3))
for run in sorted({r["run"] for r in df}):
    w = [r["won"] for r in df if r["run"] == run]
    print(f"  {run:22s} n={len(w):4d} win={sum(w) / len(w):.3f}")
