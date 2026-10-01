#!/usr/bin/env python3
"""Fixed train / dev / confirm split of the all-bosses act-1 dataset. Writes results/split_tables.md + dev_fights.csv.

Run: PYTHONPATH=. .venv/bin/python experiments/act-1-combat/2026-09-26-all-bosses/split.py

Split by whole runs (run_seed; consecutive seeds, so % 10 buckets are ~1,210 runs each):
  dev = run_seed % 10 in (0, 1), confirm = (2, 3), train = 4..9.
Dev eval subset (1,000 fights, = the configs' dev query): 200 per boss, 100 per elite, 100 other (hallway + events,
natural mix); random by hash(episode_id).
"""
from pathlib import Path

from runs import query

HERE = Path(__file__).parent
DATA = "act1-all-bosses-a20-scaled-search"

PART = """case when run_seed % 10 in (0, 1) then 'dev' when run_seed % 10 in (2, 3) then 'confirm' else 'train' end"""

db = query.connect()
db.sql("set enable_progress_bar = false")
db.sql(f"""create temp table fights as
    select any_value({PART}) part, run_seed, episode_id, any_value(encounter) encounter, any_value(category) category,
           count(*) filter (where row_kind = 'decision') decisions, count(*) rows_,
           any_value(won) won, any_value(starting_hp) - any_value(final_hp) hp_lost,
           any_value(terminal_value) tv, sum(simulations_used) filter (where row_kind = 'decision') sims
    from combat_v3 where id = '{DATA}' and source_episode_id is null group by run_seed, episode_id""")
db.sql(f"""create temp table dev_eval as select * from (
    select *, row_number() over (partition by case when category in ('boss', 'elite') then encounter else 'other' end
                                 order by hash(episode_id)) k from fights where part = 'dev')
    where (category = 'boss' and k <= 200) or (category = 'elite' and k <= 100)
       or (category not in ('boss', 'elite') and k <= 100)""")
db.sql(f"copy (select episode_id, run_seed, encounter, category from dev_eval order by category, encounter, episode_id) "
       f"to '{HERE / 'dev_fights.csv'}' (header)")


def md(sql: str) -> str:
    rel = db.sql(sql)
    rows = rel.fetchall()
    head = "| " + " | ".join(rel.columns) + " |\n|" + "---|" * len(rel.columns) + "\n"
    return head + "".join("| " + " | ".join("" if v is None else str(v) for v in r) + " |\n" for r in rows)


out = []
out.append("## Fights per encounter and part\n\n" + md("""
    select category, encounter,
        count(*) filter (where part = 'train') train, count(*) filter (where part = 'dev') dev,
        (select count(*) from dev_eval e where e.encounter = f.encounter and e.category = f.category) dev_eval,
        count(*) filter (where part = 'confirm') confirm,
        round(100.0 * sum(rows_) filter (where part = 'train') / (select sum(rows_) from fights where part = 'train'), 1) "train rows %"
    from fights f group by all order by category, encounter"""))
out.append("## Totals\n\n" + md("""
    select part, count(distinct run_seed) runs, count(*) fights, sum(decisions) decision_rows, sum(rows_) all_rows
    from fights group by part order by part"""))
out.append("## Stored teacher (scaled search, one random move) by part: win % / mean terminal value\n\n" + md("""
    select category, encounter,
        round(100 * avg(won::int) filter (where part = 'train'), 1) "train win %",
        round(100 * avg(won::int) filter (where part = 'dev'), 1) "dev win %",
        round(100 * avg(won::int) filter (where part = 'confirm'), 1) "confirm win %",
        round(avg(hp_lost) filter (where won and part = 'train'), 1) "train HP lost (wins)"
    from fights group by all order by category, encounter"""))
out.append("## Dev eval subset size\n\n" + md("""
    select category = 'boss' boss, count(*) fights, sum(decisions) decisions from dev_eval group by all order by boss"""))
(HERE / "results" / "split_tables.md").write_text("\n".join(out))
print("\n".join(out))
