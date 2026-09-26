#!/usr/bin/env python3
"""Fixed train / dev / confirm split for the act-1 general value net. Writes split.md tables + dev_fights.csv.

Run: PYTHONPATH=python .venv/bin/python experiments/act-1-combat/2026-09-26-mirror-mcts/split.py
"""
from pathlib import Path

from sts_combat_rl import query

HERE = Path(__file__).parent
DEV_CAP = 250  # dev fights per non-boss encounter (boss: all 999, reusing the slime-v8 dev set)

# Same rule as rundecks/slime-v8/SPLIT.md, extended from Slime Boss to every act-1 fight of those runs.
PART = """case when id = 'act1-a20-8' and run_seed % 10 in (0, 1) then 'dev'
               when id = 'act1-a20-8' and run_seed % 10 in (2, 3) then 'confirm'
               else 'train' end"""
BOOTSTRAP = "id like 'act1-a20%' and source_episode_id is null"

db = query.connect()
db.sql("set enable_progress_bar = false")
db.sql(f"""create temp table fights as
    select any_value({PART}) part, run_seed, episode_id, any_value(encounter) encounter, any_value(category) category,
           count(*) filter (where row_kind = 'decision') decisions, count(*) rows_,
           any_value(won) won, any_value(starting_hp) - any_value(final_hp) hp_lost,
           any_value(terminal_value) tv, any_value(starting_max_hp) max_hp,
           sum(simulations_used) filter (where row_kind = 'decision') sims
    from combat_v3 where {BOOTSTRAP} group by run_seed, episode_id""")
# dev eval subset: first DEV_CAP dev fights per encounter by episode_id (run seeds are arbitrary, so this is an
# unbiased, version-stable pick); every Slime Boss dev fight.
db.sql(f"""create temp table dev_eval as select * from (
    select *, row_number() over (partition by category, encounter order by episode_id) k from fights where part = 'dev')
    where encounter = 'slime_boss' or k <= {DEV_CAP}""")
db.sql(f"copy (select episode_id, run_seed, encounter, category from dev_eval order by encounter, episode_id) "
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
out.append("## Teacher (stored bootstrap, 15k sims, one random move) on train fights\n\n" + md("""
    select category, encounter, count(*) fights, round(100 * avg(won::int), 1) "win %",
        round(avg(hp_lost) filter (where won), 1) "HP lost (wins)", round(stddev(hp_lost) filter (where won), 1) "sd"
    from fights where part = 'train' group by all order by category, encounter"""))
out.append("## Dev eval cost proxy (stored teacher simulations)\n\n" + md("""
    select encounter = 'slime_boss' boss, count(*) fights, sum(decisions) decisions, round(sum(sims) / 1e6) "M sims"
    from dev_eval group by all order by boss"""))
(HERE / "split_tables.md").write_text("\n".join(out))
print("\n".join(out))
