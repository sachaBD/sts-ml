"""combat_v3 rows by SQL: the one way to load them (DuckDB over the runs/ layout, runs/README.md).

Views:
  combat_v3  every combat_v3 row, plus run_id, schema, date, id (from its run directory)
  runs       every run.json: run_id, schema, status, inputs, summary (JSON)

A query selects whole rows, e.g.
  select * from combat_v3 where id like 'act1-a20%' and encounter = 'slime_boss'
and callers ask for the columns they need. Oracle rows (played with perfect foresight: an upper bound, not
fair play; runs/README.md `oracle`) raise unless oracle=True.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from .run import RUNS


def connect(root: Path = RUNS) -> duckdb.DuckDBPyConnection:
    """An in-memory DuckDB with the combat_v3 and runs views over `root`."""
    db = duckdb.connect()
    db.execute(f"""
        create view combat_v3 as
        select *, concat_ws('/', schema, date, id) as run_id
        from read_parquet('{root}/schema=combat_v3/*/*/out/*.parquet',
                          hive_partitioning = true, hive_types_autocast = false, union_by_name = true)""")
    db.execute(f"""
        create view runs as
        select * from read_json('{root}/*/*/*/run.json', columns = {{
            run_id: 'VARCHAR', schema: 'VARCHAR', status: 'VARCHAR', inputs: 'VARCHAR[]', summary: 'JSON'}})""")
    return db


def sql_list(values) -> str:
    """Values as a SQL list literal, e.g. for `run_id in {sql_list(ids)}`."""
    quote = lambda v: str(v) if isinstance(v, int) else "'" + str(v).replace("'", "''") + "'"
    return "(" + ", ".join(map(quote, values)) + ")" if values else "(null)"


def rows(sql: str, columns=None, where: str | None = None, oracle: bool = False,
         root: Path = RUNS) -> list[dict[str, Any]]:
    """The rows of `sql` that also match `where`: only `columns` (plus run_id) if given, else every column."""
    source = f"(select * from ({sql}) where {where})" if where else f"({sql})"
    wanted = "*" if columns is None else ", ".join(dict.fromkeys([*columns, "run_id", "oracle"]))
    table = connect(root).sql(f"select {wanted} from {source}").fetch_arrow_table()
    if not oracle and table["oracle"].to_pylist().count(True):
        raise ValueError(f"the query returns oracle rows (perfect-foresight teacher); allow them explicitly: {sql}")
    if columns is not None and "oracle" not in columns:
        table = table.drop_columns(["oracle"])
    return table.to_pylist()


def run_ids(sql: str, root: Path = RUNS) -> list[str]:
    """The runs the rows of `sql` come from (lineage)."""
    return [r for (r,) in connect(root).sql(f"select distinct run_id from ({sql}) order by 1").fetchall()]
