import duckdb, sys
out, enc = sys.argv[1], int(sys.argv[2])
g = 'runs/schema=overworld_v1/date=2026-10-0[23]/id=ah-*/out/combat/'
c = duckdb.connect()
c.execute(f"""copy (select * exclude (date, id, schema) from read_parquet('{g}fights-*.parquet', hive_partitioning=true) where start.encounter = {enc})
              to '{out}/fights.parquet' (format parquet, compression zstd)""")
c.execute(f"""copy (select s.* exclude (date, id, schema) from read_parquet('{g}search-*.parquet', hive_partitioning=true) s
              where s.fight_id in (select fight_id from '{out}/fights.parquet'))
              to '{out}/search.parquet' (format parquet, compression zstd)""")
