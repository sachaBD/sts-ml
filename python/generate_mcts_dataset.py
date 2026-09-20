#!/usr/bin/env python3
import argparse
import datetime
import hashlib
import math
import subprocess
import time
from pathlib import Path

import msgpack
import pyarrow as pa
import pyarrow.parquet as pq

F32 = pa.float32()
SCHEMA = pa.schema([
    ("encoding_version", pa.int32()), ("episode_id", pa.int64()), ("seed", pa.uint64()), ("decision_index", pa.int32()),
    ("global_numeric", pa.list_(F32, 22)), ("input_state", pa.int16()), ("card_selection_task", pa.int16()),
    ("cards", pa.list_(pa.struct([( "card_id", pa.int16()), ("zone", pa.int8()), ("card_type", pa.int8()), ("target_type", pa.int8()), ("numeric", pa.list_(F32, 14))]))),
    ("monsters", pa.list_(pa.struct([("monster_id", pa.int16()), ("move_id", pa.int16()), ("numeric", pa.list_(F32, 9))]))),
    ("card_monster_interactions", pa.list_(pa.struct([("card_index", pa.int16()), ("monster_index", pa.int8()), ("numeric", pa.list_(F32, 6))]))),
    ("mcts_value", F32), ("root_visits", pa.int64()), ("chosen_action", pa.int64()), ("terminal_outcome", pa.int8()),
    ("final_player_hp", pa.int16()), ("final_player_max_hp", pa.int16()), ("terminal_value", F32),
])

def git_revision(path):
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except subprocess.CalledProcessError:
        return "unknown"

def git_dirty(path):
    return subprocess.run(["git", "-C", str(path), "diff", "--quiet"]).returncode != 0

def records(args):
    command = [args.generator, str(args.seed_start), str(args.seed_count), str(args.simulations), str(args.rollout_limit), str(args.exploration)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=None)
    unpacker = msgpack.Unpacker(process.stdout, raw=False)
    rows = list(unpacker)
    if process.wait() != 0:
        raise RuntimeError("C++ record generator failed")
    return rows

def validate(rows, args):
    expected = {(episode, args.seed_start + episode) for episode in range(args.seed_count)}
    groups = {}
    for row in rows:
        if row["encoding_version"] != 2 or len(row["global_numeric"]) != 22:
            raise ValueError("invalid v1 encoding")
        for token, width in ((row["cards"], 14), (row["monsters"], 9), (row["card_monster_interactions"], 6)):
            if any(len(item["numeric"]) != width for item in token):
                raise ValueError("invalid nested numeric width")
        if row["root_visits"] != args.simulations or not all(math.isfinite(row[x]) and -1 <= row[x] <= 1 for x in ("mcts_value", "terminal_value")):
            raise ValueError("invalid MCTS result")
        groups.setdefault((row["episode_id"], row["seed"]), []).append(row)
    if set(groups) != expected:
        raise ValueError("episode IDs or seeds differ from request")
    for episode_rows in groups.values():
        episode_rows.sort(key=lambda row: row["decision_index"])
        if [row["decision_index"] for row in episode_rows] != list(range(len(episode_rows))):
            raise ValueError("non-contiguous decision indices")
        terminal = {(row["terminal_outcome"], row["final_player_hp"], row["final_player_max_hp"], row["terminal_value"]) for row in episode_rows}
        if len(terminal) != 1:
            raise ValueError("inconsistent terminal fields")
    return groups

def write_manifest(path, shard, args, groups, rows):
    wins = sum(next(iter(group))["terminal_outcome"] == 1 for group in groups.values())
    digest = hashlib.sha256(shard.read_bytes()).hexdigest()
    root = Path(__file__).resolve().parents[1]
    text = f'''dataset_id = "a1-slime-boss-mcts-value-v2"
created_utc = "{datetime.datetime.now(datetime.UTC).isoformat()}"
schema_version = 2
scenario = "A1 Slime Boss fixed enhanced Ironclad deck"
project_git_revision = "{git_revision(root)}"
project_git_dirty = {str(git_dirty(root)).lower()}
simulator_git_revision = "{git_revision(root / 'sts_lightspeed')}"
seed_start = {args.seed_start}
seed_count = {args.seed_count}
simulations = {args.simulations}
rollout_limit = {args.rollout_limit}
exploration = {args.exploration}
episodes = {len(groups)}
decisions = {len(rows)}
wins = {wins}
shard = "{shard.name}"
sha256 = "{digest}"
'''
    (path / "manifest.toml").write_text(text)
    return wins

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--simulations", type=int, default=2000)
    parser.add_argument("--rollout-limit", type=int, default=512)
    parser.add_argument("--exploration", type=float, default=1.4142135623730951)
    parser.add_argument("--generator", default="build/generate_mcts_records")
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic(); rows = records(args); groups = validate(rows, args)
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    if any(column.null_count for column in table.columns): raise ValueError("Arrow null values")
    shard = args.output_dir / "mcts_slime_v2.parquet"; pq.write_table(table, shard, compression="zstd")
    wins = write_manifest(args.output_dir, shard, args, groups, rows)
    print(f"rows={len(rows)} episodes={len(groups)} wins={wins} runtime={time.monotonic()-started:.1f}s")
if __name__ == "__main__": main()
