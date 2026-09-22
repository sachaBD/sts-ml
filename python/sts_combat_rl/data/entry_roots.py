"""Entry-root combat data writer (data/combat/README.md format) and deck-group splitting."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path
from random import Random

import msgpack
import pyarrow as pa
import pyarrow.parquet as pq

F32 = pa.float32()
COMBAT_SCHEMA = pa.schema(
    [
        ("episode_id", pa.int64()),
        ("decision_index", pa.int32()),
        ("turn", pa.int32()),
        ("entry_id", pa.string()),
        ("deck_signature", pa.string()),
        ("combat_seed", pa.uint64()),
        ("starting_hp", pa.int16()),
        ("starting_max_hp", pa.int16()),
        ("encoding_version", pa.int32()),
        ("global_numeric", pa.list_(F32, 50)),
        ("cards", pa.list_(pa.struct([("card_id", pa.int16()), ("zone", pa.int8()), ("card_type", pa.int8()),
                                      ("target_type", pa.int8()), ("numeric", pa.list_(F32, 14))]))),
        ("monsters", pa.list_(pa.struct([("monster_id", pa.int16()), ("move_id", pa.int16()),
                                         ("numeric", pa.list_(F32, 9))]))),
        ("card_monster_interactions", pa.list_(pa.struct([("card_index", pa.int16()), ("monster_index", pa.int8()),
                                                          ("numeric", pa.list_(F32, 6))]))),
        ("input_state", pa.int16()),
        ("card_selection_task", pa.int16()),
        ("actions", pa.list_(pa.struct([("action", pa.int32()), ("description", pa.string()),
                                        ("visits", pa.int64()), ("mean_value", F32)]))),
        ("chosen_action", pa.int32()),
        ("was_random", pa.bool_()),
        ("root_value", F32),
        ("won", pa.bool_()),
        ("final_hp", pa.int16()),
        ("potions", pa.int8()),
        ("terminal_value", F32),
    ]
)
PARTITION = Path("act=1/floor=16/encounter=slime_boss")
TEACHER = "sts_ml PublicBeliefCombatSearch, rollout mode 2 (guided), objective mode 0"
PARTICLES = 8


def git_provenance(path: Path) -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    dirty = (
        subprocess.run(
            ["git", "-C", str(path), "diff", "--quiet"], capture_output=True, check=False
        ).returncode
        != 0
    )
    return (revision.stdout.strip() if revision.returncode == 0 else "unknown", dirty)


def deck_signature_split(rows, validation_fraction, seed):
    signatures = sorted({row["deck_signature"] for row in rows})
    if len(signatures) < 2:
        raise ValueError("deck-signature split requires at least two signatures")
    Random(seed).shuffle(signatures)
    cut = min(
        max(1, round(len(signatures) * (1 - validation_fraction))), len(signatures) - 1
    )
    train_signatures, valid_signatures = set(signatures[:cut]), set(signatures[cut:])
    return (
        [r for r in rows if r["deck_signature"] in train_signatures],
        [r for r in rows if r["deck_signature"] in valid_signatures],
        sorted(train_signatures),
        sorted(valid_signatures),
    )


def _run_worker(command, part, result, on_episode):
    """Stream one generator process into one parquet part; store (rows, episodes, stderr) or an error."""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stderr = []
    drain = threading.Thread(target=lambda: stderr.extend(process.stderr), daemon=True)
    drain.start()
    try:
        episodes, batch, count = {}, [], 0
        with pq.ParquetWriter(part, COMBAT_SCHEMA, compression="zstd") as writer:
            for row in msgpack.Unpacker(process.stdout, raw=False):
                if row["decision_index"] != episodes.get(row["episode_id"], (0,))[0]:
                    raise ValueError(f"decision stream broken at episode {row['episode_id']}")
                if row["decision_index"] == 0:  # the generator emits a fight's rows once it has finished
                    on_episode(row["won"])
                episodes[row["episode_id"]] = (row["decision_index"] + 1, row["won"])
                batch.append(row)
                count += 1
                if len(batch) >= 256:
                    writer.write_table(pa.Table.from_pylist(batch, schema=COMBAT_SCHEMA))
                    batch = []
            if batch:
                writer.write_table(pa.Table.from_pylist(batch, schema=COMBAT_SCHEMA))
        code = process.wait()
        drain.join()
        text = b"".join(stderr).decode()
        if code:
            raise RuntimeError(f"generator exited {code}: {text}")
        result.update(rows=count, episodes=episodes, stderr=text)
    except Exception as error:  # reported by the caller
        process.kill()
        result["error"] = error


def write_combat_run(
    source,
    tag,
    generator,
    simulations=20000,
    max_actions=512,
    replicates=1,
    root_limit=10**9,
    random_window=24,
    workers=1,
    root="data/combat",
):
    source, root = Path(source), Path(root)
    run_id = f"{datetime.date.today().isoformat()}_slime_pbcs{simulations // 1000}k_{tag}"
    final = root / f"run_id={run_id}"
    temp = root.parent / f".combat-tmp-{run_id}"  # outside the query glob until complete
    if final.exists():
        raise FileExistsError(f"{final} exists; runs are never modified, pick a new tag")
    accepted = sum(
        json.loads(line).get("status") == "accepted" for line in source.read_text().splitlines() if line
    )
    roots = min(accepted, root_limit)
    workers = max(1, min(workers, roots))
    shutil.rmtree(temp, ignore_errors=True)
    (temp / PARTITION).mkdir(parents=True)
    start = time.monotonic()
    total, done, wins, lock = roots * replicates, 0, 0, threading.Lock()
    print(f"{run_id}: {total} fights, {workers} workers", flush=True)

    def on_episode(won):
        nonlocal done, wins
        with lock:
            done, wins = done + 1, wins + bool(won)
            elapsed = time.monotonic() - start
            eta = elapsed / done * (total - done)
            print(f"{time.strftime('%H:%M:%S')} {done}/{total} fights, {wins} wins, "
                  f"elapsed {elapsed / 60:.1f} min, eta {eta / 60:.1f} min", flush=True)

    try:
        results, threads = [], []
        for index in range(workers):
            command = [str(generator), str(source), str(simulations), str(max_actions), str(replicates),
                       str(root_limit), str(random_window), str(index), str(workers)]
            results.append({})
            threads.append(threading.Thread(
                target=_run_worker, args=(command, temp / PARTITION / f"part-{index:03d}.parquet", results[-1], on_episode)))
            threads[-1].start()
        for thread in threads:
            thread.join()
        for result in results:
            if "error" in result:
                raise result["error"]
        episodes = {k: v for r in results for k, v in r["episodes"].items()}
        if len(episodes) != roots * replicates:
            raise RuntimeError(f"expected {roots * replicates} episodes, got {len(episodes)}")
        project = Path(__file__).resolve().parents[3]
        repos = {name: project.parent / name for name in ("sts_lightspeed", "sts_ml")}
        manifest = {
            "run_id": run_id,
            "teacher": TEACHER,
            "simulations": simulations,
            "max_actions": max_actions,
            "particles": PARTICLES,
            "random_move": f"one per fight, uniformly random legal action at decision U[0, {random_window})",
            "workers": workers,
            "root_source": str(source.resolve()),
            "root_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "roots": roots,
            "replicates": replicates,
            "seeds": {
                "combat_seed": "source_seed xor 0x9e3779b97f4a7c15*(replicate+1)",
                "random_move_rng": "mt19937_64(combat_seed xor 0xe9510)",
                "search": "particle seeds from PublicBeliefCombatSearch::publicObservation",
            },
            "terminal_value": "win: (35 + final_hp + 4*potions) / (55 + max_hp); loss: 0",
            "rows": sum(r["rows"] for r in results),
            "episodes": len(episodes),
            "wins": sum(bool(won) for _, won in episodes.values()),
            "wall_seconds": round(time.monotonic() - start, 1),
            "generator": str(Path(generator).resolve()),
            "git": {
                name: dict(zip(("revision", "dirty"), git_provenance(path)))
                for name, path in {"sts_combat_rl": project, **repos}.items()
                if path.exists()
            },
            "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        }
        (temp / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (temp / "generator.stderr.log").write_text("".join(r["stderr"] for r in results))
        root.mkdir(parents=True, exist_ok=True)
        temp.rename(final)
        return final, manifest
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description="Write one data/combat run from natural Slime entry roots.")
    parser.add_argument("source", type=Path, help="entry-root JSONL (e.g. ../sts_ml/runs/slime-entry-natural-220.jsonl)")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--generator", default="build/generate_entry_mcts_records")
    parser.add_argument("--simulations", type=int, default=20000)
    parser.add_argument("--max-actions", type=int, default=512)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--root-limit", type=int, default=10**9)
    parser.add_argument("--random-window", type=int, default=24, help="one random move per fight at decision U[0, N); 0 disables")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--root", type=Path, default=Path("data/combat"))
    args = parser.parse_args()
    final, manifest = write_combat_run(
        args.source, args.tag, args.generator, args.simulations, args.max_actions, args.replicates,
        args.root_limit, args.random_window, args.workers, args.root,
    )
    print(final, manifest["rows"], "rows", manifest["episodes"], "episodes", manifest["wall_seconds"], "s")


if __name__ == "__main__":
    main()
