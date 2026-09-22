"""Paired rollout-vs-neural teacher-search evaluation on the held-out entry decks.

Runs build/play_entry_pbcs once per (arm, deck, seed) in a process pool. Each worker
loads the value net once and answers the binary's batched leaf requests. Appends one
JSON line per fight to <output>/episodes.jsonl (fights already there are skipped) and
one progress line per finished fight to <output>/eval.log.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import struct
import subprocess
import time
from pathlib import Path

import msgpack
import torch

from sts_combat_rl.models.deep_sets import DeepSetsValue

from .data import collate_states
from .neural_gameplay import exact_read

SEED_XORS = (0xA0761D6478BD642F, 0x5851F42D4C957F2D)

_model: DeepSetsValue | None = None


def _init_worker(checkpoint_path: str) -> None:
    global _model
    torch.set_num_threads(1)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("encoding_version") != 3:
        raise ValueError("checkpoint is not encoding v3")
    _model = DeepSetsValue(**checkpoint["architecture"])
    _model.load_state_dict(checkpoint["model_state"])
    _model.eval()


def play(task: dict) -> dict:
    command = [
        task["binary"],
        task["source"],
        task["deck_signature"],
        str(task["seed"]),
        task["arm"],
        str(task["simulations"]),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    assert process.stdin and process.stdout
    try:
        while True:
            size = struct.unpack("!I", exact_read(process.stdout, 4))[0]
            message = msgpack.unpackb(exact_read(process.stdout, size), raw=False)
            if message["type"] == "result":
                if process.wait() != 0:
                    raise RuntimeError(f"play_entry_pbcs failed: {command}")
                message.pop("type")
                return message
            if message["type"] != "leaves":
                raise ValueError(f"unknown message {message['type']}")
            with torch.no_grad():
                batch = collate_states(message["states"])
                batch.pop("target")
                values = _model(**batch).reshape(-1).tolist()
            process.stdin.write(struct.pack(f"!{len(values)}f", *values))
            process.stdin.flush()
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()


def arm_name(r: dict) -> str:
    return f"{r['mode']}-{r['simulations']}"


def summarize(results: list[dict], baseline: str) -> dict:
    """Per arm: wins, time per fight, and paired changes against the baseline arm."""
    by_key = {(arm_name(r), r["deck_signature"], r["seed"]): r for r in results}
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    summary = {}
    for arm in sorted({arm_name(r) for r in results}):
        rows = [r for r in results if arm_name(r) == arm]
        pairs = [
            (by_key[(baseline, r["deck_signature"], r["seed"])], r)
            for r in rows
            if (baseline, r["deck_signature"], r["seed"]) in by_key
        ]
        summary[arm] = {
            "fights": len(rows),
            "wins": sum(r["won"] for r in rows),
            "mean_wall_seconds": round(mean([r["wall_seconds"] for r in rows]), 1),
            "pairs_vs_baseline": len(pairs),
            "won_where_baseline_lost": sum(b["won"] and not a["won"] for a, b in pairs),
            "lost_where_baseline_won": sum(a["won"] and not b["won"] for a, b in pairs),
            "mean_final_hp_minus_baseline": round(mean([b["final_hp"] - a["final_hp"] for a, b in pairs]), 2),
        }
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--binary", type=Path, default=Path("build/play_entry_pbcs"))
    p.add_argument(
        "--arms",
        default="rollout:20000,neural:400",
        help="comma-separated mode:simulations; the first arm is the paired baseline",
    )
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--decks", type=int, default=None, help="limit to the first N decks (smoke runs)")
    p.add_argument("--seeds", type=int, default=2)
    args = p.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    signatures = checkpoint["validation_deck_signatures"][: args.decks]
    source_seed = {}
    for line in args.source.read_text().splitlines():
        row = json.loads(line)
        if row.get("status") == "accepted" and row["deck_signature"] in signatures:
            source_seed.setdefault(row["deck_signature"], int(row["seed"]))
    if set(source_seed) != set(signatures):
        raise ValueError("held-out signatures missing from source")

    args.output.mkdir(parents=True, exist_ok=True)
    episodes = args.output / "episodes.jsonl"
    done = []
    if episodes.exists():
        done = [json.loads(x) for x in episodes.read_text().splitlines() if x.strip()]
    done_keys = {(arm_name(r), r["deck_signature"], r["seed"]) for r in done}
    arms = [(m, int(n)) for m, n in (a.split(":") for a in args.arms.split(","))]
    tasks = [
        {
            "binary": str(args.binary.resolve()),
            "source": str(args.source.resolve()),
            "deck_signature": d,
            "seed": source_seed[d] ^ x,
            "arm": mode,
            "simulations": n,
        }
        for d in signatures
        for x in SEED_XORS[: args.seeds]
        for mode, n in arms
        if (f"{mode}-{n}", d, source_seed[d] ^ x) not in done_keys
    ]

    log = (args.output / "eval.log").open("a", buffering=1)

    def say(text: str) -> None:
        print(text, flush=True)
        log.write(text + "\n")
        log.flush()

    say(
        f"start {time.strftime('%Y-%m-%d %H:%M:%S')} tasks={len(tasks)} already_done={len(done)} "
        f"arms={args.arms} workers={args.workers}"
    )
    results = list(done)
    counts = {f"{m}-{n}": [0, 0] for m, n in arms}  # [wins, fights] this session
    start = time.monotonic()
    with episodes.open("a") as out, mp.get_context("spawn").Pool(
        args.workers, initializer=_init_worker, initargs=(str(args.checkpoint),)
    ) as pool:
        for n, result in enumerate(pool.imap_unordered(play, tasks), 1):
            out.write(json.dumps(result, sort_keys=True) + "\n")
            out.flush()
            results.append(result)
            c = counts[arm_name(result)]
            c[0] += result["won"]
            c[1] += 1
            elapsed = time.monotonic() - start
            eta = elapsed / n * (len(tasks) - n)
            wins = " ".join(f"{a}={w}/{f}" for a, (w, f) in counts.items() if f)
            say(
                f"[{n}/{len(tasks)}] arm={arm_name(result)} deck={result['deck_signature']} "
                f"seed={result['seed']} won={int(result['won'])} hp={result['final_hp']}/{result['max_hp']} "
                f"fight_s={result['wall_seconds']:.0f} wins: {wins} elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m"
            )
    baseline = f"{arms[0][0]}-{arms[0][1]}"
    say(f"summary (baseline {baseline}) " + json.dumps(summarize(results, baseline), sort_keys=True))


if __name__ == "__main__":
    main()
