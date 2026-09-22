"""Atomic streaming writer and deck-group splitting for natural Slime roots."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from random import Random

import msgpack
import pyarrow as pa
import pyarrow.parquet as pq

from .dataset import SCHEMA, validate_row

ENTRY_SCHEMA = pa.schema(
    list(SCHEMA)
    + [
        ("entry_id", pa.string()),
        ("source_seed", pa.uint64()),
        ("public_snapshot_sha256", pa.string()),
        ("deck_signature", pa.string()),
        ("starting_hp", pa.int16()),
        ("starting_max_hp", pa.int16()),
        ("combat_seed", pa.uint64()),
        ("replicate", pa.int32()),
    ]
)


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


def write_entry_roots(
    source,
    output,
    generator,
    simulations,
    rollout,
    replicates,
    root_limit,
    overwrite=False,
):
    source, output = Path(source), Path(output)
    shard, manifest, partial = (
        output / "entry_roots.parquet",
        output / "manifest.toml",
        output / "entry_roots.parquet.partial",
    )
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; pass overwrite=True")
    rows_in = [json.loads(line) for line in source.read_text().splitlines() if line]
    accepted = [row for row in rows_in if row.get("status") == "accepted"][:root_limit]
    output.mkdir(parents=True, exist_ok=True)
    partial.unlink(missing_ok=True)
    shard.unlink(missing_ok=True)
    manifest.unlink(missing_ok=True)
    process = None
    try:
        process = subprocess.Popen(
            [
                generator,
                str(source),
                str(simulations),
                str(rollout),
                str(replicates),
                str(root_limit),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdout is not None
        episodes, batch, count = {}, [], 0
        with pq.ParquetWriter(partial, ENTRY_SCHEMA, compression="zstd") as writer:
            for row in msgpack.Unpacker(process.stdout, raw=False):
                validate_row(row, simulations)
                if any(value is None for value in row.values()):
                    raise ValueError("null generated value")
                ep = row["episode_id"]
                provenance = tuple(
                    row[x]
                    for x in (
                        "entry_id",
                        "source_seed",
                        "public_snapshot_sha256",
                        "deck_signature",
                        "starting_hp",
                        "starting_max_hp",
                        "combat_seed",
                        "replicate",
                    )
                )
                terminal = (
                    row["terminal_outcome"],
                    row["final_player_hp"],
                    row["final_player_max_hp"],
                    row["terminal_value"],
                )
                old = episodes.setdefault(ep, [0, provenance, terminal])
                if (
                    row["decision_index"] != old[0]
                    or old[1] != provenance
                    or old[2] != terminal
                ):
                    raise ValueError("episode stream/provenance invariant failed")
                old[0] += 1
                batch.append(row)
                count += 1
                if len(batch) >= 256:
                    writer.write_table(pa.Table.from_pylist(batch, schema=ENTRY_SCHEMA))
                    batch = []
            if batch:
                writer.write_table(pa.Table.from_pylist(batch, schema=ENTRY_SCHEMA))
        stderr = process.stderr.read().decode() if process.stderr else ""
        if (
            process.wait()
            or len(episodes) != len(accepted) * replicates
            or len({(x[1][0], x[1][-1]) for x in episodes.values()}) != len(episodes)
        ):
            raise RuntimeError(stderr or "episode invariant failed")
        project = Path(__file__).resolve().parents[3]
        project_revision, project_dirty = git_provenance(project)
        generator_revision, generator_dirty = git_provenance(
            Path(generator).resolve().parent
        )
        partial.replace(shard)
        (output / "generator.stderr.log").write_text(stderr)
        lines = [
            f"{k} = {json.dumps(v)}"
            for k, v in {
                "dataset_id": "a1-slime-entry-roots-v1",
                "projection": "deck_hp_only",
                "source_jsonl": str(source),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "source_accepted_roots": sum(
                    r.get("status") == "accepted" for r in rows_in
                ),
                "included_roots": len(accepted),
                "skipped_rows": len(rows_in)
                - sum(r.get("status") == "accepted" for r in rows_in),
                "episodes": len(episodes),
                "decisions": count,
                "wins": sum(x[2][0] == 1 for x in episodes.values()),
                "simulations": simulations,
                "rollout": rollout,
                "replicates": replicates,
                "root_limit": root_limit,
                "encoding_version": 3,
                "schema_version": 1,
                "group_split_key": "deck_signature",
                "combat_seed_contract": "source_seed xor 0x9e3779b97f4a7c15*(replicate+1)",
                "shard": shard.name,
                "shard_sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
                "generator": generator,
                "project_git_revision": project_revision,
                "project_git_dirty": project_dirty,
                "generator_git_revision": generator_revision,
                "generator_git_dirty": generator_dirty,
            }.items()
        ]
        manifest.write_text("\n".join(lines) + "\n")
        return shard, count, stderr
    except Exception:
        partial.unlink(missing_ok=True)
        shard.unlink(missing_ok=True)
        manifest.unlink(missing_ok=True)
        raise
    finally:
        if process is not None:
            if process.poll() is None:
                process.terminate()
            process.wait()
            if process.stdout:
                process.stdout.close()
            if process.stderr:
                process.stderr.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--generator", default="build/generate_entry_mcts_records")
    parser.add_argument("--simulations", type=int, default=8)
    parser.add_argument("--rollout", type=int, default=128)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--root-limit", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(
        write_entry_roots(
            args.source,
            args.output,
            args.generator,
            args.simulations,
            args.rollout,
            args.replicates,
            args.root_limit,
            args.overwrite,
        )[:2]
    )


if __name__ == "__main__":
    main()
