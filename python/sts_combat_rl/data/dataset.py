from __future__ import annotations

import argparse
import datetime
import hashlib
import math
import subprocess
import time
from collections.abc import Iterator
from contextlib import suppress
from pathlib import Path
from typing import Any

import msgpack
import pyarrow as pa
import pyarrow.parquet as pq

INT8_MIN, INT8_MAX = -128, 127
INT16_MIN, INT16_MAX = -32768, 32767
INT32_MIN, INT32_MAX = -2147483648, 2147483647
INT64_MIN, INT64_MAX = -9223372036854775808, 9223372036854775807
UINT64_MIN, UINT64_MAX = 0, 18446744073709551615

F32 = pa.float32()
SCHEMA = pa.schema(
    [
        ("encoding_version", pa.int32()),
        ("episode_id", pa.int64()),
        ("seed", pa.uint64()),
        ("decision_index", pa.int32()),
        ("global_numeric", pa.list_(F32, 50)),
        ("input_state", pa.int16()),
        ("card_selection_task", pa.int16()),
        (
            "cards",
            pa.list_(
                pa.struct(
                    [
                        ("card_id", pa.int16()),
                        ("zone", pa.int8()),
                        ("card_type", pa.int8()),
                        ("target_type", pa.int8()),
                        ("numeric", pa.list_(F32, 14)),
                    ]
                )
            ),
        ),
        (
            "monsters",
            pa.list_(
                pa.struct(
                    [
                        ("monster_id", pa.int16()),
                        ("move_id", pa.int16()),
                        ("numeric", pa.list_(F32, 9)),
                    ]
                )
            ),
        ),
        (
            "card_monster_interactions",
            pa.list_(
                pa.struct(
                    [
                        ("card_index", pa.int16()),
                        ("monster_index", pa.int8()),
                        ("numeric", pa.list_(F32, 6)),
                    ]
                )
            ),
        ),
        ("mcts_value", F32),
        ("root_visits", pa.int64()),
        ("chosen_action", pa.int64()),
        ("terminal_outcome", pa.int8()),
        ("final_player_hp", pa.int16()),
        ("final_player_max_hp", pa.int16()),
        ("terminal_value", F32),
    ]
)


def git_revision(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def git_dirty(path: Path) -> bool:
    try:
        return (
            subprocess.run(
                ["git", "-C", str(path), "diff", "--quiet"],
                capture_output=True,
                check=False,
            ).returncode
            != 0
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def validate_int_range(name: str, val: Any, min_val: int, max_val: int) -> None:
    if not isinstance(val, int) or isinstance(val, bool):
        raise TypeError(
            f"field '{name}' must be an integer, got {type(val).__name__} ({val!r})"
        )
    if not (min_val <= val <= max_val):
        raise ValueError(
            f"field '{name}' value {val} out of range [{min_val}, {max_val}]"
        )


def validate_row(row: dict[str, Any], simulations: int) -> None:
    validate_int_range("encoding_version", row.get("encoding_version"), 3, 3)
    validate_int_range("episode_id", row.get("episode_id"), 0, INT64_MAX)
    validate_int_range("seed", row.get("seed"), UINT64_MIN, UINT64_MAX)
    validate_int_range("decision_index", row.get("decision_index"), 0, INT32_MAX)

    global_num = row.get("global_numeric")
    if not isinstance(global_num, (list, tuple)) or len(global_num) != 50:
        length = (
            len(global_num) if isinstance(global_num, (list, tuple)) else "non-sequence"
        )
        raise ValueError(f"global_numeric must have length 50, got {length}")
    for i, x in enumerate(global_num):
        if (
            not isinstance(x, (int, float))
            or isinstance(x, bool)
            or not math.isfinite(x)
        ):
            raise ValueError(f"global_numeric[{i}] must be a finite float, got {x!r}")

    validate_int_range("input_state", row.get("input_state"), INT16_MIN, INT16_MAX)
    validate_int_range(
        "card_selection_task", row.get("card_selection_task"), INT16_MIN, INT16_MAX
    )

    cards = row.get("cards")
    if not isinstance(cards, (list, tuple)):
        raise TypeError("cards must be a list")
    for ci, card in enumerate(cards):
        validate_int_range(
            f"cards[{ci}].card_id", card.get("card_id"), INT16_MIN, INT16_MAX
        )
        validate_int_range(f"cards[{ci}].zone", card.get("zone"), INT8_MIN, INT8_MAX)
        validate_int_range(
            f"cards[{ci}].card_type", card.get("card_type"), INT8_MIN, INT8_MAX
        )
        validate_int_range(
            f"cards[{ci}].target_type", card.get("target_type"), INT8_MIN, INT8_MAX
        )
        card_num = card.get("numeric")
        if not isinstance(card_num, (list, tuple)) or len(card_num) != 14:
            raise ValueError(f"cards[{ci}].numeric must have length 14")
        for j, x in enumerate(card_num):
            if (
                not isinstance(x, (int, float))
                or isinstance(x, bool)
                or not math.isfinite(x)
            ):
                raise ValueError(
                    f"cards[{ci}].numeric[{j}] must be a finite float, got {x!r}"
                )

    monsters = row.get("monsters")
    if not isinstance(monsters, (list, tuple)):
        raise TypeError("monsters must be a list")
    for mi, monster in enumerate(monsters):
        validate_int_range(
            f"monsters[{mi}].monster_id",
            monster.get("monster_id"),
            INT16_MIN,
            INT16_MAX,
        )
        validate_int_range(
            f"monsters[{mi}].move_id", monster.get("move_id"), INT16_MIN, INT16_MAX
        )
        mon_num = monster.get("numeric")
        if not isinstance(mon_num, (list, tuple)) or len(mon_num) != 9:
            raise ValueError(f"monsters[{mi}].numeric must have length 9")
        for j, x in enumerate(mon_num):
            if (
                not isinstance(x, (int, float))
                or isinstance(x, bool)
                or not math.isfinite(x)
            ):
                raise ValueError(
                    f"monsters[{mi}].numeric[{j}] must be a finite float, got {x!r}"
                )

    interactions = row.get("card_monster_interactions")
    if not isinstance(interactions, (list, tuple)):
        raise TypeError("card_monster_interactions must be a list")
    num_cards = len(cards)
    num_monsters = len(monsters)
    for ii, item in enumerate(interactions):
        validate_int_range(
            f"interactions[{ii}].card_index",
            item.get("card_index"),
            INT16_MIN,
            INT16_MAX,
        )
        card_idx = item["card_index"]
        if not (0 <= card_idx < num_cards):
            raise ValueError(
                f"interactions[{ii}].card_index {card_idx} out of bounds (num_cards={num_cards})"
            )
        validate_int_range(
            f"interactions[{ii}].monster_index",
            item.get("monster_index"),
            INT8_MIN,
            INT8_MAX,
        )
        monster_idx = item["monster_index"]
        if not (0 <= monster_idx < num_monsters):
            raise ValueError(
                f"interactions[{ii}].monster_index {monster_idx} out of bounds (num_monsters={num_monsters})"
            )
        inter_num = item.get("numeric")
        if not isinstance(inter_num, (list, tuple)) or len(inter_num) != 6:
            raise ValueError(f"interactions[{ii}].numeric must have length 6")
        for j, x in enumerate(inter_num):
            if (
                not isinstance(x, (int, float))
                or isinstance(x, bool)
                or not math.isfinite(x)
            ):
                raise ValueError(
                    f"interactions[{ii}].numeric[{j}] must be a finite float, got {x!r}"
                )

    mcts_val = row.get("mcts_value")
    if (
        not isinstance(mcts_val, (int, float))
        or isinstance(mcts_val, bool)
        or not math.isfinite(mcts_val)
        or not (-1.0 <= mcts_val <= 1.0)
    ):
        raise ValueError(f"mcts_value must be in [-1.0, 1.0], got {mcts_val!r}")

    validate_int_range("root_visits", row.get("root_visits"), simulations, simulations)
    validate_int_range("chosen_action", row.get("chosen_action"), 0, INT64_MAX)

    outcome = row.get("terminal_outcome")
    if outcome not in (1, -1):
        raise ValueError(f"terminal_outcome must be 1 or -1, got {outcome!r}")

    validate_int_range(
        "final_player_hp", row.get("final_player_hp"), INT16_MIN, INT16_MAX
    )
    validate_int_range(
        "final_player_max_hp", row.get("final_player_max_hp"), INT16_MIN, INT16_MAX
    )

    term_val = row.get("terminal_value")
    if (
        not isinstance(term_val, (int, float))
        or isinstance(term_val, bool)
        or not math.isfinite(term_val)
        or not (-1.0 <= term_val <= 1.0)
    ):
        raise ValueError(f"terminal_value must be in [-1.0, 1.0], got {term_val!r}")


class StreamValidator:
    def __init__(self, seed_start: int, seed_count: int, simulations: int) -> None:
        self.seed_start = seed_start
        self.seed_count = seed_count
        self.simulations = simulations
        self.current_episode = -1
        self.expected_decision_index = 0
        self.current_terminal_tuple: tuple[int, int, int, float] | None = None
        self.total_decisions = 0
        self.total_wins = 0

    def process_row(self, row: dict[str, Any]) -> None:
        validate_row(row, self.simulations)
        episode_id = row["episode_id"]
        seed = row["seed"]
        decision_index = row["decision_index"]
        terminal_tuple = (
            row["terminal_outcome"],
            row["final_player_hp"],
            row["final_player_max_hp"],
            float(row["terminal_value"]),
        )

        if episode_id != self.current_episode:
            if episode_id != self.current_episode + 1:
                raise ValueError(
                    f"out of order episode_id: expected {self.current_episode + 1}, got {episode_id}"
                )
            if episode_id >= self.seed_count:
                raise ValueError(
                    f"episode_id {episode_id} exceeds expected count {self.seed_count}"
                )
            expected_seed = self.seed_start + episode_id
            if seed != expected_seed:
                raise ValueError(
                    f"seed mismatch for episode {episode_id}: expected {expected_seed}, got {seed}"
                )
            if decision_index != 0:
                raise ValueError(
                    f"new episode {episode_id} did not start with decision_index 0 (got {decision_index})"
                )
            self.current_episode = episode_id
            self.expected_decision_index = 1
            self.current_terminal_tuple = terminal_tuple
            if terminal_tuple[0] == 1:
                self.total_wins += 1
        else:
            if decision_index != self.expected_decision_index:
                raise ValueError(
                    f"non-contiguous decision_index in episode {episode_id}: expected {self.expected_decision_index}, got {decision_index}"
                )
            if terminal_tuple != self.current_terminal_tuple:
                raise ValueError(
                    f"inconsistent terminal fields in episode {episode_id}: expected {self.current_terminal_tuple}, got {terminal_tuple}"
                )
            self.expected_decision_index += 1

        self.total_decisions += 1

    def finish(self) -> None:
        completed_episodes = self.current_episode + 1
        if completed_episodes != self.seed_count:
            raise ValueError(
                f"incomplete episodes: completed {completed_episodes}, expected {self.seed_count}"
            )


def stream_generator_records(
    generator: str,
    seed_start: int,
    seed_count: int,
    simulations: int,
    rollout_limit: int,
    exploration: float,
) -> Iterator[dict[str, Any]]:
    command = [
        generator,
        str(seed_start),
        str(seed_count),
        str(simulations),
        str(rollout_limit),
        str(exploration),
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=None)
    try:
        assert process.stdout is not None
        unpacker = msgpack.Unpacker(process.stdout, raw=False)
        yield from unpacker
    finally:
        if process.stdout is not None and not process.stdout.closed:
            process.stdout.close()
        ret = process.wait()
        if ret != 0:
            raise RuntimeError(f"C++ record generator failed with exit code {ret}")


def run_generator(
    generator: str,
    seed_start: int,
    seed_count: int,
    simulations: int,
    rollout_limit: int,
    exploration: float,
) -> list[dict[str, Any]]:
    return list(
        stream_generator_records(
            generator=generator,
            seed_start=seed_start,
            seed_count=seed_count,
            simulations=simulations,
            rollout_limit=rollout_limit,
            exploration=exploration,
        )
    )


def validate_records(
    rows: list[dict[str, Any]], seed_start: int, seed_count: int, simulations: int
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    validator = StreamValidator(
        seed_start=seed_start, seed_count=seed_count, simulations=simulations
    )
    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        validator.process_row(row)
        groups.setdefault((row["episode_id"], row["seed"]), []).append(row)
    validator.finish()
    return groups


def write_manifest(
    output_dir: Path,
    shard: Path,
    seed_start: int,
    seed_count: int,
    simulations: int,
    rollout_limit: int,
    exploration: float,
    episodes: int | None = None,
    decisions: int | None = None,
    wins: int | None = None,
    groups: dict | None = None,
    rows: list[dict] | None = None,
) -> int:
    if groups is not None:
        if episodes is None:
            episodes = len(groups)
        if wins is None:
            wins = sum(
                next(iter(group))["terminal_outcome"] == 1 for group in groups.values()
            )
    if rows is not None and decisions is None:
        decisions = len(rows)

    assert episodes is not None and decisions is not None and wins is not None

    digest = hashlib.sha256(shard.read_bytes()).hexdigest()
    root = Path(__file__).resolve().parents[3]

    text = f'''dataset_id = "a1-slime-boss-mcts-value-v3"
created_utc = "{datetime.datetime.now(datetime.timezone.utc).isoformat()}"
schema_version = 3
scenario = "A1 Slime Boss fixed enhanced Ironclad deck"
project_git_revision = "{git_revision(root)}"
project_git_dirty = {str(git_dirty(root)).lower()}
simulator_git_revision = "{git_revision(root.parent / "sts_lightspeed")}"
seed_start = {seed_start}
seed_count = {seed_count}
simulations = {simulations}
rollout_limit = {rollout_limit}
exploration = {exploration}
episodes = {episodes}
decisions = {decisions}
wins = {wins}
shard = "{shard.name}"
sha256 = "{digest}"
'''
    (output_dir / "manifest.toml").write_text(text)
    return wins


def generate_dataset(
    output_dir: Path,
    seed_start: int = 1,
    seed_count: int = 32,
    simulations: int = 2000,
    rollout_limit: int = 512,
    exploration: float = 1.4142135623730951,
    generator: str = "build/generate_mcts_records",
    chunk_size: int = 2000,
) -> tuple[Path, int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    shard = output_dir / "mcts_slime_v3.parquet"
    partial_shard = output_dir / (shard.name + ".partial")

    if partial_shard.exists():
        partial_shard.unlink()

    validator = StreamValidator(
        seed_start=seed_start, seed_count=seed_count, simulations=simulations
    )
    writer = None
    buffer: list[dict[str, Any]] = []

    try:
        writer = pq.ParquetWriter(partial_shard, SCHEMA, compression="zstd")

        def flush_buffer() -> None:
            if not buffer:
                return
            batch_table = pa.Table.from_pylist(buffer, schema=SCHEMA)
            if any(column.null_count for column in batch_table.columns):
                raise ValueError("Arrow null values detected")
            assert writer is not None
            writer.write_table(batch_table)
            buffer.clear()

        for row in stream_generator_records(
            generator=generator,
            seed_start=seed_start,
            seed_count=seed_count,
            simulations=simulations,
            rollout_limit=rollout_limit,
            exploration=exploration,
        ):
            validator.process_row(row)
            buffer.append(row)
            if len(buffer) >= chunk_size:
                flush_buffer()

        flush_buffer()
        validator.finish()
        writer.close()
        writer = None

        partial_shard.replace(shard)
    except Exception:
        if writer is not None:
            with suppress(Exception):
                writer.close()
        if partial_shard.exists():
            with suppress(OSError):
                partial_shard.unlink()
        raise

    wins = write_manifest(
        output_dir=output_dir,
        shard=shard,
        seed_start=seed_start,
        seed_count=seed_count,
        simulations=simulations,
        rollout_limit=rollout_limit,
        exploration=exploration,
        episodes=seed_count,
        decisions=validator.total_decisions,
        wins=validator.total_wins,
    )
    return shard, validator.total_decisions, wins


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate MCTS Parquet dataset using C++ generator"
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--simulations", type=int, default=2000)
    parser.add_argument("--rollout-limit", type=int, default=512)
    parser.add_argument("--exploration", type=float, default=1.4142135623730951)
    parser.add_argument("--generator", default="build/generate_mcts_records")
    parser.add_argument("--chunk-size", type=int, default=2000)
    args = parser.parse_args()

    started = time.monotonic()
    shard, decisions, wins = generate_dataset(
        output_dir=args.output_dir,
        seed_start=args.seed_start,
        seed_count=args.seed_count,
        simulations=args.simulations,
        rollout_limit=args.rollout_limit,
        exploration=args.exploration,
        generator=args.generator,
        chunk_size=args.chunk_size,
    )
    elapsed = time.monotonic() - started
    print(f"shard={shard} decisions={decisions} wins={wins} runtime={elapsed:.1f}s")


if __name__ == "__main__":
    main()
