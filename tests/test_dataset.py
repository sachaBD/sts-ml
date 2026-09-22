import shutil
import tempfile
import unittest
from pathlib import Path

from sts_combat_rl.data.dataset import (
    StreamValidator,
    generate_dataset,
    validate_row,
)
from sts_combat_rl.data.reader import read_parquet_records, read_parquet_row


def make_valid_row() -> dict:
    return {
        "encoding_version": 3,
        "episode_id": 0,
        "seed": 1,
        "decision_index": 0,
        "global_numeric": [0.0] * 50,
        "input_state": 1,
        "card_selection_task": 0,
        "cards": [
            {
                "card_id": 10,
                "zone": 0,
                "card_type": 0,
                "target_type": 1,
                "numeric": [0.0] * 14,
            },
            {
                "card_id": 20,
                "zone": 1,
                "card_type": 1,
                "target_type": 0,
                "numeric": [0.0] * 14,
            },
        ],
        "monsters": [
            {
                "monster_id": 42,
                "move_id": 5,
                "numeric": [0.0] * 9,
            }
        ],
        "card_monster_interactions": [
            {
                "card_index": 0,
                "monster_index": 0,
                "numeric": [0.0] * 6,
            }
        ],
        "mcts_value": 0.25,
        "root_visits": 10,
        "chosen_action": 0,
        "terminal_outcome": 1,
        "final_player_hp": 40,
        "final_player_max_hp": 80,
        "terminal_value": 0.5,
    }


class TestDatasetValidation(unittest.TestCase):
    def test_valid_row_passes(self):
        row = make_valid_row()
        validate_row(row, simulations=10)

    def test_reject_card_selection_task_overflow_int16(self):
        row = make_valid_row()
        # Value observed from reading uninitialized memory: -637534208
        row["card_selection_task"] = -637534208
        with self.assertRaises(ValueError) as ctx:
            validate_row(row, simulations=10)
        self.assertIn("card_selection_task", str(ctx.exception))

    def test_reject_zone_overflow_int8(self):
        row = make_valid_row()
        row["cards"][0]["zone"] = 200
        with self.assertRaises(ValueError) as ctx:
            validate_row(row, simulations=10)
        self.assertIn("zone", str(ctx.exception))

    def test_reject_card_id_overflow_int16(self):
        row = make_valid_row()
        row["cards"][0]["card_id"] = 40000
        with self.assertRaises(ValueError) as ctx:
            validate_row(row, simulations=10)
        self.assertIn("card_id", str(ctx.exception))

    def test_reject_interaction_card_index_out_of_bounds(self):
        row = make_valid_row()
        row["card_monster_interactions"][0]["card_index"] = 99
        with self.assertRaises(ValueError) as ctx:
            validate_row(row, simulations=10)
        self.assertIn("card_index", str(ctx.exception))

    def test_reject_interaction_monster_index_out_of_bounds(self):
        row = make_valid_row()
        row["card_monster_interactions"][0]["monster_index"] = 5
        with self.assertRaises(ValueError) as ctx:
            validate_row(row, simulations=10)
        self.assertIn("monster_index", str(ctx.exception))

    def test_reject_invalid_outcome(self):
        row = make_valid_row()
        row["terminal_outcome"] = 0
        with self.assertRaises(ValueError) as ctx:
            validate_row(row, simulations=10)
        self.assertIn("terminal_outcome", str(ctx.exception))

    def test_reject_non_finite_float(self):
        row = make_valid_row()
        row["global_numeric"][3] = float("nan")
        with self.assertRaises(ValueError) as ctx:
            validate_row(row, simulations=10)
        self.assertIn("global_numeric", str(ctx.exception))

    def test_reject_wrong_root_visits(self):
        row = make_valid_row()
        row["root_visits"] = 50
        with self.assertRaises(ValueError) as ctx:
            validate_row(row, simulations=10)
        self.assertIn("root_visits", str(ctx.exception))

    def test_stream_validator_detects_non_contiguous_decisions(self):
        validator = StreamValidator(seed_start=1, seed_count=1, simulations=10)
        row0 = make_valid_row()
        row0["decision_index"] = 0
        validator.process_row(row0)

        row2 = make_valid_row()
        row2["decision_index"] = 2  # skipped 1!
        with self.assertRaises(ValueError) as ctx:
            validator.process_row(row2)
        self.assertIn("non-contiguous", str(ctx.exception))

    def test_stream_validator_detects_inconsistent_terminals(self):
        validator = StreamValidator(seed_start=1, seed_count=1, simulations=10)
        row0 = make_valid_row()
        row0["decision_index"] = 0
        validator.process_row(row0)

        row1 = make_valid_row()
        row1["decision_index"] = 1
        row1["terminal_outcome"] = -1  # inconsistent with row0 outcome 1!
        with self.assertRaises(ValueError) as ctx:
            validator.process_row(row1)
        self.assertIn("inconsistent terminal", str(ctx.exception))

    def test_short_real_generator_run_writes_and_reads_back(self):
        tmp_dir = Path(tempfile.mkdtemp(prefix="sts_test_gen_"))
        try:
            shard, decisions, wins = generate_dataset(
                output_dir=tmp_dir,
                seed_start=1,
                seed_count=1,
                simulations=10,
                generator="build/generate_mcts_records",
                chunk_size=10,
            )
            self.assertTrue(shard.exists())
            partial = tmp_dir / (shard.name + ".partial")
            self.assertFalse(
                partial.exists(), "partial file must not exist after successful write"
            )
            self.assertGreater(decisions, 0)
            self.assertIn(wins, (0, 1))

            records = read_parquet_records(shard)
            self.assertEqual(len(records), decisions)
            row0 = read_parquet_row(shard, 0)
            self.assertEqual(row0["encoding_version"], 3)
            self.assertEqual(row0["card_selection_task"], 0)
            self.assertEqual(row0["episode_id"], 0)
            self.assertEqual(row0["seed"], 1)

            manifest = tmp_dir / "manifest.toml"
            self.assertTrue(manifest.exists())
            manifest_text = manifest.read_text()
            self.assertIn('dataset_id = "a1-slime-boss-mcts-value-v3"', manifest_text)
            self.assertIn(f"decisions = {decisions}", manifest_text)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
