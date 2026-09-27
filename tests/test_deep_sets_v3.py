"""deep_sets_v3 end to end: C++ encoder rows -> the combat_v3 table schema -> the training loader (data.py,
data_v4.py) -> the Python model, compared with the C++ inference (topology/value_net.cpp): values and policy
logits. Also the score-formula output, policy targets, and training losses.

Runs build/<STSRL_BUILD_DIR, default main>/encoding_v4_test; skipped if that is not built.
"""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch

from sts_combat_rl.schemas.combat_v3 import COMBAT_V3
from sts_combat_rl.topology import ENCODING_VERSIONS, build
from sts_combat_rl.training import data, data_v4
from sts_combat_rl.training.export_value_weights import export
from sts_combat_rl.training.train_value import policy_ce, pop_aux, segment_log_softmax

ROOT = Path(__file__).resolve().parents[1]
V3 = {"kind": "deep_sets_v3", "card_vocab": 512, "monster_vocab": 128, "move_vocab": 512, "potion_vocab": 64,
      "relic_vocab": 192, "width": 64, "card_id_dim": 16, "monster_id_dim": 8, "move_dim": 8, "potion_id_dim": 8,
      "relic_id_dim": 8, "id_dropout": 0.15, "pool_count_features": True, "head_input_norm": True,
      "head_width": 64, "head_blocks": 2, "zero_init_blocks": False, "score_hp_offset": 35.0,
      "score_potion_hp": 4.0, "score_max_hp_offset": 55.0, "policy_width": 32}


def model(seed=0, **overrides):
    torch.manual_seed(seed)
    net = build({**V3, **overrides})
    with torch.no_grad():  # larger random output heads so every term matters
        for head in (net.won_out, net.hp_out, net.keep_out, *([net.policy_out] if net.policy_width else [])):
            head.weight.normal_(0, 0.5)
            head.bias.normal_(0, 0.5)
    return net.eval()


def cpp(net, *mode):
    binary = ROOT / "build" / os.environ.get("STSRL_BUILD_DIR", "main") / "encoding_v4_test"
    if not binary.exists():
        raise unittest.SkipTest(f"{binary} is not built")
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint, weights, out = Path(tmp, "c.pt"), Path(tmp, "w.bin"), Path(tmp, "out.json")
        torch.save({"architecture": net.config, "encoding_version": ENCODING_VERSIONS["deep_sets_v3"],
                    "model_state": net.state_dict()}, checkpoint)
        export(checkpoint, weights)
        subprocess.run([str(binary), str(weights), str(out), *mode], check=True)
        return json.loads(out.read_text())


def table_rows(results):
    """The C++ states as combat_v3 rows (the writer's columns plus made-up outcome / search columns: visits
    1, 2, 3, ... on the legal actions in order), packed and loaded as training rows."""
    rows = []
    for i, r in enumerate(results):
        state = r["state"]
        actions = [{"action": a["action"], "description": "", "visits": k + 1, "mean_value": 0.5}
                   for k, a in enumerate(state["legal_actions"])]
        rows.append({**state, "run_seed": i, "episode_id": i, "decision_index": 0, "row_kind": "decision",
                     "was_random": False, "root_value": 0.5, "terminal_value": 0.5, "won": True, "final_hp": 30,
                     "starting_max_hp": 80, "encounter": "x", "category": "elite", "potions": 1, "oracle": False,
                     "actions": actions})
    table = pa.Table.from_pylist(rows, schema=COMBAT_V3)
    table = table.append_column("run_id", pa.array(["test"] * len(rows)))
    batch = table.combine_chunks().to_batches()[0]
    out = data._pack(batch, 0)
    data_v4.pack(batch, 0, out)
    loaded = data.Rows(out)
    loaded.columns["target"] = np.zeros(len(loaded))
    data.assign_aux_targets(loaded)
    data_v4.assign_keep_targets(loaded.columns)
    return loaded


def inputs(rows, index):
    batch = rows.collate(index, aux=True)
    batch.pop("target")
    return batch, pop_aux(batch)


class DeepSetsV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.net = model()
        cls.results = cpp(cls.net)
        cls.rows = table_rows(cls.results)

    def test_value_and_policy_parity_with_cpp(self):
        states = [r["state"] for r in self.results]
        self.assertTrue(any(s["potion_tokens"] for s in states) and any(s["relic_tokens"] for s in states))
        batch, _ = inputs(self.rows, np.arange(len(self.rows)))
        with torch.no_grad():
            out = self.net.forward_all(**batch)
        kept = self.rows["legal_actions.action"]
        owners = batch["action_state_indices"].numpy()
        for i, r in enumerate(self.results):
            self.assertAlmostEqual(out["value"][i].item(), r["value"], places=5)
            # merged tokens: the kept first token of each identical group, by its action index
            cpp_logit = {a["action"]: x for a, x in zip(r["state"]["legal_actions"], r["logits"])}  # (sorted tokens)
            for logit, action in zip(out["policy_logits"][owners == i], kept[owners == i]):
                self.assertAlmostEqual(logit.item(), cpp_logit[action], places=4)
            # one state at a time: pooling / policy must not mix states
            single, _ = inputs(self.rows, [i])
            with torch.no_grad():
                self.assertAlmostEqual(self.net(**single).item(), r["value"], places=5)

    def test_identical_actions_merge_and_policy_targets(self):
        c = self.rows.columns
        total = np.zeros(len(self.rows))
        owners = np.repeat(np.arange(len(self.rows)), c["legal_actions.count"])
        np.add.at(total, owners, c["legal_actions.policy_target"])
        np.testing.assert_allclose(total, 1.0, rtol=1e-6)
        for i, r in enumerate(self.results):  # every merged group gets the sum of its visits
            legal = r["state"]["legal_actions"]
            visits = np.arange(1, len(legal) + 1, dtype=float)
            groups = {}
            for a, v in zip(legal, visits):
                key = json.dumps({k: a[k] for k in a if k != "action"}, sort_keys=True)
                groups[key] = groups.get(key, 0.0) + v
            self.assertEqual(c["legal_actions.count"][i], len(groups))
            np.testing.assert_allclose(sorted(c["legal_actions.policy_target"][owners == i]),
                                       sorted(np.array(list(groups.values())) / visits.sum()), rtol=1e-5)
            # C++ gives identical logits to identical tokens
            by_key = {}
            for a, logit in zip(legal, r["logits"]):
                by_key.setdefault(json.dumps({k: a[k] for k in a if k != "action"}, sort_keys=True), set()).add(
                    round(logit, 5))
            self.assertTrue(all(len(v) == 1 for v in by_key.values()))

    def test_score_formula_and_keep_target(self):
        batch, aux = inputs(self.rows, np.arange(len(self.rows)))
        with torch.no_grad():
            out = self.net.forward_all(**batch)
        potions = torch.bincount(batch["potion_state_indices"], minlength=len(self.rows)).float()
        max_hp = batch["max_hp"]
        expected = torch.sigmoid(out["won_logit"]) * (35 + out["hp_fraction"] * max_hp
                                                      + 4 * out["keep_fraction"] * potions) / (55 + max_hp)
        torch.testing.assert_close(out["value"], expected)
        held = self.rows["potion_tokens.count"]
        np.testing.assert_allclose(aux["aux_keep"].numpy(), np.clip(1 / np.maximum(held, 1), 0, 1))
        np.testing.assert_array_equal(aux["aux_keep_mask"].numpy() > 0, (held > 0) & (aux["aux_hp_mask"] > 0).numpy())

    def test_policy_loss(self):
        logits = torch.tensor([1.0, 2.0, 0.5, 3.0])
        owner = torch.tensor([0, 0, 0, 1])
        logp = segment_log_softmax(logits, owner, 2)
        torch.testing.assert_close(logp[:3], torch.log_softmax(logits[:3], 0))
        torch.testing.assert_close(logp[3], torch.tensor(0.0))
        target = torch.tensor([0.5, 0.5, 0.0, 1.0])
        _, loss = policy_ce({"policy_logits": logits}, {"action_state_indices": owner},
                            {"policy_target": target, "policy_mask": torch.tensor([1.0, 0.0])})
        torch.testing.assert_close(loss, -(0.5 * logp[0] + 0.5 * logp[1]))

    def test_training_step_reduces_losses(self):
        net = model(3)
        net.train()
        batch, aux = inputs(self.rows, np.arange(len(self.rows)))
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
        first = None
        for _ in range(30):
            out = net.forward_all(**batch)
            loss = policy_ce(out, batch, aux)[1] + ((out["value"] - 0.5) ** 2).mean()
            first = first if first is not None else loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.assertLess(loss.item(), first)

    def test_id_dropout_only_in_training(self):
        net = model(2)
        batch, _ = inputs(self.rows, np.arange(len(self.rows)))
        with torch.no_grad():
            torch.testing.assert_close(net(**batch), net(**batch))
        net.train()
        net.id_dropout = 1.0
        self.assertTrue((net._drop_ids(batch["potion_ids"]) == 0).all())

    def test_value_only_variant_has_no_policy(self):
        net = model(4, policy_width=0)
        results = cpp(net)
        self.assertNotIn("logits", results[0])
        batch, _ = inputs(table_rows(results), np.arange(len(results)))
        with torch.no_grad():
            out = net.forward_all(**batch)
        self.assertNotIn("policy_logits", out)
        for i, r in enumerate(results):
            self.assertAlmostEqual(out["value"][i].item(), r["value"], places=5)

    def test_policy_net_search_runs(self):
        """Leaf policy_net (every-node priors from the policy head) plays a legal move within budget."""
        for r in cpp(self.net, "search"):
            self.assertTrue(0 <= r["chosen"] < r["legal"])
            self.assertTrue(1 <= r["used"] <= 400)
            self.assertGreaterEqual(r["root_visits"], 1)

    def test_architecture_is_explicit(self):
        with self.assertRaises(ValueError):
            build({k: v for k, v in V3.items() if k != "policy_width"})


if __name__ == "__main__":
    unittest.main()
