"""Offline tests for to_starts.convert/build with hand-made champ-start rows."""
import collections
import unittest

from apps.megacrit_dump import to_starts as ts

TEMPLATE = {"ascension": 20, "act": 2, "floor": 33, "encounter": 39, "cur_room": 6, "last_room": 1,
            "burning_elite_buff": -1, "gold": 4,
            "misc_rng": {"counter": 0, "seed0": 1, "seed1": 2}, "potion_rng": {"counter": 66, "seed0": 3, "seed1": 4}}
CARDS, RELICS = ts.enum_index()


def row(**kw):
    r = {"play_id": "p1", "exact": True, "issues": [], "unmapped": [], "hp": 61, "max_hp": 85,
         "deck": [{"card": "strike_red", "upgrades": 0}, {"card": "searing_blow", "upgrades": 3},
                  {"card": "bash", "upgrades": 1}],
         "relics": ["burning_blood", "neows_lament", "kunai"]}
    r.update(kw)
    return r


class T(unittest.TestCase):
    def test_convert(self):
        st = collections.Counter()
        s = ts.convert(row(), 123, TEMPLATE, CARDS, RELICS, st)
        self.assertEqual((s["hp"], s["max_hp"], s["seed"]), (61, 85, 123))
        self.assertEqual(s["deck"], [{"id": CARDS["strike_red"], "upgraded": False, "misc": 0},
                                     {"id": CARDS["searing_blow"], "upgraded": True, "misc": 0},
                                     {"id": CARDS["bash"], "upgraded": True, "misc": 0}])
        self.assertEqual([r["id"] for r in s["relics"]], [RELICS["burning_blood"], RELICS["kunai"]])  # lament dropped
        self.assertEqual((s["potion_capacity"], s["potions"], s["bottled"]), (2, [1, 1], [-1, -1, -1]))
        self.assertEqual((s["ascension"], s["act"], s["floor"], s["encounter"], s["gold"]), (20, 2, 33, 39, 4))
        self.assertEqual(st["neows_lament_dropped"], 1)
        self.assertEqual(st["searing_blow_multi_upgrade_clamped"], 1)

    def test_potion_belt_adds_two_slots(self):
        s = ts.convert(row(relics=["burning_blood", "potion_belt"]), 1, TEMPLATE, CARDS, RELICS)
        self.assertEqual((s["potion_capacity"], s["potions"]), (4, [1, 1, 1, 1]))

    def test_each_kept_run_is_one_deck_in_one_part(self):
        rows = [row(play_id=f"p{i}", relics=["burning_blood"] +
                    (["runic_dome" if i % 2 else "prismatic_shard"] if i % 5 == 0 else [])) for i in range(400)]
        out, _ = ts.build(rows, {}, TEMPLATE, 2, 5)
        parts = ts.decks(out)
        kept = {f"mc-p{i}" for i in range(400) if i % 5}
        named = [d["fight_id"] for part in parts.values() for d in part]
        self.assertEqual(sorted(named), sorted(kept))
        firsts = {f"mc-{f['play_id']}": f["start"] for f in out if f["copy"] == 1}
        self.assertTrue(all(d["start"] == firsts[d["fight_id"]] for part in parts.values() for d in part))
        smaller = ts.decks(out[:200])
        for name, ds in smaller.items():
            self.assertTrue(all(d in parts[name] for d in ds))

    def test_a_run_keeps_its_part_and_about_two_percent_are_held_out_each(self):
        parts = [ts.part(f"run-{i}") for i in range(20_000)]
        self.assertEqual(parts, [ts.part(f"run-{i}") for i in range(20_000)])
        for name in ("validation", "final"):
            self.assertAlmostEqual(parts.count(name) / len(parts), 0.02, delta=0.004)

    def test_held_out_decks_play_n_fresh_fights_with_potions(self):
        out, _ = ts.build([row(play_id=f"p{i}") for i in range(5)], {}, TEMPLATE, 1, 5)
        held = [{"fight_id": f"mc-{f['play_id']}", "start": f["start"]} for f in out]
        fights = ts.held_out_fights(held, "megacrit", 4)
        self.assertEqual(len({f["fight_id"] for f in fights}), 20)
        self.assertEqual(len({f["start"]["seed"] for f in fights}), 20)
        self.assertTrue(all(f["fight_id"].split(":")[1].startswith("mc-p") for f in fights))
        self.assertTrue(all(f["start"]["hp"] == 61 and len(f["start"]["potions"]) == f["start"]["potion_capacity"]
                            for f in fights))

    def test_unmapped_skipped(self):
        self.assertIsNone(ts.convert(row(unmapped=["relic:Dodecahedron"]), 1, TEMPLATE, CARDS, RELICS))
        self.assertIsNone(ts.convert(row(deck=[{"card": "not_a_card", "upgrades": 0}]), 1, TEMPLATE, CARDS, RELICS))

    def test_seed_twos_complement(self):
        self.assertEqual(ts.to_u64(-1), 2 ** 64 - 1)
        self.assertEqual(ts.to_i64("-5"), -5)
        self.assertIsNone(ts.to_i64("abc"))

    def test_build_copies_filter_dedupe(self):
        rows = [row(play_id="a"), row(play_id="a"), row(play_id="b", exact=False, issues=["event_card_upgrade_unknown:x"]),
                row(play_id="c", exact=False, issues=["remove_card_not_in_deck:x"])]
        out, st = ts.build(rows, {"a": -7, "b": 50}, TEMPLATE, 2, 997_000_000_000)
        self.assertEqual([f["fight_id"] for f in out], ["mc:a:1", "mc:a:2", "mc:b:1", "mc:b:2"])
        self.assertEqual(out[0]["start"]["seed"], ts.to_u64(-7))
        self.assertEqual((out[1]["start"]["seed"], out[3]["start"]["seed"]), (997_000_000_000, 997_000_000_001))
        self.assertEqual(out[2]["start"]["seed"], 50)
        self.assertEqual((st["duplicate_play_id"], st["excluded_not_exact"]), (1, 1))
        self.assertEqual([f["base_exact"] for f in out], [True] * 4)
        self.assertEqual([f["exact"] for f in out], [True, True, False, False])
        out, _ = ts.build(rows, {}, TEMPLATE, 2, 5, include="exact")
        self.assertEqual(len(out), 2)


if __name__ == "__main__":
    unittest.main()
