"""Offline tests for champ_starts.reconstruct with small hand-built run records (Champ on floor F=5)."""
import copy
import unittest

from apps.megacrit_dump import champ_starts as cs

F = 5
BASE = {
    "play_id": "p", "build_version": "b", "timestamp": 1, "victory": False, "floor_reached": F,
    "master_deck": ["Strike_R", "Strike_R", "Defend_R", "Bash", "Metallicize"],
    "relics": ["Burning Blood", "Kunai"],
    "current_hp_per_floor": [80, 70, 60, 50, 40, 30, 20, 10],   # [F-2] = 50: end of floor 4
    "max_hp_per_floor": [90, 91, 92, 93, 94, 95, 96, 97],       # [F-2] = 93
    "damage_taken": [{"enemies": "Champ", "damage": 12, "turns": 8, "floor": F}],
    "card_choices": [], "campfire_choices": [], "event_choices": [], "items_purchased": [], "item_purchase_floors": [],
    "items_purged": [], "items_purged_floors": [], "relics_obtained": [], "boss_relics": [{"picked": "Pantograph"}],
}


def rec(**kw):
    ev = copy.deepcopy(BASE)
    ev.update(kw)
    return ev


def deck(r):
    return [(d["card"], d["upgrades"]) for d in r["deck"]]


class T(unittest.TestCase):
    def test_died_at_champ_is_noop(self):
        r = cs.reconstruct(rec())
        self.assertEqual(r["deck_raw"], sorted(BASE["master_deck"]))
        self.assertEqual(r["relics_raw"], ["Burning Blood", "Kunai"])
        self.assertTrue(r["exact"], r["issues"])
        self.assertEqual((r["F"], r["hp"], r["max_hp"], r["champ_won"]), (F, 50, 93, False))  # hp: index F-2, not F-1
        self.assertEqual((r["champ_damage"], r["champ_turns"]), (12, 8))

    def test_died_at_champ_with_later_records_is_flagged_not_applied(self):
        r = cs.reconstruct(rec(card_choices=[{"floor": F, "picked": "Bash", "not_picked": []}]))
        self.assertFalse(r["exact"])
        self.assertEqual(r["deck_raw"], sorted(BASE["master_deck"]))

    def test_win_undoes_pick_and_later_smith(self):
        # Reaper taken on floor F+3 and smithed on 40 (final Reaper+1); Anger picked at F (Champ reward)
        ev = rec(victory=True, floor_reached=55,
                 master_deck=BASE["master_deck"] + ["Reaper+1", "Anger"],
                 card_choices=[{"floor": F, "picked": "Anger", "not_picked": []},
                               {"floor": 36, "picked": "Reaper", "not_picked": []},
                               {"floor": 38, "picked": "SKIP", "not_picked": []},
                               {"floor": 39, "picked": "Singing Bowl", "not_picked": []}],
                 campfire_choices=[{"floor": 40, "key": "SMITH", "data": "Reaper"},
                                   {"floor": 3, "key": "SMITH", "data": "Bash"}])
        r = cs.reconstruct(ev)
        self.assertEqual(r["deck_raw"], sorted(BASE["master_deck"]), r["issues"])
        self.assertTrue(r["exact"], r["issues"])
        self.assertTrue(r["champ_won"])

    def test_smith_before_champ_is_kept(self):
        ev = rec(victory=True, floor_reached=55, master_deck=["Bash+1", "Strike_R"],
                 campfire_choices=[{"floor": 3, "key": "SMITH", "data": "Bash"}])
        self.assertEqual(cs.reconstruct(ev)["deck_raw"], ["Bash+1", "Strike_R"])

    def test_searing_blow_unupgrade_and_mapping(self):
        ev = rec(victory=True, floor_reached=55, master_deck=["Searing Blow+3", "Wraith Form v2"],
                 campfire_choices=[{"floor": 7, "key": "SMITH", "data": "Searing Blow+2"}])
        r = cs.reconstruct(ev)
        self.assertEqual(r["deck_raw"], ["Searing Blow+2", "Wraith Form v2"])
        self.assertEqual(deck(r), [("searing_blow", 2), ("wraith_form", 0)])
        self.assertEqual(cs.reconstruct(rec(master_deck=["Searing Blow+3", "Wraith Form v2"]))["deck"],
                         [{"card": "searing_blow", "upgrades": 3}, {"card": "wraith_form", "upgrades": 0}])

    def test_shop_relic_after_champ_removed(self):
        ev = rec(victory=True, floor_reached=55, relics=["Burning Blood", "Kunai", "Orrery"],
                 items_purchased=["Orrery", "Fire Potion", "Kunai"], item_purchase_floors=[40, 40, 2])
        r = cs.reconstruct(ev)
        self.assertEqual(r["relics_raw"], ["Burning Blood", "Kunai"])
        self.assertEqual(r["relics"], ["burning_blood", "kunai"])
        self.assertTrue(r["exact"], r["issues"])

    def test_shop_card_and_purge_and_event_relic(self):
        ev = rec(victory=True, floor_reached=55, master_deck=["Strike_R", "Whirlwind+1"],
                 relics=["Burning Blood", "Golden Idol"],
                 items_purchased=["Whirlwind+1"], item_purchase_floors=[41],
                 items_purged=["Defend_R"], items_purged_floors=[42],
                 event_choices=[{"floor": 44, "event_name": "X", "relics_obtained": ["Golden Idol"]},
                                {"floor": 45, "event_name": "Y", "relics_lost": ["Kunai"], "cards_obtained": ["Regret"]}],
                 master_deck_extra=None)
        ev["master_deck"] = ["Strike_R", "Whirlwind+1", "Regret"]
        r = cs.reconstruct(ev)
        self.assertEqual(r["deck_raw"], ["Defend_R", "Strike_R"], r["issues"])
        self.assertEqual(r["relics_raw"], ["Burning Blood", "Kunai"])
        self.assertTrue(r["exact"], r["issues"])

    def test_black_blood(self):
        ev = rec(victory=True, floor_reached=55, relics=["Black Blood", "Kunai"],
                 boss_relics=[{"picked": "Pantograph"}, {"picked": "Black Blood"}])
        ev["relics"] = ["Black Blood", "Kunai", "Pantograph"]
        r = cs.reconstruct(ev)
        self.assertEqual(r["relics_raw"], ["Burning Blood", "Kunai", "Pantograph"])
        self.assertTrue(r["exact"], r["issues"])

    def test_impossible_removal_is_flagged_not_guessed(self):
        ev = rec(victory=True, floor_reached=55, card_choices=[{"floor": 40, "picked": "Reaper", "not_picked": []}])
        r = cs.reconstruct(ev)
        self.assertFalse(r["exact"])
        self.assertTrue(any(i.startswith("remove_card_not_in_deck:Reaper") for i in r["issues"]), r["issues"])
        self.assertEqual(r["deck_raw"], sorted(BASE["master_deck"]))  # untouched

    def test_unupgrade_without_upgraded_copy(self):
        ev = rec(victory=True, floor_reached=55, campfire_choices=[{"floor": 40, "key": "SMITH", "data": "Bash"}])
        r = cs.reconstruct(ev)
        self.assertFalse(r["exact"])
        self.assertTrue(r["issues"][0].startswith("unupgrade_no_upgraded_copy:Bash"))

    def test_deck_changing_relic_bottled_and_unmapped(self):
        ev = rec(victory=True, floor_reached=55, relics=["Burning Blood", "Astrolabe", "Bottled Flame", "Dodecahedron"],
                 boss_relics=[{"picked": "Pantograph"}, {"picked": "Astrolabe"}])
        r = cs.reconstruct(ev)
        reasons = {i.split(":")[0] for i in r["issues"]}
        self.assertEqual(reasons, {"deck_changing_relic_after_champ", "bottled_relic", "unmapped"})
        self.assertEqual(r["unmapped"], ["relic:Dodecahedron"])
        self.assertFalse(r["exact"])

    def test_event_logs_have_no_upgrade_info(self):
        # Duplicator logs "Disarm" although the copy is Disarm+1; the deck has only upgraded Disarm -> unambiguous
        ev = rec(victory=True, floor_reached=55, master_deck=["Disarm+1", "Disarm+1"],
                 event_choices=[{"floor": 38, "event_name": "Duplicator", "cards_obtained": ["Disarm"]}])
        r = cs.reconstruct(ev)
        self.assertEqual(r["deck_raw"], ["Disarm+1"])
        self.assertTrue(r["exact"], r["issues"])
        # both variants present -> cannot tell which copy was obtained
        ev["master_deck"] = ["Disarm", "Disarm+1"]
        self.assertTrue(any(i.startswith("event_card_upgrade_unknown") for i in cs.reconstruct(ev)["issues"]))
        # a removed non-curse card's upgrade state is unknowable; a removed curse is exact
        ev = rec(victory=True, floor_reached=55, master_deck=["Bash"],
                 event_choices=[{"floor": 42, "event_name": "Falling", "cards_removed": ["Feed"]}])
        self.assertFalse(cs.reconstruct(ev)["exact"])
        ev["event_choices"][0]["cards_removed"] = ["Regret"]
        r = cs.reconstruct(ev)
        self.assertTrue(r["exact"], r["issues"])
        self.assertEqual(r["deck_raw"], ["Bash", "Regret"])

    def test_event_unupgrade_and_unknown_field(self):
        ev = rec(victory=True, floor_reached=55, master_deck=["Bash+1", "Strike_R"],
                 event_choices=[{"floor": 41, "event_name": "Upgrade Shrine", "cards_upgraded": ["Bash"], "new_field": 1}])
        r = cs.reconstruct(ev)
        self.assertEqual(r["deck_raw"], ["Bash", "Strike_R"])
        self.assertEqual([i.split(":")[0] for i in r["issues"]], ["unknown_event_field"])

    def test_not_champ_and_header_mapping(self):
        self.assertIsNone(cs.reconstruct(rec(damage_taken=[])))
        self.assertEqual(cs.CARDS["Strike_R"], "STRIKE_RED")
        self.assertEqual(cs.RELICS["Black Blood"], "BLACK_BLOOD")
        self.assertIn("REGRET", {cs.CARDS[c] for c in cs.CURSES})
        self.assertNotIn("BASH", {cs.CARDS[c] for c in cs.CURSES})


if __name__ == "__main__":
    unittest.main()
