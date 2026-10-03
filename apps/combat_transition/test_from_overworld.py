"""Pre-state reconstruction and final-HP inversion of from_overworld."""
import unittest

from apps.combat_transition.from_overworld import battle_final_hp, fights

CARD = dict(card_id=1, upgraded=0, misc=0, name="strike_red")
BLOOD = dict(relic_id=86, data=0, name="burning_blood")


def state(hp, deck=1, potions=(), relics=(BLOOD,), floor=1, act=1):
    return dict(act=act, floor=floor, hp=hp, max_hp=80, gold=99, potion_capacity=3, deck=[CARD] * deck,
                potions=[dict(potion_id=p, name=f"p{p}") for p in potions], relics=list(relics))


class FromOverworldTest(unittest.TestCase):
    def test_final_hp(self):
        self.assertEqual(battle_final_hp(state(50), True), (44, False))
        self.assertEqual(battle_final_hp(state(80), True), (77, True))  # capped: midpoint of 74..80
        self.assertEqual(battle_final_hp(state(50, relics=()), True), (50, False))
        self.assertEqual(battle_final_hp(state(50), False), (0, False))

    def test_pre_state(self):
        run = dict(seed=12, steps=[
            dict(kind="start", state=state(70, potions=(3,))),
            # pre = the path decision's after-state; the potion used in the fight is gone afterwards.
            dict(kind="decide", decision="path", choice=0, after=state(70, potions=(3,))),
            dict(kind="fight", state=state(56), encounter="cultist", category="easy", won=True, hp_before=70),
            dict(kind="pick", state=state(56), options=[CARD], choice=0),
            dict(kind="fight", state=state(46, deck=2, floor=2), encounter="jaw_worm", category="easy", won=True,
                 hp_before=56),
            # an unrecorded change (e.g. an event) leaves the rebuilt pre-state stale
            dict(kind="fight", state=state(30, deck=2, floor=3), encounter="cultist", category="easy", won=False,
                 hp_before=20)])
        rows = list(fights(run, "src"))
        self.assertEqual([r["replay"] for r in rows], ["ok", "ok", "stale"])
        self.assertEqual(rows[0]["pre"]["potions"], [dict(potion_id=3, name="p3")])
        self.assertEqual((rows[0]["final_hp"], rows[1]["final_hp"], rows[2]["final_hp"]), (50, 40, 0))
        self.assertEqual(len(rows[1]["pre"]["deck"]), 2)
        self.assertEqual(rows[0]["bucket"], 2)


if __name__ == "__main__":
    unittest.main()
