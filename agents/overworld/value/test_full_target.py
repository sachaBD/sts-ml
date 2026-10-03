"""Heart-mode "full" target and the emerald key rule (no model needed)."""
from agents.overworld.value.core import full_score, run_score
import random

from agents.overworld.value.policy import Policy, emerald_options, key_rule_on


def close(a, b):
    return abs(a - b) < 1e-9


def test_full_score_endpoints():
    assert close(full_score(56, 3, True, True), 1.0)
    assert close(full_score(51, 3, False, False), 0.3 * 51 / 56 + 0.30)  # act 3 cleared, heart_locked
    assert close(full_score(53, 3, True, False), 0.3 * 53 / 56 + 0.30 + 0.10)  # died in act 4
    assert close(full_score(10, 0, False, False), 0.3 * 10 / 56)
    assert close(full_score(80, 3, True, True), 1.0)  # floor is capped


def test_run_score_full():
    def run(status, floor, **kw):
        return {"status": status, "floor": floor, "steps": [], **kw}
    assert close(run_score(run("won", 56, acts_cleared=3, heart_cleared=True), "full"), 1.0)
    assert close(run_score(run("heart_locked", 51, acts_cleared=3, heart_cleared=False), "full"), 0.3 * 51 / 56 + 0.30)
    assert close(run_score(run("died", 53, acts_cleared=3), "full"), 0.3 * 53 / 56 + 0.40)
    assert close(run_score(run("died", 10, acts_cleared=0), "full"), 0.3 * 10 / 56)
    boss = [{"kind": "fight", "category": "boss", "won": True}] * 2  # acts_cleared fallback from steps
    assert close(run_score(run("died", 40, steps=boss), "full"), 0.3 * 40 / 56 + 0.15)


def nd(x, y, *edges):
    return {"x": x, "y": y, "edges": list(edges)}


def msg(act=3, emerald=False, burning=True, nexts=((0,), (1,))):
    # y=1: nodes 0,1 ; y=2: nodes 0 (->0,1), 1 (->2) ; y=3 elite candidates; burning elite at (0, 3)
    m = {"nodes": [nd(0, 1, 0), nd(1, 1, 2), nd(0, 2, 0), nd(2, 2, 2), nd(0, 3), nd(2, 3)], "current": {"y": 0}}
    if burning:
        m["burning_elite"] = {"x": 0, "y": 3}
    return {"state": {"act": act, "overworld": {"keys": {"emerald": emerald}}, "map": m},
            "after": [{"map": {"next_xs": list(n)}} for n in nexts]}


def test_emerald_options():
    assert emerald_options(msg()) == [0]  # only x=0 reaches (0,3)
    assert emerald_options(msg(nexts=((0, 1), (1,)))) == [0]
    assert emerald_options(msg(nexts=((1,), (0,)))) == [1]
    assert emerald_options(msg(emerald=True)) is None
    assert emerald_options(msg(act=1)) is None
    assert emerald_options(msg(act=2)) is None  # Act 3 only
    assert emerald_options(msg(burning=False)) is None
    assert emerald_options(msg(nexts=((1,), (1,)))) is None  # nothing reaches it


def test_key_rule_p():
    assert key_rule_on(5, 1.0) and not key_rule_on(5, 0.0) and key_rule_on(None, 1.0) and not key_rule_on(None, 0.5)
    on = [key_rule_on(s, 0.3) for s in range(2000)]
    assert on == [random.Random(s * 104729 + 17).random() < 0.3 for s in range(2000)]
    assert 0.25 < sum(on) / 2000 < 0.35
    assert Policy("simple", key_rule=True).key_rule_for(7) and not Policy("simple").key_rule_for(7)
    seeds = [s for s in range(50) if key_rule_on(s, 0.5)]
    p = Policy("simple", key_rule_p=0.5)
    assert [s for s in range(50) if p.key_rule_for(s)] == seeds and 0 < len(seeds) < 50


def test_key_rule_p0_never_restricts():
    import agents.overworld.value.policy as pol
    calls, orig = [], pol.emerald_options
    pol.emerald_options = lambda m: calls.append(m["seed"])  # spy; returns None = no restriction
    try:
        m = {**msg(), "decision": "path", "options": [0, 1], "simple": 1, "boss": "champ", "seed": 3}
        assert Policy("simple", key_rule_p=0.0).decide(m) == (1, "simple", None) and not calls
        assert Policy("simple", key_rule=True).decide(m) == (1, "simple", None) and calls == [3]
        on, off = next(s for s in range(99) if key_rule_on(s, .5)), next(s for s in range(99) if not key_rule_on(s, .5))
        p = Policy("simple", key_rule_p=0.5)
        p.decide({**m, "seed": off}); p.decide({**m, "seed": on})
        assert calls == [3, on]
    finally:
        pol.emerald_options = orig


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
