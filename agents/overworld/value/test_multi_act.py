"""Multi-act target, per-node bosses and policy exploration (no model needed)."""
from agents.overworld.value.core import acts_cleared, floor_score, node_bosses, nodes, run_score
from agents.overworld.value.policy import Policy

STATE = {"floor": 1, "hp": 50, "max_hp": 80}


def run(status, floor, steps, **kw):
    return {"status": status, "floor": floor, "boss": "champ", "seed": 1, "steps": steps, **kw}


def test_floor_score():
    assert floor_score(0, 0) == 0
    assert abs(floor_score(33, 2) - 1.0) < 1e-9
    assert floor_score(16, 0) < floor_score(17, 1)  # beating the act-1 boss is a step
    assert floor_score(33, 1) < floor_score(33, 2)
    assert floor_score(40, 2) == floor_score(33, 2)


def test_acts_cleared_and_bosses():
    steps = [{"kind": "start", "state": STATE, "boss": "hexaghost"},
             {"kind": "fight", "state": STATE, "category": "boss", "won": True, "boss": "hexaghost"},
             {"kind": "pick", "state": STATE, "options": [], "choice": 0, "boss": "hexaghost"},
             {"kind": "fight", "state": STATE, "category": "hard", "won": False, "boss": "champ"}]
    r = run("died", 20, steps)
    assert acts_cleared(r) == 1 and run_score(r, "floors") == floor_score(20, 1)
    assert acts_cleared({**r, "acts_cleared": 0}) == 0
    assert len(nodes(r)) == 3 and node_bosses(r) == ["hexaghost", "hexaghost", "hexaghost"]
    legacy = run("act_complete", 16, [{"kind": "start", "state": STATE}])
    assert node_bosses(legacy) == ["champ"] and run_score(legacy) == 1.0


def test_policy_exploration():
    p = Policy("simple", route_p=0.5)
    routes = {s: p.random_route({"seed": s}) for s in range(400)}
    assert 150 < sum(routes.values()) < 250
    assert all(p.random_route({"seed": s}) == v for s, v in routes.items())  # deterministic per seed
    s = next(k for k, v in routes.items() if v)
    msg = {"decision": "path", "options": [1, 2, 3], "after": [], "simple": 0, "seed": s, "boss": "champ"}
    assert p.decide(msg)[1] == "explore_route"
    assert p.decide({**msg, "decision": "rest"})[1] == "simple"
    q = Policy("random", eps=1.0)
    pick = {"options": [{}, {}], "skip_allowed": False, "simple": 0}
    assert all(q(pick)[0] < 2 for _ in range(200))
    t = Policy("simple", target="floors")
    assert t.terminal({"terminal": "cleared", "floor": 33}) == 1.0
    assert t.terminal({"terminal": "died", "floor": 20}) == floor_score(20, 1)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
