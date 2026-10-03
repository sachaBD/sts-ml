"""Learned after-state chooser and baseline policies for overworld decisions."""
import random
import threading
import torch
from .core import encode, floor_score, floor_score3, full_terminal, load_model


class Policy:
    """eps: per-decision uniform exploration (real decisions only; lookahead samples are greedy).
    route_p: fraction of runs (by seed) whose every path decision is uniform random (reaches elites / odd routes).
    target: lookahead scoring: act1 (clear 1, death progress * floor / 16), floors (2 acts), floors3 (3 acts)."""

    def __init__(self, kind, ckpt=None, eps=0.0, seed=0, route_p=0.0, target="act1", key_rule=False, key_rule_p=None,
                 late_ckpt=None, late_act=3):
        self.kind, self.eps, self.route_p, self.target, self.seed = kind, eps, route_p, target, seed
        self.key_rule_p = 1.0 if key_rule else (key_rule_p or 0.0)  # per-run fraction; key_rule = 1
        self.key_rule = key_rule  # Heart-mode crutch for key-blind nets (emerald routing here; ruby in the worker)
        self.rng = random.Random(seed)
        self.model = load_model(ckpt) if kind == "net" else None
        # Optional second model for decisions from Act `late_act` on (e.g. an Act-2 specialist early, a full-game model late).
        self.late, self.late_act = (load_model(late_ckpt) if late_ckpt else None), late_act
        self.lock = threading.Lock()

    def random_route(self, msg):
        """Whether this run (msg["seed"]) follows a random route: a fixed function of seed, so resumes agree."""
        return self.route_p > 0 and "seed" in msg and random.Random(msg["seed"] * 7919 + self.seed).random() < self.route_p

    def __call__(self, msg):
        n = len(msg["options"])
        m = n + 1 if msg.get("skip_allowed", True) else n  # choice n = skip
        eps = 0.0 if msg.get("lookahead") else self.eps  # samples play the greedy policy
        with self.lock:
            if self.rng.random() < eps:
                return self.rng.randrange(m), "explore", None
            if self.kind == "simple":
                return msg["simple"], "simple", None
            if self.kind == "random":
                return self.rng.randrange(m), "random", None
            net = self.pick_model(msg)
            with torch.no_grad():
                b = encode([(msg["state"], msg["options"])], [msg["boss"]], kind=net.KIND)
                v = torch.sigmoid(net(b)[0][0])  # [K+1]; K = max(n, 1), skip last
            vals = [v[i].item() for i in range(n)] + [v[-1].item()]
            return max(range(m), key=lambda i: vals[i]), "net", vals

    def terminal(self, e, progress=0.25):
        if self.target == "heart":
            return float(e.get("heart_cleared", False))
        if self.target == "full":
            return full_terminal(e)
        if self.target == "act1":
            return 1.0 if e["terminal"] == "cleared" else progress * min(e["floor"], 16) / 16
        if self.target == "floors3":
            default_cleared = 3 if e["terminal"] == "cleared" else int(e["floor"] > 16) + int(e["floor"] > 33)
            return floor_score3(e["floor"], e.get("acts_cleared", default_cleared))
        if e["terminal"] == "cleared":
            return floor_score(e["floor"], e.get("acts_cleared", 2))
        return floor_score(e["floor"], e.get("acts_cleared", 1 if e["floor"] > 16 else 0))

    def evaluate(self, msg, progress=0.25):
        """Sampled lookahead (events): option value = mean over its samples of V(end state); terminals scored by
        self.terminal. Explores with eps like other decisions."""
        n = len(msg["options"])
        with self.lock:
            if self.kind == "simple":
                return max(msg["simple"], 0), "simple", None
            explore_ok = not msg.get("lookahead") and msg.get("decision") != "rest_lookahead"  # refines a greedy choice
            if self.kind == "random" or (explore_ok and self.rng.random() < self.eps):
                return self.rng.randrange(n), "explore", None
            flat = [(i, e) for i, ends in enumerate(msg["ends"]) for e in ends]
            states = [e["state"] for _, e in flat if "state" in e]
            vals = []
            if states:
                with torch.no_grad():
                    net = self.pick_model(msg, states[0])
                    b = encode([(st, []) for st in states], [msg["boss"]] * len(states), kind=net.KIND)
                    vals = torch.sigmoid(net(b)[0][:, -1]).tolist()
            it = iter(vals)
            sums, counts = [0.0] * n, [0] * n
            for i, e in flat:
                v = next(it) if "state" in e else self.terminal(e, progress)
                sums[i] += v; counts[i] += 1
            means = [sums[i] / max(counts[i], 1) for i in range(n)]
            return max(range(n), key=lambda i: means[i]), "net", means

    def pick_model(self, msg, fallback_state=None):
        """The model for this decision: the late model from Act late_act on (by the decision's state), else the main one."""
        if self.late is None:
            return self.model
        state = msg.get("state") or fallback_state or (msg.get("after") or [{}])[0]
        return self.late if state.get("act", 1) >= self.late_act else self.model

    def key_rule_for(self, seed):
        """Whether the key rule is on for this run: a fixed function of seed, so resumes agree (None seed: p >= 1 only)."""
        return key_rule_on(seed, self.key_rule_p)

    def decide(self, msg):
        """rest / path / shop / boss_relic: one after-state per option; -1 = defer to SimpleAgent (only when simple is -1)."""
        n = len(msg["options"])
        with self.lock:
            if not msg.get("lookahead") and msg.get("decision") == "path" and self.random_route(msg):
                return self.rng.randrange(n), "explore_route", None
            allowed = emerald_options(msg) if self.key_rule_p > 0 and msg.get("decision") == "path" and self.key_rule_for(msg.get("seed")) else None
            if allowed is not None and len(allowed) < n and self.kind == "net":
                with torch.no_grad():
                    net = self.pick_model(msg)
                    b = encode([(msg["after"][i], []) for i in allowed], [msg["boss"]] * len(allowed), kind=net.KIND)
                    vals = torch.sigmoid(net(b)[0][:, -1]).tolist()
                return allowed[max(range(len(allowed)), key=lambda j: vals[j])], "key_rule", None
            if self.kind == "simple" or (self.kind == "random" and msg["simple"] == -1):
                return msg["simple"], "simple", None
            if (not msg.get("lookahead") and self.rng.random() < self.eps) or self.kind == "random":
                return self.rng.randrange(n), "explore", None
            with torch.no_grad():
                net = self.pick_model(msg)
                b = encode([(a, []) for a in msg["after"]], [msg["boss"]] * n, kind=net.KIND)
                vals = torch.sigmoid(net(b)[0][:, -1]).tolist()  # skip column = V(after-state)
            return max(range(n), key=lambda i: vals[i]), "net", vals


def key_rule_on(seed, p):
    if p >= 1:
        return True
    return p > 0 and seed is not None and random.Random(seed * 104729 + 17).random() < p


def emerald_options(msg):
    """Path options (indices) from which the burning elite is still reachable, in Act 3 only (ah-c01: Act 2 forcing cost ~6 pp Act 2 clears) while the emerald key is
    missing; None when the rule does not apply (no burning elite ahead, key owned, Act 1, or no option reaches it)."""
    state = msg.get("state") or {}
    ow, m = state.get("overworld", {}), state.get("map", {})
    burning = m.get("burning_elite")
    if not burning or ow.get("keys", {}).get("emerald", True) or state.get("act", 1) != 3 or "nodes" not in m:
        return None
    edges = {(nd["x"], nd["y"]): nd["edges"] for nd in m["nodes"]}
    y0 = m.get("current", {}).get("y", -1) + 1
    goal = (burning["x"], burning["y"])

    def reaches(x):
        todo, seen = [(x, y0)], set()
        while todo:
            node = todo.pop()
            if node == goal:
                return True
            if node in seen or node not in edges or node[1] >= goal[1]:
                continue
            seen.add(node)
            todo += [(z, node[1] + 1) for z in edges[node]]
        return False
    ok = [i for i, a in enumerate(msg["after"]) if any(reaches(x) for x in a["map"].get("next_xs", []))]
    return ok or None
