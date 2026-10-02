"""Learned after-state chooser and baseline policies for overworld decisions."""
import random
import threading
import torch
from .core import encode, floor_score, load_model


class Policy:
    """eps: per-decision uniform exploration (real decisions only; lookahead samples are greedy).
    route_p: fraction of runs (by seed) whose every path decision is uniform random (reaches elites / odd routes).
    target: how lookahead terminals are scored: act1 (clear 1, death progress * floor / 16) or floors (floor_score)."""

    def __init__(self, kind, ckpt=None, eps=0.0, seed=0, route_p=0.0, target="act1"):
        self.kind, self.eps, self.route_p, self.target, self.seed = kind, eps, route_p, target, seed
        self.rng = random.Random(seed)
        self.model = load_model(ckpt) if kind == "net" else None
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
            with torch.no_grad():
                b = encode([(msg["state"], msg["options"])], [msg["boss"]])
                v = torch.sigmoid(self.model(b)[0][0])  # [K+1]; K = max(n, 1), skip last
            vals = [v[i].item() for i in range(n)] + [v[-1].item()]
            return max(range(m), key=lambda i: vals[i]), "net", vals

    def terminal(self, e, progress=0.25):
        if self.target == "act1":
            return 1.0 if e["terminal"] == "cleared" else progress * min(e["floor"], 16) / 16
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
            if self.kind == "random" or (not msg.get("lookahead") and self.rng.random() < self.eps):
                return self.rng.randrange(n), "explore", None
            flat = [(i, e) for i, ends in enumerate(msg["ends"]) for e in ends]
            states = [e["state"] for _, e in flat if "state" in e]
            vals = []
            if states:
                with torch.no_grad():
                    b = encode([(st, []) for st in states], [msg["boss"]] * len(states))
                    vals = torch.sigmoid(self.model(b)[0][:, -1]).tolist()
            it = iter(vals)
            sums, counts = [0.0] * n, [0] * n
            for i, e in flat:
                v = next(it) if "state" in e else self.terminal(e, progress)
                sums[i] += v; counts[i] += 1
            means = [sums[i] / max(counts[i], 1) for i in range(n)]
            return max(range(n), key=lambda i: means[i]), "net", means

    def decide(self, msg):
        """rest / path / shop / boss_relic: one after-state per option; -1 = defer to SimpleAgent (only when simple is -1)."""
        n = len(msg["options"])
        with self.lock:
            if not msg.get("lookahead") and msg.get("decision") == "path" and self.random_route(msg):
                return self.rng.randrange(n), "explore_route", None
            if self.kind == "simple" or (self.kind == "random" and msg["simple"] == -1):
                return msg["simple"], "simple", None
            if (not msg.get("lookahead") and self.rng.random() < self.eps) or self.kind == "random":
                return self.rng.randrange(n), "explore", None
            with torch.no_grad():
                b = encode([(a, []) for a in msg["after"]], [msg["boss"]] * n)
                vals = torch.sigmoid(self.model(b)[0][:, -1]).tolist()  # skip column = V(after-state)
            return max(range(n), key=lambda i: vals[i]), "net", vals
