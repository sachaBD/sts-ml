"""Learned after-state chooser and baseline policies for overworld decisions."""
import random
import threading
import torch
from .core import encode, load_model


class Policy:
    def __init__(self, kind, ckpt=None, eps=0.0, seed=0):
        self.kind, self.eps = kind, eps
        self.rng = random.Random(seed)
        self.model = load_model(ckpt) if kind == "net" else None
        self.lock = threading.Lock()

    def __call__(self, msg):
        n = len(msg["options"])
        eps = 0.0 if msg.get("lookahead") else self.eps  # samples play the greedy policy
        with self.lock:
            if self.rng.random() < eps:
                return self.rng.randrange(n + 1), "explore", None
            if self.kind == "simple":
                return msg["simple"], "simple", None
            if self.kind == "random":
                return self.rng.randrange(n + 1), "random", None
            with torch.no_grad():
                b = encode([(msg["state"], msg["options"])], [msg["boss"]])
                v = torch.sigmoid(self.model(b)[0][0])  # [K+1]; K = max(n, 1), skip last
            vals = [v[i].item() for i in range(n)] + [v[-1].item()]
            return max(range(n + 1), key=lambda i: vals[i]), "net", vals

    def evaluate(self, msg, progress=0.25):
        """Sampled lookahead (events): option value = mean over its samples of V(end state); a death scores
        progress * floor / 16 and a clear 1 (the run score). Explores with eps like other decisions."""
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
                if "state" in e:
                    v = next(it)
                elif e["terminal"] == "cleared":
                    v = 1.0
                else:
                    v = progress * min(e["floor"], 16) / 16
                sums[i] += v; counts[i] += 1
            means = [sums[i] / max(counts[i], 1) for i in range(n)]
            return max(range(n), key=lambda i: means[i]), "net", means

    def decide(self, msg):
        """rest / path: one after-state per option; -1 = defer to SimpleAgent (only when simple is -1)."""
        n = len(msg["options"])
        with self.lock:
            if self.kind == "simple" or (self.kind == "random" and msg["simple"] == -1):
                return msg["simple"], "simple", None
            if (not msg.get("lookahead") and self.rng.random() < self.eps) or self.kind == "random":
                return self.rng.randrange(n), "explore", None
            with torch.no_grad():
                b = encode([(a, []) for a in msg["after"]], [msg["boss"]] * n)
                vals = torch.sigmoid(self.model(b)[0][:, -1]).tolist()  # skip column = V(after-state)
            return max(range(n), key=lambda i: vals[i]), "net", vals
