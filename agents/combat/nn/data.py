"""Self-play samples (nn_play SAMPLES.bin) as dense tensors.

Record: int32 tokens, int32 actions, float32 won, float32 tokens x WIDTH, float32 policy[actions] (the last tokens).
"""
import os

import numpy as np
import torch

from .model import WIDTH


def read(paths):
    """(tokens [N, T, WIDTH] float16, policy [N, T] float32, won [N] float32), padded to the longest decision."""
    records = []
    for path in paths:
        raw = np.fromfile(path, np.uint8)
        at = 0
        while at < len(raw):
            count, actions = raw[at:at + 8].view(np.int32)
            won = raw[at + 8:at + 12].view(np.float32)[0]
            at += 12
            tokens = raw[at:at + 4 * count * WIDTH].view(np.float32).reshape(count, WIDTH)
            at += 4 * count * WIDTH
            policy = raw[at:at + 4 * actions].view(np.float32)
            at += 4 * actions
            records.append((tokens, policy, won))
    longest = max(len(t) for t, _, _ in records)
    tokens = np.zeros((len(records), longest, WIDTH), np.float16)
    policy = np.zeros((len(records), longest), np.float32)
    for i, (t, p, _) in enumerate(records):
        tokens[i, :len(t)] = t
        policy[i, len(t) - len(p):len(t)] = p
    won = np.array([w for _, _, w in records], np.float32)
    return torch.from_numpy(tokens), torch.from_numpy(policy), torch.from_numpy(won)


class Replay:
    """read() over a sliding window that reads each file once. A file is keyed by (path, size, mtime), so a changed
    file is read again; files that leave the window are dropped."""
    def __init__(self):
        self.files = {}

    def read(self, paths):
        keys = [(str(p), os.stat(p).st_size, os.stat(p).st_mtime_ns) for p in paths]
        self.files = {k: self.files[k] if k in self.files else read([k[0]]) for k in keys if k[1]}
        parts = [self.files[k] for k in keys if k[1]]
        longest = max(t.shape[1] for t, _, _ in parts)  # ValueError with no samples, like read()
        tokens = torch.zeros(sum(len(w) for _, _, w in parts), longest, WIDTH, dtype=torch.float16)
        policy = torch.zeros(len(tokens), longest)
        i = 0
        for t, p, w in parts:
            tokens[i:i + len(w), :t.shape[1]], policy[i:i + len(w), :t.shape[1]] = t, p
            i += len(w)
        return tokens, policy, torch.cat([w for _, _, w in parts])
