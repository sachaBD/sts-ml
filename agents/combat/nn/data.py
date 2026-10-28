"""Self-play samples (nn_play SAMPLES.bin) as dense tensors.

Record: int32 tokens, int32 actions, float32 won, float32 tokens x WIDTH, float32 policy[actions] (the last tokens).
"""
import numpy as np
import torch

from .model import WIDTH


def read(paths, max_rows=0):
    """(tokens [N, T, WIDTH] float16, policy [N, T] float32, won [N] float32), padded to the longest decision."""
    records = []
    for path in paths:
        # Replay can span gigabytes: let the OS evict file pages while we build the padded tensors.
        raw = np.memmap(path, dtype=np.uint8, mode='r')
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
    if max_rows and len(records) > max_rows:
        # Use checkpointed torch RNG; every replay row has equal inclusion probability.
        indices = torch.randperm(len(records))[:max_rows].tolist()
        records = [records[i] for i in indices]
    longest = max(len(t) for t, _, _ in records)
    tokens = np.zeros((len(records), longest, WIDTH), np.float16)
    policy = np.zeros((len(records), longest), np.float32)
    for i, (t, p, _) in enumerate(records):
        tokens[i, :len(t)] = t
        policy[i, len(t) - len(p):len(t)] = p
    won = np.array([w for _, _, w in records], np.float32)
    return torch.from_numpy(tokens), torch.from_numpy(policy), torch.from_numpy(won)
