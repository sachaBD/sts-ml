"""Fresh-seeded copies of combat starts (combat_v4 `start` objects), seeded the way the simulator seeds them."""
import copy
import hashlib

MASK = (1 << 64) - 1


def rng_state(seed):
    """sts::Random(seed) before any draw (libgdx RandomXS128 seeding)."""
    def murmur(x):
        x ^= x >> 33; x = x * (-49064778989728563 & MASK) & MASK
        x ^= x >> 33; x = x * (-4265267296055464877 & MASK) & MASK
        return (x ^ (x >> 33)) & MASK
    first = murmur(seed if seed else 1 << 63)
    return {'counter': 0, 'seed0': first, 'seed1': murmur(first)}


def fights(base, name, n):
    """n copies of base ({fight_id, start}, or a list of them) with fresh seeds derived from name."""
    if isinstance(base, list):
        if not base or n % len(base):
            raise ValueError('games must be a multiple of the nonempty start list length')
        names = [f['fight_id'] for f in base]
        if len(set(names)) != len(names):
            raise ValueError('start list fight_ids must be unique')
        # Interleave decks so collection/draining does not run one deck at a time.
        decks = [fights(f, f'{name}:{f["fight_id"]}', n // len(base)) for f in base]
        return [decks[d][i] for i in range(n // len(base)) for d in range(len(base))]
    out = []
    for i in range(n):
        seed = int.from_bytes(hashlib.sha256(f'{name}:{i}'.encode()).digest()[:8], 'little')
        f = copy.deepcopy(base)
        f['fight_id'] = f'{name}:{i}'
        f['start'].update(seed=seed, misc_rng=rng_state(seed), potion_rng=rng_state(seed))
        out.append(f)
    return out
