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


def seeded(base, fight_id):
    """A copy of base ({fight_id, start}) named fight_id, with a fresh seed derived from that name."""
    seed = int.from_bytes(hashlib.sha256(fight_id.encode()).digest()[:8], 'little')
    f = copy.deepcopy(base)
    f['fight_id'] = fight_id
    f['start'].update(seed=seed, misc_rng=rng_state(seed), potion_rng=rng_state(seed))
    return f


def fights(base, name, n):
    """n copies of base with fresh seeds derived from name."""
    return [seeded(base, f'{name}:{i}') for i in range(n)]
