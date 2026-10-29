"""App-owned result parsing and statistics shared by the terminal and web dashboards."""
import json
import math

Z = 1.96


def jsonl(path):
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # live writer may be midway through its last line
    return rows


def deck_of(fight_id):
    parts = fight_id.split(':')  # training: ns:iter:deck:i; evaluation: ns:deck:split:i
    return parts[2] if parts[1].isdigit() else parts[1]


def wilson(w, n):
    if not n:
        return [0, 0]
    p, d = w / n, 1 + Z * Z / n
    c, h = (p + Z * Z / (2 * n)) / d, Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return [c - h, c + h]


def rate(w, n):
    return dict(w=w, n=n, p=w / n if n else None, ci=wilson(w, n))


def outcomes(path):
    return {r['fight_id']: bool(r['won']) for r in jsonl(path)}


def by_deck(results):
    decks = {}
    for k, won in results.items():
        decks.setdefault(deck_of(k), []).append(won)
    out = {d: rate(sum(v), len(v)) for d, v in sorted(decks.items())}
    out['all'] = rate(sum(results.values()), len(results))
    return out


def mean_bounds(values):
    mean = sum(values) / len(values)
    se = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) * (len(values) - 1))) if len(values) > 1 else None
    return mean, [mean - Z * se, mean + Z * se] if se is not None else None


def unpaired(new, old):
    if not new['n'] or not old['n']:
        return None  # held-out-only decks have no collection trend
    p1, n1, p2, n2 = new['p'], new['n'], old['p'], old['n']
    d, se = p1 - p2, math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return dict(d=d, lo=d - Z * se, hi=d + Z * se)


def paired(new, old, keys):
    if len(keys) < 2:
        return None  # insufficient pairs to estimate uncertainty
    gap, bounds = mean_bounds([int(new[k]) - int(old[k]) for k in keys])
    return dict(d=gap, lo=bounds[0], hi=bounds[1])


def score(results, baselines):
    n, wins = len(results), sum(results.values())
    comparisons = {}
    for name, reference in baselines.items():
        keys = results.keys() & reference.keys()
        if not keys:
            continue
        differences = [int(results[k]) - int(reference[k]) for k in keys]
        m = len(keys)
        gap, bounds = mean_bounds(differences)
        grouped = {}
        for k in keys:
            grouped.setdefault(deck_of(k), []).append(int(results[k]) - int(reference[k]))
        macro, cluster_ci = mean_bounds([sum(v) / len(v) for v in grouped.values()])
        comparisons[name] = dict(n=m, wins=sum(results[k] for k in keys), baseline_wins=sum(reference[k] for k in keys),
                                 gap=gap, gap_95=bounds,
                                 families=len(grouped), macro_gap=macro, family_gap_95=cluster_ci)
    grouped = {}
    for k, won in results.items():
        grouped.setdefault(deck_of(k), []).append(won)
    macro, cluster_ci = mean_bounds([sum(v) / len(v) for v in grouped.values()])
    return dict(n=n, wins=wins, win_rate=wins / n, wilson_95=wilson(wins, n), comparisons=comparisons,
                families=len(grouped), macro_win_rate=macro, family_95=cluster_ci)


def baselines(out):
    path, reference = out / 'inputs/baselines.json', out / 'inputs/reference.json'
    values = json.loads(path.read_text()) if path.exists() else {}
    if not values and reference.exists():
        values['mcts20k'] = json.loads(reference.read_text())
    return values


def pool(batches, deck):
    w = sum(b['rates'][deck]['w'] for b in batches if deck in b['rates'])
    n = sum(b['rates'][deck]['n'] for b in batches if deck in b['rates'])
    return rate(w, n)
