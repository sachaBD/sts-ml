"""Synthetic checks for continuation-label estimators (no binaries, no compute)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import continuation as T  # noqa: E402


def test_debiased_sq_error_is_unbiased():
    rng = np.random.default_rng(0)
    for v, t in [(.3, .7), (.9, .9), (.5, .1)]:
        wins = rng.binomial(4, t, 400000)
        assert abs(T.debiased_sq_error(v, wins, 4).mean() - (v - t) ** 2) < 2e-3


def test_summarize_labels_paired_gap():
    xs = [dict(update15_value=50.0, learner_won=i % 2 == 0) for i in range(4)]
    labels = [dict(playouts=[dict(won=j < 3) for j in range(4)]) for _ in range(4)]
    s = T.summarize_labels(xs, labels)
    assert abs(s['teacher_continuation']['mean'] - .75) < 1e-12 and abs(s['teacher_minus_learner']['mean'] - .25) < 1e-12
    assert abs(s['teacher_minus_value']['mean'] - .25) < 1e-12



def test_infer_known_prefix():
    v = lambda kind, n: dict(kind=kind, draw=[0] * n)
    # play, headbutt select (+1 on top), end turn draws 5 (consumes known), headbutt again, partial draw of 0
    views = [v('play', 10), v('headbutt', 10), v('play', 11), v('play', 6), v('headbutt', 6), v('play', 7), v('play', 7)]
    ks, anomalies = T.infer_known_prefix(views)
    assert ks == [0, 0, 1, 0, 0, 1, 1] and not anomalies


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'): fn(); print('ok', name)
