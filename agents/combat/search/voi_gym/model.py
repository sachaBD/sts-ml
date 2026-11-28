"""Finite expectimax policies, linear Gaussian measurements, exact one-step VOI."""
from dataclasses import dataclass
from itertools import product
from math import erfc, exp, inf, pi, sqrt

import numpy as np


@dataclass(frozen=True)
class Leaf:
    index: int


@dataclass(frozen=True)
class Choice:
    actions: dict


@dataclass(frozen=True)
class Chance:
    outcomes: dict  # observable label -> (probability, child)


def policies(node, size):
    """Enumerate feasible contingent plans, not clairvoyant choices of outcomes."""
    if isinstance(node, Leaf):
        reach = np.zeros(size)
        reach[node.index] = 1.
        yield {}, reach
    elif isinstance(node, Choice):
        for action, child in node.actions.items():
            for plan, reach in policies(child, size):
                yield {action: plan}, reach
    elif isinstance(node, Chance):
        labels = list(node.outcomes)
        weights, children = zip(*node.outcomes.values())
        if min(weights) < 0 or not np.isclose(sum(weights), 1):
            raise ValueError('invalid chance probabilities')
        for combination in product(*(policies(c, size) for c in children)):
            plans, reaches = zip(*combination)
            yield dict(zip(labels, plans)), weights @ np.stack(reaches)
    else:
        raise TypeError(f'unknown tree node: {node!r}')


@dataclass(frozen=True)
class Measurement:
    name: str
    h: np.ndarray  # observes h @ latent_values + zero-mean noise
    noise_var: float
    cost: int


def envelope_gain(intercepts, slopes):
    """E[max_i(b_i + m_i Z)] - max_i b_i, Z standard normal."""
    hull = []
    for m, b in sorted(zip(slopes, intercepts)):
        while hull:
            start, old_m, old_b = hull[-1]
            if m != old_m:
                cross = (old_b - b) / (m - old_m)
                if cross > start:
                    break
            hull.pop()
        hull.append((cross if hull else -inf, m, b))
    # Each kink contributes a call option. Subtract its intrinsic value at Z=0.
    gain = 0.
    for (_, prev, _), (x, m, _) in zip(hull, hull[1:]):
        t = abs(float(x))
        gain += (m - prev) * (exp(-0.5*t*t)/sqrt(2*pi) - t*erfc(t/sqrt(2))/2)
    return max(0., float(gain))


class Belief:
    """Gaussian scale mixture: regimes share a mean and Kalman gain, not confidence.

    In regime j, both latent covariance and measurement noise scale by s[j]^2.
    A large standardized residual is evidence for the less predictable regime.
    """
    def __init__(self, means, cov, scales=(1., 3.), probabilities=(.9, .1)):
        self.mean = np.array(means, dtype=float)
        self.cov = np.array(cov, dtype=float)
        self.scales = np.array(scales, dtype=float)
        self.log_weights = np.log(np.array(probabilities, dtype=float))
        if (self.scales.shape != self.log_weights.shape or min(self.scales) <= 0 or
                not np.isfinite(self.log_weights).all() or not np.isclose(sum(probabilities), 1)):
            raise ValueError('invalid scale mixture')

    def prediction(self, m):
        ph = self.cov @ m.h
        sd = sqrt(m.h @ ph + m.noise_var)
        return float(m.h @ self.mean), sd, ph / sd

    def observe(self, m, y):
        center, sd, shift = self.prediction(m)
        z = (y - center) / sd
        self.mean += shift * z
        self.cov -= np.outer(shift, shift)
        self.log_weights -= np.log(self.scales) + .5*(z/self.scales)**2
        self.log_weights -= np.logaddexp.reduce(self.log_weights)

    def voi(self, m, reaches):
        _, _, shift = self.prediction(m)
        intercepts, slopes = reaches @ self.mean, reaches @ shift
        return sum(float(w)*envelope_gain(intercepts, slopes*s)
                   for w, s in zip(np.exp(self.log_weights), self.scales))


def rank_measurements(belief, reaches, measurements, price, remaining):
    rows = []
    for m in measurements:
        gain = belief.voi(m, reaches)
        rows.append(dict(id=m.name, gain=gain, cost=m.cost, net=gain-price*m.cost,
                         affordable=m.cost <= remaining))
    return sorted(rows, key=lambda r: r['net'], reverse=True)


def buy(rows):
    return next((r['id'] for r in rows if r['affordable'] and r['net'] > 0), None)
