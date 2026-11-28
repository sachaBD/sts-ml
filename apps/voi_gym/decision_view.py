import numpy as np

from agents.combat.search.voi_gym.root import DEFAULT_INITIAL_SD_CAP, DEFAULT_PRIOR_SCALE


class DecisionView:
    def __init__(self, decision, labels, root_value, prior, budget, price, prior_scale=DEFAULT_PRIOR_SCALE,
                 initial_sd_cap=DEFAULT_INITIAL_SD_CAP):
        self.decision, self.labels = decision, labels
        self.root_value, self.prior = root_value, prior
        self.budget, self.price = budget, price
        self.prior_scale, self.initial_sd_cap = prior_scale, initial_sd_cap

    def snapshot(self, spent, q_calls, s_calls):
        d = self.decision
        b = d.belief
        weights = np.exp(b.log_weights)
        scale_variance = float(weights @ b.scales**2)
        return dict(spent=spent, budget=self.budget, price=self.price,
                    move=self.labels[d.move()], root_q={self.labels[a]: float(v) for a, v in zip(d.actions, b.mean)},
                    actions=[self.labels[a] for a in d.actions],
                    root_value=self.root_value, prior_scale=self.prior_scale, initial_sd_cap=self.initial_sd_cap,
                    prior={a: p for a, p in zip(self.labels, self.prior)},
                    scales=b.scales.tolist(), weights=weights.tolist(), covariance=b.cov.tolist(),
                    sd={self.labels[a]: float(v) for a, v in zip(d.actions, np.sqrt(scale_variance*b.cov.diagonal()))},
                    q_calls=q_calls, s_calls=s_calls)

    def candidates(self, spent):
        d = self.decision
        b = d.belief
        rows = d.candidates(self.price, self.budget-spent)
        for row in rows:
            m = d.measurement(row['id'])
            left, right = d.pair(m)
            center, base_sd, shift = b.prediction(m)
            mixture_sd = base_sd*np.sqrt(np.exp(b.log_weights) @ b.scales**2)
            row.update(left=self.labels[left], right=self.labels[right], predicted=center, base_sd=base_sd,
                       predictive_sd=float(mixture_sd), noise_var=m.noise_var,
                       shift=shift.tolist(), gains=b.gain_by_scale(m, d.reaches))
        return rows
