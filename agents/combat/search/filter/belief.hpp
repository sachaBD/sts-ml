#pragma once
#include <algorithm>
#include <cmath>
#include <numbers>
#include <span>
#include <stdexcept>

namespace stsrl::search::filter {
// Policy value temperature (win probability per unit logit): converts policy logit gaps into Kalman priors.
inline constexpr double kPolicyTemp = 0.03;

// Scalar static-state Kalman update in information form. Refit measurement noise from
// observed world outcomes, with one prior variance unit until empirical scatter is available.
struct Belief {
    double mean = 0, variance = 0, noise = 0, innovation = 0;
    double weight() const { return noise > 0 ? variance / noise : 0.; }
};

inline Belief kalman(double prior, double prior_var, std::span<const double> ys, double obs_prior_var = -1.) {
    if (prior_var <= 0)
        return {prior, 0., 0., 0.};
    const int n = int(ys.size());
    double average = 0, scatter = 0;
    for (int i = 0; i < n; ++i) {
        const double delta = ys[i] - average;
        average += delta / (i + 1);
        scatter += delta * (ys[i] - average);
    }
    const double r0 = obs_prior_var > 0 ? obs_prior_var : prior_var;
    const double noise = (r0 + scatter) / std::max(1, n);
    const double variance = 1. / (1. / prior_var + n / noise);
    const double mean = variance * (prior / prior_var + n * average / noise);
    // SD of the next posterior mean, NOT posterior SD; only reducible uncertainty pays.
    const double innovation = variance / std::sqrt(variance + noise);
    return {mean, variance, noise, innovation};
}

// E[max(0, N(m, sd^2))]: exact expected improvement above 0.
inline double positive_part(double m, double sd) {
    if (sd <= 0)
        return std::max(0., m);
    return std::max(0., .5 * m * std::erfc(-m / (sd * std::numbers::sqrt2)) +
                            sd / std::sqrt(2 * std::numbers::pi) * std::exp(-.5 * (m / sd) * (m / sd)));
}

// Node uncertainty SD: exact Gaussian SD matching NN thinking gain (E[max(0,N(0,s^2))] = s/sqrt(2*pi))
// with a Bernoulli floor, capped by distance to 0% / 100%.
inline double prior_sd(double value, std::span<const float> thinking) {
    double best = 0;
    for (float g : thinking) {
        if (!std::isfinite(g))
            throw std::invalid_argument{"filter: nonfinite thinking output"};
        best = std::max(best, double(g));
    }
    const double room = std::max(0., std::min(value, 1. - value));
    const double floor = 0.005 * std::sqrt(std::max(0., value * (1. - value)));
    return std::min(2. * room, std::max(floor, std::sqrt(2. * std::numbers::pi) * best));
}
}
