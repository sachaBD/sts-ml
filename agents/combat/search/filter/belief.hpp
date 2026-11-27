#pragma once
#include <algorithm>
#include <cmath>
#include <numbers>

namespace stsrl::search::filter {
// Scalar static-state Kalman update in information form. Refit measurement noise from paired
// world differences, with one prior variance unit until empirical scatter is available.
struct Gap {
    double prior = 0, prior_var = 0.0025;
    int n = 0;
    double average = 0, scatter = 0;
    void observe(double y) {
        const double delta = y - average;
        average += delta / ++n;
        scatter += delta * (y - average);
    }
    double noise() const { return (prior_var + scatter) / std::max(1, n); }
    double variance() const { return 1 / (1 / prior_var + n / noise()); }
    double mean() const { return variance() * (prior / prior_var + n * average / noise()); }
    // SD of the next posterior mean, NOT posterior SD; only reducible uncertainty pays.
    double innovation() const {
        const double p = variance();
        return p / std::sqrt(p + noise());
    }
};
inline double positive_part(double m, double sd) {
    return std::max(0., .5 * m * std::erfc(-m / (sd * std::numbers::sqrt2)) +
                            sd / std::sqrt(2 * std::numbers::pi) * std::exp(-.5 * (m / sd) * (m / sd)));
}
// E[max(0,b+d*max(0,Y))], Y normal. Only use the interior alternative when its mean is positive.
inline double branch_option(double b, double d, double m, double sd) {
    if (d == 0)
        return std::max(0., b);
    if (d > 0)
        return b >= 0 ? b + d * positive_part(m, sd) : d * positive_part(m + b / d, sd);
    return b <= 0 ? 0. : std::max(0., b + d * (positive_part(m, sd) - positive_part(m + b / d, sd)));
}
}
