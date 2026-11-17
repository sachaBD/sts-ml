#include "agents/combat/player/budget.hpp"
#include <stdexcept>

namespace stsrl::player {
void PlayoutCap::validate() const {
    if (full < 0 || cheap < 0 || !(full_share >= 0 && full_share <= 1))
        throw std::invalid_argument{"budget: nonnegative full and cheap simulations, full share in [0, 1]"};
}

Budget PlayoutCap::operator()(const Moment& moment, std::mt19937_64& random) const {
    if (moment.after_chance || std::bernoulli_distribution{full_share}(random)) return {full, true};
    return {cheap, false};
}
}  // namespace stsrl::player
