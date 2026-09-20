#include "agents/random_agent.hpp"

#include <stdexcept>

namespace stsrl {

RandomAgent::RandomAgent(const std::uint64_t seed) : random_{seed} {}

std::size_t RandomAgent::choose_action(
    [[maybe_unused]] const CombatObservation& observation,
    const std::span<const LegalAction> legal_actions) {
    if (legal_actions.empty()) {
        throw std::logic_error{"RandomAgent was asked to choose from no actions"};
    }

    std::uniform_int_distribution<std::size_t> distribution{
        0, legal_actions.size() - 1};
    return distribution(random_);
}

}  // namespace stsrl
