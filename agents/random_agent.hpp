#pragma once

#include "agents/agent.hpp"

#include <cstdint>
#include <random>

namespace stsrl {

class RandomAgent final : public Agent {
public:
    explicit RandomAgent(std::uint64_t seed);

    [[nodiscard]] std::size_t choose_action(
        const CombatObservation& observation,
        std::span<const LegalAction> legal_actions) override;

private:
    std::mt19937_64 random_;
};

}  // namespace stsrl
