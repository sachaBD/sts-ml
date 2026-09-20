#pragma once

#include <cstddef>
#include <span>
#include <string>

namespace stsrl {

struct CombatObservation {
    int turn{};
    int player_hp{};
    int player_max_hp{};
    int player_block{};
    int energy{};
    int enemy_hp{};
    int enemy_max_hp{};
    int enemy_block{};
};

struct LegalAction {
    std::size_t index{};
    std::string description;
};

class Agent {
public:
    virtual ~Agent() = default;

    [[nodiscard]] virtual std::size_t choose_action(
        const CombatObservation& observation,
        std::span<const LegalAction> legal_actions) = 0;
};

}  // namespace stsrl
