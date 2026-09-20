#pragma once

#include "agents/agent.hpp"
#include "combat/encoding.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace sts {
struct BattleContext;
}

namespace stsrl {

struct SearchActionKey {
    int kind{}; int card_id{}; int upgraded{}; int cost{}; int cost_for_turn{}; int special{}; int free{}; int retain{};
    int target_id{}; int target_hp{}; int target_max_hp{}; int target_block{}; int target_move{}; int target_strength{}; int target_weak{}; int target_vulnerable{}; int selection_task{}; int selected{}; int potion{};
    auto operator==(const SearchActionKey&) const -> bool = default;
};
struct SearchAction { std::size_t index{}; SearchActionKey key{}; };

struct Decision {
    CombatObservation observation;
    EncodedCombatState encoding;
    std::vector<LegalAction> legal_actions;
};

class CombatEnvironment final {
public:
    explicit CombatEnvironment(sts::BattleContext initial_state);
    ~CombatEnvironment();

    CombatEnvironment(CombatEnvironment&&) noexcept;
    CombatEnvironment& operator=(CombatEnvironment&&) noexcept;
    CombatEnvironment(const CombatEnvironment&) = delete;
    CombatEnvironment& operator=(const CombatEnvironment&) = delete;

    [[nodiscard]] Decision decision();
    [[nodiscard]] std::vector<SearchAction> search_actions();
    [[nodiscard]] CombatEnvironment determinized(std::uint64_t seed) const;
    void step(std::size_t action_index);
    [[nodiscard]] bool done() const noexcept;
    [[nodiscard]] bool won() const noexcept;
    [[nodiscard]] double combat_value() const noexcept;
    [[nodiscard]] int player_hp() const noexcept;
    [[nodiscard]] int player_max_hp() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace stsrl
