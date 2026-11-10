#pragma once

#include "environments/combat/encoding.hpp"

#include <compare>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sts {
struct BattleContext;
}

namespace stsrl {

struct LegalAction {
    std::size_t index{};
    std::string description;
};

struct Decision {
    EncodedCombatState encoding;
    std::vector<LegalAction> legal_actions;
};

enum class CardCoverage { legacy_act1, ironclad_events };
struct CardLocation {
    CardZone zone;
    int index;
    auto operator<=>(const CardLocation&) const = default;
};

// Sorted features; optional locations map rows back to simulator storage, not model inputs.
[[nodiscard]] std::vector<CardToken> encode_cards(const sts::BattleContext&, CardCoverage,
                                                 std::vector<CardLocation>* locations = nullptr);
[[nodiscard]] std::vector<MonsterToken> encode_monsters(const sts::BattleContext&, std::vector<int>* slots = nullptr);
[[nodiscard]] int incoming_damage(const sts::BattleContext&);

// The state part of Decision::encoding (all but legal_actions): what the value net reads.
// monster_slots, when requested, maps the sorted monster tokens back to simulator slots.
[[nodiscard]] EncodedCombatState encode_state(const sts::BattleContext& state,
                                             CardCoverage coverage = CardCoverage::legacy_act1,
                                             std::vector<int>* monster_slots = nullptr);
// The legal action `action_bits` at `state` as a policy token (legal_actions of Decision::encoding;
// execution_index is the caller's). Search edges are encoded with it too.
[[nodiscard]] ActionToken encode_action(const sts::BattleContext& state, std::uint32_t action_bits,
                                        std::size_t execution_index,
                                        CardCoverage coverage = CardCoverage::legacy_act1);

class CombatEnvironment final {
public:
    explicit CombatEnvironment(sts::BattleContext initial_state);
    ~CombatEnvironment();

    CombatEnvironment(CombatEnvironment&&) noexcept;
    CombatEnvironment& operator=(CombatEnvironment&&) noexcept;
    CombatEnvironment(const CombatEnvironment&) = delete;
    CombatEnvironment& operator=(const CombatEnvironment&) = delete;

    [[nodiscard]] Decision decision();
    // decision()'s legal actions (same order, same indices for step / action_bits) without the encoding or
    // descriptions: their count. For players that record nothing.
    std::size_t legal_action_count();
    [[nodiscard]] std::string action_description(std::size_t action_index) const;
    void step(std::size_t action_index);
    [[nodiscard]] bool done() const noexcept;
    [[nodiscard]] bool won() const noexcept;
    [[nodiscard]] int player_hp() const noexcept;
    [[nodiscard]] int player_max_hp() const noexcept;
    // Raw simulator access for external teachers (e.g. sts_ml's search).
    [[nodiscard]] const sts::BattleContext& battle() const noexcept;
    [[nodiscard]] std::uint32_t action_bits(std::size_t action_index) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace stsrl
