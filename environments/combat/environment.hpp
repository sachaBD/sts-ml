#pragma once

#include "environments/combat/encoding.hpp"

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

// The state part of Decision::encoding (all but legal_actions): what the value net reads.
enum class CardCoverage { legacy_act1, ironclad_events };
[[nodiscard]] EncodedCombatState encode_state(const sts::BattleContext& state,
                                             CardCoverage coverage = CardCoverage::legacy_act1);
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
