#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace stsrl {

inline constexpr std::uint32_t combat_encoding_schema_version = 2;

enum class CardZone : std::uint8_t { hand, draw, discard, exhaust, offered };
enum class EncodedActionKind : std::uint8_t { card, potion, single_card_selection, multi_card_selection, end_turn };

enum class CardType : std::uint8_t { attack, skill, power, status, curse };
enum class TargetType : std::uint8_t { none, one_enemy, all_enemies, random_enemy };

struct GlobalFeatures {
    std::array<float, 22> numeric{};
    int input_state{};
    int card_selection_task{};
    auto operator==(const GlobalFeatures&) const -> bool = default;
};

struct CardToken {
    int card_id{};
    CardZone zone{};
    CardType card_type{};
    TargetType target_type{};
    std::array<float, 14> numeric{};
    auto operator==(const CardToken&) const -> bool = default;
};

struct MonsterToken {
    int monster_id{};
    int move_id{};
    std::array<float, 9> numeric{};
    auto operator==(const MonsterToken&) const -> bool = default;
};

struct CardMonsterInteraction {
    std::uint16_t card_index{};
    std::uint8_t monster_index{};
    std::array<float, 6> numeric{};
    auto operator==(const CardMonsterInteraction&) const -> bool = default;
};

struct ActionToken {
    EncodedActionKind kind{};
    std::optional<CardToken> source_card;
    std::optional<MonsterToken> target_monster;
    std::optional<int> potion_id;
    int card_selection_task{};
    bool skips_selection{};
    std::size_t execution_index{};
    auto operator==(const ActionToken&) const -> bool = default;
};

struct EncodedCombatState {
    static constexpr std::uint32_t schema_version = combat_encoding_schema_version;
    std::uint32_t version = schema_version;
    GlobalFeatures global;
    std::vector<CardToken> cards;
    std::vector<MonsterToken> monsters;
    std::vector<CardMonsterInteraction> card_monster_interactions;
    std::vector<ActionToken> legal_actions;
    auto operator==(const EncodedCombatState&) const -> bool = default;
};

} // namespace stsrl
