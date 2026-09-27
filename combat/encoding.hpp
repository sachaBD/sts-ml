#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <nlohmann/json.hpp>

namespace stsrl {

// Encoding v4 = every v3 field (unchanged) plus the fields marked "v4" below: player statuses outside the v3
// global numeric, monster statuses and previous move, and potion / relic tokens (combat/encoding_v4.cpp, which
// also lists what is covered: Ironclad, act 1). deep_sets_v1 / v2 read only the v3 fields; deep_sets_v3 reads
// all. to_json writes the v3 fields only, as encoding_version 3 (the combat_v3 table).
inline constexpr std::uint32_t combat_encoding_schema_version = 4;
inline constexpr std::uint32_t combat_v3_json_encoding_version = 3;

// v4 feature widths (layouts: combat/encoding_v4.cpp; mirrored by python/sts_combat_rl/topology/deep_sets_v3.py).
inline constexpr std::size_t player_v4_features = 12;
inline constexpr std::size_t monster_status_features = 15;
inline constexpr std::size_t potion_features = 18;
inline constexpr std::size_t relic_features = 3;

enum class CardZone : std::uint8_t { hand, draw, discard, exhaust, offered };
enum class EncodedActionKind : std::uint8_t { card, potion, single_card_selection, multi_card_selection, end_turn };

enum class CardType : std::uint8_t { attack, skill, power, status, curse };
enum class TargetType : std::uint8_t { none, one_enemy, all_enemies, random_enemy };

struct GlobalFeatures {
    std::array<float, 50> numeric{};
    int input_state{};
    int card_selection_task{};
    std::array<float, player_v4_features> player{};  // v4
    float max_hp{};                                   // v4: raw, for deep_sets_v3's score formula
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
    int previous_move_id{};                               // v4: moveHistory[1]
    std::array<float, monster_status_features> status{};  // v4
    auto operator==(const MonsterToken&) const -> bool = default;
};

// v4: one held potion (empty slots have no token). potion_id is sts::Potion (0 = INVALID, the model's unknown id).
struct PotionToken {
    int potion_id{};
    std::array<float, potion_features> numeric{};
    auto operator==(const PotionToken&) const -> bool = default;
};

// v4: one relic of the combat state's relic bits. relic_id is sts::RelicId + 1 (0 = the model's unknown id).
struct RelicToken {
    int relic_id{};
    std::array<float, relic_features> numeric{};
    auto operator==(const RelicToken&) const -> bool = default;
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
    std::vector<PotionToken> potions;  // v4
    std::vector<RelicToken> relics;    // v4
    std::vector<ActionToken> legal_actions;
    auto operator==(const EncodedCombatState&) const -> bool = default;
};

inline void to_json(nlohmann::json& j, const CardToken& x) {
    j = nlohmann::json{
        {"card_id", x.card_id},
        {"zone", static_cast<int>(x.zone)},
        {"card_type", static_cast<int>(x.card_type)},
        {"target_type", static_cast<int>(x.target_type)},
        {"numeric", x.numeric}
    };
}

inline void to_json(nlohmann::json& j, const MonsterToken& x) {
    j = nlohmann::json{
        {"monster_id", x.monster_id},
        {"move_id", x.move_id},
        {"numeric", x.numeric}
    };
}

inline void to_json(nlohmann::json& j, const CardMonsterInteraction& x) {
    j = nlohmann::json{
        {"card_index", x.card_index},
        {"monster_index", x.monster_index},
        {"numeric", x.numeric}
    };
}

inline void to_json(nlohmann::json& j, const GlobalFeatures& g) {
    j = nlohmann::json{
        {"numeric", g.numeric},
        {"input_state", g.input_state},
        {"card_selection_task", g.card_selection_task}
    };
}

inline void to_json(nlohmann::json& j, const EncodedCombatState& s) {
    j = nlohmann::json{
        {"encoding_version", combat_v3_json_encoding_version},
        {"global_numeric", s.global.numeric},
        {"input_state", s.global.input_state},
        {"card_selection_task", s.global.card_selection_task},
        {"cards", s.cards},
        {"monsters", s.monsters},
        {"card_monster_interactions", s.card_monster_interactions}
    };
}

// The full v4 state (every v3 key plus the v4 fields); read by python/sts_combat_rl/training/encoding_v4.py.
inline nlohmann::json to_json_v4(const EncodedCombatState& s) {
    nlohmann::json j = s;
    j["encoding_version"] = combat_encoding_schema_version;
    j["player_numeric"] = s.global.player;
    j["max_hp"] = s.global.max_hp;
    for (std::size_t m = 0; m < s.monsters.size(); ++m) {
        j["monsters"][m]["previous_move_id"] = s.monsters[m].previous_move_id;
        j["monsters"][m]["status"] = s.monsters[m].status;
    }
    j["potions"] = nlohmann::json::array();
    for (const auto& p : s.potions) j["potions"].push_back({{"potion_id", p.potion_id}, {"numeric", p.numeric}});
    j["relics"] = nlohmann::json::array();
    for (const auto& r : s.relics) j["relics"].push_back({{"relic_id", r.relic_id}, {"numeric", r.numeric}});
    return j;
}

} // namespace stsrl
