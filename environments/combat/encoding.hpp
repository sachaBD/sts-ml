#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <nlohmann/json.hpp>

namespace stsrl {

// Encoding v4 = every v3 field (unchanged) plus the fields marked "v4" below: player statuses outside the v3
// global numeric, monster statuses and previous move, and potion / relic tokens (environments/combat/encoding_v4.cpp, which
// also lists what is covered: Ironclad, act 1). deep_sets_v1 / v2 read only the v3 fields; deep_sets_v3 reads
// all. to_json writes the v3 fields only, as encoding_version 3 (the combat_v3 table).
inline constexpr std::uint32_t combat_encoding_schema_version = 4;
inline constexpr std::uint32_t combat_v3_json_encoding_version = 3;

// v4 feature widths (layouts: environments/combat/encoding_v4.cpp; mirrored by agents/combat/value/deep_sets_v3.py).
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
    std::optional<PotionToken> potion;                // v4: the drunk potion's token
    std::optional<std::array<float, 6>> interaction;  // v4: card play x target (CardMonsterInteraction numeric)
    bool discards_potion{};                           // v4: a potion action that discards (target > 5), not drinks
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

inline nlohmann::json v4_json(const MonsterToken& m) {
    return {{"monster_id", m.monster_id}, {"move_id", m.move_id}, {"numeric", m.numeric},
            {"previous_move_id", m.previous_move_id}, {"status", m.status}};
}

inline nlohmann::json v4_json(const ActionToken& a) {
    nlohmann::json j = {{"action", a.execution_index}, {"kind", static_cast<int>(a.kind)},
                        {"card_selection_task", a.card_selection_task}, {"skips_selection", a.skips_selection},
                        {"discards_potion", a.discards_potion},
                        {"card", nullptr}, {"monster", nullptr}, {"potion", nullptr}, {"interaction", nullptr}};
    if (a.source_card) j["card"] = *a.source_card;
    if (a.target_monster) j["monster"] = v4_json(*a.target_monster);
    if (a.potion) j["potion"] = {{"potion_id", a.potion->potion_id}, {"numeric", a.potion->numeric}};
    if (a.interaction) j["interaction"] = *a.interaction;
    return j;
}

// The v4 columns of a combat_v3 row (environments/combat/schema.py; nullable, so rows written before
// them read as NULL). to_json(s) plus these is a whole v4 state. legal_actions (the policy's inputs, `action` =
// the row's actions[].action / chosen_action index) only with with_actions: decision rows, not child rows.
inline nlohmann::json v4_columns(const EncodedCombatState& s, bool with_actions) {
    nlohmann::json j = {{"v4_encoding_version", combat_encoding_schema_version}, {"player_numeric", s.global.player},
                        {"max_hp", s.global.max_hp}, {"monster_v4", nlohmann::json::array()},
                        {"potion_tokens", nlohmann::json::array()}, {"relic_tokens", nlohmann::json::array()}};
    for (const auto& m : s.monsters)
        j["monster_v4"].push_back({{"previous_move_id", m.previous_move_id}, {"status", m.status}});
    for (const auto& p : s.potions) j["potion_tokens"].push_back({{"potion_id", p.potion_id}, {"numeric", p.numeric}});
    for (const auto& r : s.relics) j["relic_tokens"].push_back({{"relic_id", r.relic_id}, {"numeric", r.numeric}});
    if (with_actions) {
        j["legal_actions"] = nlohmann::json::array();
        for (const auto& a : s.legal_actions) j["legal_actions"].push_back(v4_json(a));
    }
    return j;
}

// A whole row's state: to_json (v3 columns) plus v4_columns.
inline nlohmann::json state_row(const EncodedCombatState& s, bool with_actions) {
    nlohmann::json j = s;
    j.update(v4_columns(s, with_actions));
    return j;
}

} // namespace stsrl
