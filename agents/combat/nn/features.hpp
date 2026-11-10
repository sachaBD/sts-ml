#pragma once
// Combat features the network encodings are built from: what a player can see, as numbers.
// Both encodings (entity/encoding.cpp, inventory/encoding.cpp) read these. Values are frozen for trained models:
// changing any layout or scaling here needs a new encoding, never an edit.
// Coverage: Ironclad, act 1 (+ event cards). Unsupported cards throw rather than encode as zeros.
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>
#include "sim/search/Action.h"

namespace sts {
struct BattleContext;
}

namespace stsrl::nn {
inline constexpr std::size_t player_status_features = 12;
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
    std::array<float, player_status_features> player{};
    float max_hp{};
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
    int previous_move_id{};
    std::array<float, monster_status_features> status{};
    auto operator==(const MonsterToken&) const -> bool = default;
};


struct PotionToken {
    int potion_id{};
    std::array<float, potion_features> numeric{};
    auto operator==(const PotionToken&) const -> bool = default;
};


struct RelicToken {
    int relic_id{};
    std::array<float, relic_features> numeric{};
    auto operator==(const RelicToken&) const -> bool = default;
};

struct ActionToken {
    EncodedActionKind kind{};
    std::optional<CardToken> source_card;
    std::optional<MonsterToken> target_monster;
    std::optional<int> potion_id;
    int card_selection_task{};
    bool skips_selection{};
    std::optional<PotionToken> potion;
    std::optional<std::array<float, 6>> interaction;
    bool discards_potion{};
    auto operator==(const ActionToken&) const -> bool = default;
};

struct EncodedCombatState {
    GlobalFeatures global;
    std::vector<CardToken> cards;
    std::vector<MonsterToken> monsters;
    std::vector<PotionToken> potions;
    std::vector<RelicToken> relics;
    auto operator==(const EncodedCombatState&) const -> bool = default;
};

// Where a card feature row lives in the simulator (not a model input).
struct CardLocation {
    CardZone zone;
    int index;
    auto operator<=>(const CardLocation&) const = default;
};

// Features are sorted, so equal situations give equal inputs; the optional outputs map rows back to the simulator.
[[nodiscard]] std::vector<CardToken> encode_cards(const sts::BattleContext&, std::vector<CardLocation>* locations = nullptr);
[[nodiscard]] std::vector<MonsterToken> encode_monsters(const sts::BattleContext&, std::vector<int>* slots = nullptr);
[[nodiscard]] std::vector<PotionToken> encode_potions(const sts::BattleContext&, std::vector<int>* slots = nullptr);
[[nodiscard]] std::vector<RelicToken> encode_relics(const sts::BattleContext&);
[[nodiscard]] int incoming_damage(const sts::BattleContext&);
// Everything above plus the player's global features.
[[nodiscard]] EncodedCombatState encode_state(const sts::BattleContext&, std::vector<int>* monster_slots = nullptr);
// One legal move: its source card or potion, its target monster, and the card x target interaction.
[[nodiscard]] ActionToken encode_action(const sts::BattleContext&, sts::search::Action);
}  // namespace stsrl::nn
