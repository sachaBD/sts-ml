// Legal rest/shop actions and their publicly observable after-states.
#pragma once
#include "environments/overworld/macro_sim.hpp"
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace stsrl::overworld {
// sts::Neow::Bonus / Drawback names (Neow.h order).
inline constexpr const char* neow_bonus_names[] = {
    "three_cards", "one_random_rare_card", "remove_card", "upgrade_card", "transform_card", "random_colorless",
    "three_small_potions", "random_common_relic", "ten_percent_hp_bonus", "three_enemy_kill", "hundred_gold",
    "random_colorless_2", "remove_two", "one_rare_relic", "three_rare_cards", "two_fifty_gold",
    "transform_two_cards", "twenty_percent_hp_bonus", "boss_relic", "invalid"};
inline constexpr const char* neow_drawback_names[] = {
    "invalid", "none", "ten_percent_hp_loss", "no_gold", "curse", "percent_damage", "lose_starter_relic"};

struct Option {
    nlohmann::json label;
    std::vector<int> actions;  // GameActions executed in order
    nlohmann::json after;
};
std::string outcome_key(const sts::GameContext& gc);
std::vector<Option> rest_options(const sts::GameContext& gc, std::vector<std::string>& keys);
std::vector<Option> shop_options(const sts::GameContext& gc, std::vector<std::string>& keys);
}  // namespace stsrl::overworld
