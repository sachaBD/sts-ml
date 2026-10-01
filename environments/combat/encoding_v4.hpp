#pragma once

// Encoding v4 additions (see environments/combat/encoding_v4.cpp for the layouts and their act-1 Ironclad coverage).

#include "environments/combat/encoding.hpp"

#include <vector>

namespace sts {
struct BattleContext;
struct Monster;
enum class Potion : std::uint8_t;
enum class RelicId : std::uint8_t;
}  // namespace sts

namespace stsrl::v4 {

// Fills global.player and global.max_hp.
void encode_player(const sts::BattleContext& state, GlobalFeatures& global);
// Fills token.previous_move_id and token.status.
void encode_monster(const sts::Monster& monster, MonsterToken& token);
[[nodiscard]] PotionToken encode_potion(const sts::BattleContext& state, sts::Potion potion);
[[nodiscard]] std::vector<PotionToken> encode_potions(const sts::BattleContext& state);
[[nodiscard]] RelicToken encode_relic(const sts::BattleContext& state, sts::RelicId relic);
[[nodiscard]] std::vector<RelicToken> encode_relics(const sts::BattleContext& state);

}  // namespace stsrl::v4
