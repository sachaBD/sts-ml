#pragma once

#include "combat/environment.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace stsrl::scenarios {

struct SlimeEntryProjection {
    std::string entry_id;
    std::uint64_t source_seed{};
    std::string public_snapshot_sha256;
    std::string deck_signature;
    int hp{};
    int max_hp{};
    std::vector<int> card_ids;
    std::vector<int> upgrades;
    std::vector<int> misc;
};

// Deck+HP-only projection: relics, potions, bottles, gold, map and all other
// strategic/run context are deliberately cleared rather than approximated.
[[nodiscard]] std::vector<SlimeEntryProjection> load_slime_entry_projections(
    const std::string& jsonl_path, int& skipped_nonaccepted);
// One accepted entry row (the JSON object, already extracted from its JSONL line).
[[nodiscard]] SlimeEntryProjection parse_slime_entry_projection(const std::string& json_object);
[[nodiscard]] CombatEnvironment slime_entry_projection(const SlimeEntryProjection& entry, std::uint64_t combat_seed = 0);

}  // namespace stsrl::scenarios
