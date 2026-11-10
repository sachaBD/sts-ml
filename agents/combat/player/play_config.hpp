#pragma once
// nn_play's settings: PLAY.toml, which play.py writes for every play stage. Its tables and keys are the fields below.
#include "agents/combat/search/config.hpp"
#include "infra/workers/actors.hpp"
#include <filesystem>
#include <string_view>

namespace stsrl::player {
struct PlayConfig {
    search::SearchConfig search;  // [search]
    infra::ActorConfig workers;   // [workers]
};

// Throws naming the table and key of a missing, mistyped, unknown or out-of-range setting.
PlayConfig parse_play_config(std::string_view toml);
PlayConfig read_play_config(const std::filesystem::path&);
}  // namespace stsrl::player
