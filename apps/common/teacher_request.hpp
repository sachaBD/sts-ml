// The teacher settings of a worker request, written by apps/common/app.py teacher_settings:
//   {leaf, simulations, oracle, random_move, particles (unless oracle), rollout_turns / rollout_steps (leaf hybrid
//    only), c_puct / fpu_reduction / prior_floor (leaf policy_net), merge_identical_cards / stop_factor / search_salt / tree_reuse (opt-in
//    search tweaks, teacher::set_tweak)}.
// Every setting that applies is required; anything else is an error.
#pragma once

#include "agents/teacher_leaves.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace stsrl::teacher {

struct Request {
    Leaf leaf;
    Budget budget;
    bool oracle;
    bool random_move;
};

inline Request parse_request(const nlohmann::json& t) {
    const auto positive = [&](const std::string& key) {
        const auto& value = t.at(key);
        if (!value.is_number_integer() || value.get<std::int64_t>() < 1)
            throw std::invalid_argument{key + " must be a positive integer"};
        return value.get<std::int64_t>();
    };
    const bool oracle = t.at("oracle").get<bool>();
    if (oracle == t.contains("particles"))
        throw std::invalid_argument{"particles is required unless oracle (one true-state particle), and only then"};
    Request request{.leaf = {t.at("leaf").get<std::string>()},
                    .budget = {.simulations = positive("simulations"),
                               .particles = oracle ? 1 : static_cast<int>(positive("particles"))},
                    .oracle = oracle, .random_move = t.at("random_move").get<bool>()};
    const bool hybrid = request.leaf.kind == "hybrid";
    if (hybrid != t.contains("rollout_turns") || hybrid != t.contains("rollout_steps"))
        throw std::invalid_argument{"rollout_turns / rollout_steps are required with leaf hybrid, and only then"};
    if (hybrid) {
        request.leaf.rollout_turns = static_cast<int>(positive("rollout_turns"));
        request.leaf.rollout_steps = static_cast<int>(positive("rollout_steps"));
    }
    for (const auto& [key, value] : t.items()) {
        if (key == "leaf" || key == "simulations" || key == "oracle" || key == "random_move" || key == "particles" ||
            key == "rollout_turns" || key == "rollout_steps")
            continue;
        if (!set_tweak(key, value)) throw std::invalid_argument{"unknown teacher setting: " + key};
    }
    return request;
}

}  // namespace stsrl::teacher
