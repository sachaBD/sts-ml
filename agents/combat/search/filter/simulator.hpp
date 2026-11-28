#pragma once
#include "agents/combat/search/filter/tree.hpp"
#include "environments/combat/public.hpp"
#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace stsrl::search::filter {
// Advances a BattleContext by one public action key and skips any forced single-action chain.
inline bool play_action(sts::BattleContext& state, std::uint64_t move_key) {
    const auto legal = combat::legal_actions(state);
    auto it = std::ranges::find_if(legal, [&](auto a) { return combat::action_key(state, a) == move_key; });
    if (it == legal.end())
        throw std::logic_error{"filter: public action missing"};
    bool chance = it->execute(state).isChance();
    for (int f = 0; f < 64 && state.outcome == sts::Outcome::UNDECIDED; ++f) {
        const auto next = combat::legal_actions(state);
        if (next.size() != 1)
            break;
        chance |= next[0].execute(state).isChance();
    }
    return chance;
}

// Materializes tree transitions using a shared pool of root particles for deterministic prefixes.
class WorldStepper {
public:
    WorldStepper(const sts::BattleContext& start, int world_count, std::uint64_t stream)
        : start_(start), world_count_(std::max(1, world_count)), stream_(stream) {}

    int world_count() const { return world_count_; }

    std::pair<sts::BattleContext, bool> step(const Tree& tree, int parent_id, int edge_idx, int sample_idx) {
        const auto& parent = tree.node(parent_id);
        sts::BattleContext state;
        if (parent.world < 0) {
            if (worlds_.empty())
                worlds_ = combat::particles(start_, world_count_, stream_);
            state = worlds_[sample_idx];
            for (const auto key : tree.path_keys(parent_id))
                play_action(state, key);
        } else if (sample_idx == 0) {
            state = parent.state;
        } else {
            const auto nested = stream_ ^ parent.key ^ (std::uint64_t(edge_idx + 1) * 0x9E3779B97F4A7C15ULL);
            state = combat::particles(parent.state, sample_idx + 1, nested)[sample_idx];
        }
        const bool chance = play_action(state, parent.action_keys[edge_idx]);
        return {std::move(state), chance};
    }

private:
    sts::BattleContext start_;
    int world_count_;
    std::uint64_t stream_;
    std::vector<sts::BattleContext> worlds_;
};
}
