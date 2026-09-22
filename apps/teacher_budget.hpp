// Teacher-search budget shared by generate_entry_mcts_records and play_entry_pbcs.
// Plays the same move as search.search(simulations), with less compute:
//   forced: one legal move -> only forced_simulations (for root_value and the actions row).
//   early stop: search in chunks; stop once N_best - N_second > simulations left, so the
//   most-visited root edge can no longer be caught (strictly, so no visit tie is possible
//   and selectedAction's value tie-break never comes into play).
#pragma once

#include "sim/search/PublicBeliefCombatSearch.h"

#include <algorithm>
#include <cstdint>

namespace stsrl {

inline std::int64_t run_teacher_search(sts::search::PublicBeliefCombatSearch& search, std::int64_t simulations,
                                       std::size_t legal_moves, bool early_stop,
                                       std::int64_t forced_simulations = 500, std::int64_t chunk = 500) {
    if (!early_stop) { search.search(simulations); return simulations; }
    if (legal_moves == 1) {
        const auto n = std::min(forced_simulations, simulations);
        search.search(n);
        return n;
    }
    std::int64_t used = 0;
    while (used < simulations) {
        const auto n = std::min(chunk, simulations - used);
        search.search(n);
        used += n;
        std::int64_t best = 0, second = 0;
        for (const auto& e : search.root().edges) {
            if (e.visits > best) { second = best; best = e.visits; }
            else if (e.visits > second) second = e.visits;
        }
        if (best - second > simulations - used) break;
    }
    return used;
}

}  // namespace stsrl
