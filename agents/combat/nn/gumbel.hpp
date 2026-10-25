#pragma once
// Gumbel MuZero search (Danihelka et al. 2022; as mctx gumbel_muzero_policy) over the real simulator.
// Root: Gumbel top-k + Sequential Halving. Interior: deterministic improved-policy selection.
// Hidden information: each simulation runs on one of `particles` states sampled from the public observation;
// tree children are keyed by public observation, edges by public action key.
//
// The search never calls the network itself, so many searches can share one batched evaluation:
//   Search s{state, settings, seed};
//   while (const Tokens* leaf = s.pending()) s.evaluated(net(*leaf));
//   Result r = s.result();
#include "agents/combat/nn/net.hpp"
#include <memory>
#include <random>

namespace stsrl::nn {
struct Settings {
    int simulations = 256;
    int considered = 16;        // Gumbel top-k at the root
    double gumbel_scale = 0;    // 1 for self-play (exploration), 0 for evaluation
    double rollout_mix = 0.5;   // leaf value = (1 - mix) * network + mix * guided rollout
    int particles = 8;
    double value_scale = 0.1, maxvisit_init = 50;  // mctx qtransform_completed_by_mix_value defaults
};

struct Result {
    sts::search::Action action;                 // move to play
    std::vector<sts::search::Action> actions;   // legal actions (legal_actions order)
    std::vector<float> policy;                  // improved policy over `actions`: the training target
};

class Search {
public:
    Search(const sts::BattleContext& observed, const Settings& settings, std::uint64_t seed);
    ~Search();
    // The tokens the search needs evaluated next; nullptr once every simulation is done.
    const Tokens* pending() const;
    // The network's prediction for pending(); runs the search on to the next leaf that needs the network.
    void evaluated(const Prediction& prediction);
    Result result() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace stsrl::nn
