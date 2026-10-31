#include "agents/combat/search/rollout_blend/rollout_blend.hpp"
#include <stdexcept>

namespace stsrl::search {
namespace {
constexpr int max_rollout_actions = 512;  // an unfinished rollout counts as a loss
}  // namespace

RolloutBlend::RolloutBlend(const sts::BattleContext& any_state, double mix, std::uint64_t seed)
    : mix_{mix}, rng_{seed}, rollout_{any_state} {
    if (!(mix >= 0 && mix <= 1)) throw std::invalid_argument{"rollout blend: mix must be in [0, 1]"};
    rollout_.rolloutMode = 2;
}

Evaluation RolloutBlend::operator()(const Query& query, Evaluation evaluation) {
    auto state = query.state;
    rollout_.randGen.seed(rng_());
    sts::search::BattleScumSearcher2::Node scratch;
    for (int i = 0; i < max_rollout_actions && state.outcome == sts::Outcome::UNDECIDED; ++i)
        rollout_.rolloutAction(scratch, state).execute(state);
    const bool won = state.outcome == sts::Outcome::PLAYER_VICTORY;
    evaluation.value = WinProbability((1 - mix_) * evaluation.value + mix_ * won);
    return evaluation;
}
}  // namespace stsrl::search
