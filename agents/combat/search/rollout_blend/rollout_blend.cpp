#include "agents/combat/search/rollout_blend/rollout_blend.hpp"
#include "sim/search/BattleScumSearcher2.h"
#include <random>
#include <stdexcept>

namespace stsrl::search {
namespace {
constexpr int max_rollout_actions = 512;  // an unfinished rollout counts as a loss

class RolloutBlended final : public CombatSearch {
public:
    RolloutBlended(std::unique_ptr<CombatSearch> inner, const sts::BattleContext& any_state, double mix, std::uint64_t seed)
        : inner_{std::move(inner)}, mix_{mix}, rng_{seed}, rollout_{any_state} {
        if (!(mix >= 0 && mix <= 1)) throw std::invalid_argument{"rollout blend: mix must be in [0, 1]"};
        rollout_.rolloutMode = 2;
    }
    std::span<const Query> pending() const override { return inner_->pending(); }
    Result result() const override { return inner_->result(); }

    void evaluated(std::span<const Evaluation> evaluations) override {
        const auto queries = inner_->pending();
        std::vector<Evaluation> blended{evaluations.begin(), evaluations.end()};
        for (std::size_t q = 0; q < blended.size(); ++q) {
            auto state = queries[q].state;
            rollout_.randGen.seed(rng_());
            sts::search::BattleScumSearcher2::Node scratch;
            for (int i = 0; i < max_rollout_actions && state.outcome == sts::Outcome::UNDECIDED; ++i)
                rollout_.rolloutAction(scratch, state).execute(state);
            const bool won = state.outcome == sts::Outcome::PLAYER_VICTORY;
            blended[q].value = WinProbability((1 - mix_) * blended[q].value + mix_ * won);
        }
        inner_->evaluated(blended);
    }

private:
    std::unique_ptr<CombatSearch> inner_;
    double mix_;
    std::mt19937_64 rng_;
    sts::search::BattleScumSearcher2 rollout_;
};
}  // namespace

std::unique_ptr<CombatSearch> with_rollouts(std::unique_ptr<CombatSearch> inner, const sts::BattleContext& any_state,
                                            double mix, std::uint64_t seed) {
    return std::make_unique<RolloutBlended>(std::move(inner), any_state, mix, seed);
}
}  // namespace stsrl::search
