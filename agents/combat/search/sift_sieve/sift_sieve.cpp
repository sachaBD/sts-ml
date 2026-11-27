#include "agents/combat/search/sift_sieve/sift_sieve.hpp"
#include "environments/combat/public.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>
#include <stdexcept>

namespace stsrl::search {
namespace {
class SiftSieve final : public CombatSearch {
public:
    SiftSieve(const sts::BattleContext& state, RootChoice choice, std::uint64_t seed)
        : state_{state}, actions_{combat::legal_actions(state_)}, root_{state_, actions_}, choice_{choice}, random_{seed} {
        if (actions_.empty()) throw std::invalid_argument{"sift-sieve: root needs a legal action"};
    }

    std::span<const Query> pending() const override {
        return result_ ? std::span<const Query>{} : std::span{&root_, 1};
    }

    void evaluated(std::span<const Evaluation> answers) override {
        if (result_ || answers.size() != 1 || answers[0].logits.size() != actions_.size())
            throw std::invalid_argument{"sift-sieve: needs one root prediction in legal action order"};
        const auto& [value, logits] = answers[0];
        if (!std::ranges::all_of(logits, [](float logit) { return std::isfinite(logit); }))
            throw std::invalid_argument{"sift-sieve: nonfinite policy logit"};
        const auto best = std::ranges::max_element(logits);
        std::vector<double> weights;
        weights.reserve(logits.size());
        for (float logit : logits) weights.push_back(std::exp(double(logit) - *best));
        const auto selected = choice_ == RootChoice::sample
            ? std::discrete_distribution<std::size_t>(weights.begin(), weights.end())(random_)
            : std::size_t(best - logits.begin());
        const double sum = std::accumulate(weights.begin(), weights.end(), 0.);
        std::vector<float> policy;
        policy.reserve(weights.size());
        for (double weight : weights) policy.push_back(float(weight / sum));
        result_.emplace(Result{actions_[selected], actions_, std::move(policy), value});
    }

    Result result() const override {
        if (!result_) throw std::logic_error{"sift-sieve: root prediction is still pending"};
        return *result_;
    }

private:
    sts::BattleContext state_;
    std::vector<Action> actions_;
    Query root_;
    RootChoice choice_;
    std::mt19937_64 random_;
    std::optional<Result> result_;
};
}  // namespace

SearchFactory sift_sieve(RootChoice choice) {
    return [choice](const sts::BattleContext& state, int, std::mt19937_64& random) {
        return std::make_unique<SiftSieve>(state, choice, random());
    };
}
}  // namespace stsrl::search
