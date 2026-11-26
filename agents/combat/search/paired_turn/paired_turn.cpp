#include "agents/combat/search/paired_turn/paired_turn.hpp"
#include "environments/combat/public.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <unordered_map>

namespace stsrl::search {
namespace {
constexpr int screen_moves = 3;
constexpr int screen_worlds = 2;
constexpr int refine_worlds = 4;
constexpr double prior_weight = 30.0;  // Logit points per unit of win probability before sampling.
constexpr float weak_margin = 0.03f;
constexpr double close_margin = 0.04;
constexpr double third_move_margin = 0.06;

struct Prediction {
    float value;
    std::uint64_t greedy;
};

// Cache only network answers. The simulated state and its endpoint belong to the lane.
class Observations {
public:
    const Prediction* request(std::uint64_t key, const sts::BattleContext& state) {
        if (const auto it = known_.find(key); it != known_.end()) return &it->second;
        if (std::ranges::find_if(leaves_, [key](const Leaf& l) { return l.key == key; }) == leaves_.end())
            leaves_.push_back({key, state, combat::legal_actions(state)});
        return nullptr;
    }

    std::span<const Query> pending() const { return queries_; }

    void batch() {
        for (const auto& leaf : leaves_) queries_.push_back({leaf.state, leaf.actions});
    }

    void evaluated(std::span<const Evaluation> answers) {
        if (answers.size() != leaves_.size() || answers.empty())
            throw std::invalid_argument{"paired turn: one evaluation per query"};
        for (std::size_t i = 0; i < answers.size(); ++i) {
            const auto& leaf = leaves_[i];
            const auto& answer = answers[i];
            if (answer.logits.size() != leaf.actions.size())
                throw std::invalid_argument{"paired turn: one logit per action"};
            const auto greedy = std::ranges::max_element(answer.logits) - answer.logits.begin();
            known_.emplace(leaf.key, Prediction{answer.value, combat::action_key(leaf.state, leaf.actions[greedy])});
        }
        queries_.clear();
        leaves_.clear();
    }

private:
    struct Leaf {
        std::uint64_t key;
        sts::BattleContext state;
        std::vector<Action> actions;
    };
    std::unordered_map<std::uint64_t, Prediction> known_;
    std::vector<Leaf> leaves_;
    std::vector<Query> queries_;
};

void play(sts::BattleContext& state, std::uint64_t key) {
    for (const auto action : combat::legal_actions(state))
        if (combat::action_key(state, action) == key) {
            action.execute(state);
            return;
        }
    throw std::runtime_error{"paired turn: action missing in a particle"};
}

struct Lane {
    std::size_t edge;
    sts::BattleContext state;
    bool extend;
    int steps = 1;
    std::optional<float> first_boundary;
    std::optional<float> value;

    void advance(Observations& observations, int root_turn) {
        constexpr int step_limit = 32;
        while (!value) {
            if (state.outcome != sts::Outcome::UNDECIDED) {
                finish(state.outcome == sts::Outcome::PLAYER_VICTORY ? 1.0f : 0.0f);
                return;
            }
            const bool at_boundary = state.turn > root_turn && !first_boundary;
            const bool finished = steps >= step_limit || state.turn > root_turn + int(extend);
            if (!at_boundary && !finished) {
                const auto actions = combat::legal_actions(state);
                if (actions.size() == 1) {
                    actions.front().execute(state);
                    ++steps;
                    continue;
                }
            }
            const auto* prediction = observations.request(combat::observation_key(state), state);
            if (!prediction) return;
            if (finished) {
                finish(prediction->value);
                return;
            }
            if (at_boundary) first_boundary = prediction->value;
            play(state, prediction->greedy);
            ++steps;
        }
    }
    void finish(float endpoint) {
        value = first_boundary ? 0.5f * (*first_boundary + endpoint) : endpoint;
    }
};

class PairedTurn final : public CombatSearch {
public:
    PairedTurn(const sts::BattleContext& observed, int budget, int particles, std::uint64_t seed)
        : observed_{observed}, budget_{budget}, particles_{particles}, seed_{seed},
          actions_{combat::legal_actions(observed)}, values_(actions_.size()), scores_(actions_.size()) {
        observations_.request(combat::observation_key(observed), observed);
        observations_.batch();
    }

    std::span<const Query> pending() const override { return observations_.pending(); }

    void evaluated(std::span<const Evaluation> answers) override {
        observations_.evaluated(answers);
        if (prior_.empty()) init(answers.front());
        while (!lanes_.empty()) {
            for (auto& lane : lanes_) lane.advance(observations_, observed_.turn);
            if (std::ranges::any_of(lanes_, [](const Lane& l) { return !l.value; })) {
                observations_.batch();
                return;
            }
            finish_wave();
        }
    }

    Result result() const override {
        const auto chosen = std::ranges::max_element(scores_) - scores_.begin();
        const double top = scores_[chosen];
        std::vector<float> policy;
        double sum = 0;
        for (const double score : scores_) sum += policy.emplace_back(std::exp(score - top));
        for (auto& p : policy) p = float(p / sum);
        return {actions_[chosen], actions_, std::move(policy), values_[chosen]};
    }

private:
    void init(const Evaluation& root) {
        root_value_ = root.value;
        prior_ = root.logits;
        std::ranges::copy(prior_, scores_.begin());
        std::ranges::fill(values_, root.value);
        if (actions_.size() < 2 || budget_ < 2 || std::min(root.value, 1 - root.value) < 5e-4f) return;

        order_.resize(actions_.size());
        std::iota(order_.begin(), order_.end(), 0);
        std::ranges::stable_sort(order_, std::greater{}, [&](std::size_t e) { return prior_[e]; });
        const auto width = std::min({std::size_t(screen_moves), actions_.size(), std::size_t(budget_)});
        worlds_ = combat::particles(observed_, std::min(particles_, refine_worlds), seed_);
        launch(std::span{order_}.first(width), screen_worlds);
    }

    void launch(std::span<const std::size_t> edges, int max_worlds) {
        const int count = std::min({max_worlds, budget_ / int(edges.size()), int(worlds_.size())});
        if (count <= 0) return;
        wave_edges_.assign(edges.begin(), edges.end());
        for (int w = 0; w < count; ++w) {
            for (const auto e : edges) {
                auto state = worlds_[w];
                play(state, combat::action_key(observed_, actions_[e]));
                lanes_.push_back({e, std::move(state), refined_, 1, {}, {}});
            }
        }
        budget_ -= int(lanes_.size());
    }

    void finish_wave() {
        const auto count = lanes_.size() / wave_edges_.size();
        for (const auto e : wave_edges_) values_[e] = 0;
        for (const auto& lane : lanes_) values_[lane.edge] += *lane.value / float(count);
        lanes_.clear();

        // All finalists are scored again, now at the same two-turn horizon.
        std::ranges::fill(scores_, -std::numeric_limits<double>::infinity());
        const auto weight = prior_weight + count;
        for (const auto e : wave_edges_) scores_[e] = prior_[e] + weight * (values_[e] - root_value_);
        if (refined_) return;

        const auto width = wave_edges_.size();
        std::ranges::stable_sort(wave_edges_, std::greater{}, [&](std::size_t e) { return scores_[e]; });
        const auto leader = wave_edges_[0], runner_up = wave_edges_[1];
        const bool weak = values_[leader] < root_value_ - weak_margin;
        const bool close = scores_[leader] - scores_[runner_up] < close_margin * weight;
        if (!weak && !close) return;

        std::vector finalists{leader, runner_up};
        if (budget_ >= 3) {
            if (weak && width < order_.size()) finalists.push_back(order_[width]);
            else if (width == 3 && scores_[leader] - scores_[wave_edges_[2]] < third_move_margin * weight)
                finalists.push_back(wave_edges_[2]);
        }
        refined_ = true;
        launch(finalists, refine_worlds);
    }

    sts::BattleContext observed_;
    int budget_, particles_;
    std::uint64_t seed_;
    std::vector<Action> actions_;
    std::vector<float> values_, prior_;
    std::vector<double> scores_;
    float root_value_ = 0;
    Observations observations_;
    std::vector<sts::BattleContext> worlds_;
    bool refined_ = false;
    std::vector<std::size_t> order_, wave_edges_;
    std::vector<Lane> lanes_;
};
}  // namespace

SearchFactory paired_turn(int particles, std::uint64_t world_draw) {
    return [=](const sts::BattleContext& state, int budget, std::mt19937_64&) {
        // Stable within a turn, so the driver can reuse network answers across decisions.
        const auto seed = state.seed ^ (std::uint64_t(state.turn) << 32) ^ (world_draw * 0x9E3779B97F4A7C15ull);
        return std::make_unique<PairedTurn>(state, budget, particles, seed);
    };
}
}  // namespace stsrl::search
