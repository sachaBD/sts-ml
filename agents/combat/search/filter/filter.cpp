#include "agents/combat/search/filter/filter.hpp"
#include "agents/combat/search/filter/belief.hpp"
#include "environments/combat/public.hpp"
#include <array>
#include <map>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <sstream>
#include <unordered_map>

namespace stsrl::search {
namespace {
using State = sts::BattleContext;
using Key = std::uint64_t;
using filter::Gap;
constexpr std::array budgets{8, 24, 64, 160};
struct Node {
    double value;
    std::vector<Key> keys;
    std::vector<float> logits;
    std::vector<int> order;
    std::array<double, 4> gain;
    Node(const Query& q, const Evaluation& e) : value(e.value), logits(e.logits), order(logits.size()) {
        if (logits.empty() || logits.size() != q.actions.size() || e.thinking.size() != gain.size())
            throw std::invalid_argument{"filter: expected policy and four Horizon thinking outputs"};
        for (std::size_t i = 0; i < gain.size(); ++i) {
            gain[i] = e.thinking[i];
            if (!std::isfinite(gain[i]))
                throw std::invalid_argument{"filter: nonfinite gain"};
        }
        for (auto a : q.actions)
            keys.push_back(combat::action_key(q.state, a));
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return logits[a] > logits[b]; });
    }
    double prior_variance() const {
        // At a tie, E[max(0,N(0,s²))] = s/sqrt(2*pi). A scale prior, not a calibrated interval.
        const double floor = .05 * std::sqrt(std::max(0., value * (1 - value)));
        const double sd =
            std::max({1e-4, floor, std::sqrt(2 * std::numbers::pi) * *std::max_element(gain.begin(), gain.end())});
        return sd * sd;
    }
};
struct Return {
    double value;
    std::optional<Key> child;
};
struct Arm {
    Gap gap;
    std::vector<Return> samples;
};
struct Trial {
    int arm, world;
    std::optional<Key> child;
};
struct Branch {
    Gap gap;
    std::vector<bool> measured;
};
struct Request {
    State state;
    std::vector<Action> actions;
    Key key;
};

class Filter final : public CombatSearch {
    State start;
    int cap, world_count;
    double price;
    FilterMode mode;
    FilterTrace* trace;
    double explore;
    Key explore_seed, stream, root;
    std::vector<State> worlds;
    std::vector<Action> actions;
    std::unordered_map<Key, Node> cache;
    std::vector<Request> requests;
    std::vector<Query> queries;
    std::vector<Arm> arms;
    std::map<Key, Branch> branches; // stable tie break
    std::optional<Trial> trial;
    int prior = 0, reserve = 0, spent = 0, lanes = 0;
    bool done = false;
    int root_trials = 0, interior_trials = 0;

    const Node* lookup(const State& state, const std::vector<Action>& legal, Key key) {
        if (auto it = cache.find(key); it != cache.end())
            return &it->second;
        if (std::ranges::none_of(requests, [&](const Request& r) { return r.key == key; }))
            requests.push_back({state, legal, key});
        return nullptr;
    }
    static void play_key(State& state, std::span<const Action> legal, Key move, std::vector<std::string>* line = nullptr) {
        auto it = std::ranges::find_if(legal, [&](auto x) { return combat::action_key(state, x) == move; });
        if (it == legal.end()) throw std::logic_error{"filter: public action missing"};
        if (line) {
            std::ostringstream label;
            it->printDesc(label, state);
            line->push_back(label.str());
        }
        it->execute(state);
    }
    // Replaying through cached public observations avoids storing hidden state in a public node.
    std::optional<Return> plunge(int a, int w, std::optional<Key> branch, std::vector<std::string>* line = nullptr) {
        State state = worlds[w];
        play_key(state, combat::legal_actions(state), cache.at(root).keys[a], line);
        std::optional<Key> first;
        for (int steps = 0;; ++steps) {
            if (state.outcome != sts::Outcome::UNDECIDED)
                return Return{double(state.outcome == sts::Outcome::PLAYER_VICTORY), first};
            auto legal = combat::legal_actions(state);
            if (legal.size() == 1 && steps < 64) {
                legal[0].execute(state);
                continue;
            }
            const Key key = combat::observation_key(state);
            const auto* node = lookup(state, legal, key);
            if (!node)
                return {};
            if (state.turn != start.turn || steps >= 64)
                return Return{node->value, first};
            const bool first_choice = !first;
            if (first_choice)
                first = key;
            const int rank = first_choice && branch == first ? 1 : 0;
            const Key move = node->keys[node->order.at(rank)];
            play_key(state, legal, move, line);
        }
    }
    double reach(int a, Key key) const {
        const auto& s = arms[a].samples;
        return s.empty() ? 0. : double(std::ranges::count(s, std::optional{key}, &Return::child)) / s.size();
    }
    std::vector<double> scores() const {
        std::vector<double> s;
        for (int a = 0; a < int(arms.size()); ++a) {
            double v = arms[a].gap.mean();
            for (const auto& [key, b] : branches)
                if (b.gap.mean() > 0)
                    v += reach(a, key) * b.gap.mean();
            s.push_back(v);
        }
        return s;
    }
    void setup() {
        const auto& r = cache.at(root);
        prior = r.order[0];
        arms.resize(actions.size());
        for (int a : r.order)
            arms[a].gap = {.prior = (r.logits[a] - r.logits[prior]) / 30., .prior_var = r.prior_variance()};
        double efficiency = 0;
        for (int i = 0; i < 4; ++i)
            if (budgets[i] <= cap && r.gain[i] / budgets[i] > efficiency) {
                efficiency = r.gain[i] / budgets[i];
                reserve = budgets[i];
            }
        done = cap == 0 || arms.size() == 1 || (price > 0 && efficiency <= price);
        if (trace) {
            trace->reserve = reserve; trace->opening_rate = efficiency; trace->initial_scores = scores();
            if (done) trace->stop = "opening declined";
        }
        if (!done)
            worlds = combat::particles(start, world_count, stream);
    }
    double innovation(const Gap& g) const {
        return mode == FilterMode::fixed_uncertainty ? std::sqrt(g.prior_var / 2.) : g.innovation();
    }
    std::optional<Trial> select() const {
        const auto s = scores();
        const int leader = int(std::ranges::max_element(s) - s.begin());
        std::optional<Trial> best;
        double rate = -1;
        const double lane_cost = lanes ? std::max(1., double(spent) / lanes) : 1.;
        auto offer = [&](Trial t, double gain, int needed) {
            const double rho = gain / (needed * lane_cost);
            if (rho > rate) {
                rate = rho;
                best = t;
            }
        };
        for (int a : cache.at(root).order)
            if (a != prior && int(arms[a].samples.size()) < world_count) {
                int w = int(arms[a].samples.size());
                double other = -1e30;
                for (int j = 0; j < int(s.size()); ++j)
                    if (j != a)
                        other = std::max(other, s[j]);
                offer({a, w, {}}, filter::positive_part(-std::abs(s[a] - other), innovation(arms[a].gap)),
                      1 + (w >= int(arms[prior].samples.size())));
            }
        for (const auto& [key, b] : branches) {
            double gain = 0;
            for (int a = 0; a < int(s.size()); ++a)
                if (a != leader) {
                    const double d = reach(a, key) - reach(leader, key);
                    gain = std::max(gain, filter::branch_option(s[a] - s[leader] - d * std::max(0., b.gap.mean()), d,
                                                                b.gap.mean(), innovation(b.gap)));
                }
            if (gain <= 0)
                continue;
            bool offered = false;
            for (int a : cache.at(root).order) {
                for (int w = 0; w < int(arms[a].samples.size()); ++w)
                    if (arms[a].samples[w].child == key && !b.measured[w]) {
                        offer({a, w, key}, gain, 1);
                        offered = true;
                        break;
                    }
                if (offered)
                    break;
            }
        }
        const bool overpriced = spent >= reserve && price > 0 && rate <= price;
        if (trace) {
            trace->last_rate = rate;
            if (!best || overpriced) trace->stop = best ? "price" : "no remaining comparison";
        }
        return overpriced ? std::nullopt : best;
    }
    void save(int a, const Return& r) {
        arms[a].samples.push_back(r);
        if (mode != FilterMode::root_only && r.child && !branches.contains(*r.child))
            branches.emplace(
                *r.child, Branch{{.prior_var = cache.at(*r.child).prior_variance()}, std::vector<bool>(world_count)});
    }
    void advance() {
        while (!done) {
            if (!trial)
                trial = select();
            if (!trial) {
                done = true;
                break;
            }
            const auto [a, w, key] = *trial;
            std::vector<std::string> line, reference_line;
            auto y = plunge(a, w, key, trace ? &line : nullptr);
            const bool need_ref = !key && w == int(arms[prior].samples.size());
            auto ref = need_ref ? plunge(prior, w, {}, trace ? &reference_line : nullptr) : std::optional<Return>{};
            if (!requests.empty()) {
                if (spent >= cap) {
                    if (trace) trace->stop = "request cap";
                    requests.clear();
                    done = true;
                } else if (int(requests.size()) > cap - spent)
                    requests.resize(cap - spent);
                break;
            }
            lanes += 1 + need_ref;
            auto& gap = key ? branches.at(*key).gap : arms[a].gap;
            const double before = trace ? gap.mean() : 0., sd = trace ? gap.innovation() : 0.;
            if (key) {
                ++interior_trials;
                auto& b = branches.at(*key);
                b.gap.observe(y->value - arms[a].samples[w].value);
                b.measured[w] = true;
            } else {
                ++root_trials;
                if (need_ref)
                    save(prior, *ref);
                arms[a].gap.observe(y->value - arms[prior].samples.at(w).value);
                save(a, *y);
            }
            if (trace) {
                const double baseline = key ? arms[a].samples[w].value : arms[prior].samples[w].value;
                trace->trials.push_back({a, w, spent, key.has_value(), trace->last_rate, y->value, baseline,
                    before, gap.mean(), sd, gap.innovation(), scores(), std::move(line), std::move(reference_line)});
            }
            trial.reset();
        }
        queries.clear();
        for (const auto& r : requests)
            queries.push_back({r.state, r.actions});
    }

public:
    Filter(const State& state, int limit, int nw, double p, FilterMode mode, Key seed, FilterTrace* trace,
           double explore, Key explore_seed)
        : start(state), cap(limit), world_count(nw), price(p), mode(mode), trace(trace), explore(explore),
          explore_seed(explore_seed), stream(seed),
          root(combat::observation_key(state)), actions(combat::legal_actions(state)) {
        requests.push_back({state, actions, root});
        queries.push_back({requests[0].state, requests[0].actions});
    }
    std::span<const Query> pending() const override { return queries; }
    void evaluated(std::span<const Evaluation> eval) override {
        if (eval.size() != requests.size() || eval.empty())
            throw std::logic_error{"filter: wrong evaluation count"};
        bool opening = arms.empty();
        for (std::size_t i = 0; i < eval.size(); ++i) {
            cache.try_emplace(requests[i].key, queries[i], eval[i]);
            if (!opening && !eval[i].cached)
                ++spent;
        }
        queries.clear();
        requests.clear();
        if (trace) {
            trace->spent = spent;
            if (opening) trace->opening = eval.front();
        }
        if (opening) setup();
        advance();
    }
    Result result() const override {
        auto s = scores();
        int a = int(std::ranges::max_element(s) - s.begin());
        double top = s[a], z = 0;
        std::vector<float> pi;
        for (double x : s) {
            pi.push_back(std::exp(30. * (x - top)));
            z += pi.back();
        }
        for (auto& x : pi)
            x /= z;
        int play = a;
        if (explore > 0) {
            std::mt19937_64 random{explore_seed};
            std::extreme_value_distribution<double> gumbel;
            double best = -std::numeric_limits<double>::infinity();
            for (int i = 0; i < int(pi.size()); ++i)
                if (const double key = std::log(double(pi[i])) + explore * gumbel(random); key > best) {
                    best = key;
                    play = i;
                }
        }
        return {actions[play], actions,        pi, float(std::clamp(cache.at(root).value + s[a], 0., 1.)),
                root_trials,   interior_trials};
    }
};
}
std::unique_ptr<CombatSearch> filter_search(const State& state, int cap, int worlds, double price, FilterMode mode,
                                            std::uint64_t stream, FilterTrace* trace, double explore,
                                            std::uint64_t explore_seed) {
    return std::make_unique<Filter>(state, cap, worlds, price, mode, stream, trace, explore, explore_seed);
}
}
