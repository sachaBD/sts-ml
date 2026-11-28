#include "agents/combat/search/filter/filter.hpp"
#include "agents/combat/search/filter/simulator.hpp"
#include "agents/combat/search/filter/tree.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace stsrl::search {
namespace {
using State = sts::BattleContext;
using Key = std::uint64_t;
using filter::Tree;
using filter::WorldStepper;

struct CachedEval {
    Tree::EvalSummary summary;
    std::vector<float> thinking;
    std::vector<std::string> labels;  // populated only when tracing
};

struct Inflight {
    Tree::Candidate cand;
    int sample_idx = 0;
    bool chance = false;
    State state;
    std::vector<Action> actions;
    Key key = 0;
};

struct Rollout {
    int node_id = 0, target_turn = 0, steps = 0;
    double voi = 0, weight = 0, gap = 0, innovation = 0;
};

class Filter final : public CombatSearch {
    State start;
    int cap;
    double price;
    FilterMode mode;
    FilterTrace* trace;
    double explore;
    Key explore_seed;
    std::vector<Action> root_actions;
    WorldStepper stepper;
    Tree tree;
    std::unordered_map<Key, CachedEval> cache;
    std::optional<Inflight> inflight;
    std::optional<Rollout> rollout;
    std::vector<Query> queries;
    int spent = 0, root_trials = 0, interior_trials = 0;
    bool done = false;

    const CachedEval& store_eval(Key key, const Query& q, const Evaluation& e) {
        if (e.logits.empty() || e.logits.size() != q.actions.size() || e.thinking.size() != 4)
            throw std::invalid_argument{"filter: expected policy and four thinking outputs"};
        CachedEval ev;
        ev.summary.value = e.value;
        ev.summary.sd = filter::prior_sd(e.value, e.thinking);
        ev.summary.logits = e.logits;
        ev.thinking = e.thinking;
        ev.summary.action_keys.reserve(q.actions.size());
        for (auto a : q.actions)
            ev.summary.action_keys.push_back(combat::action_key(q.state, a));
        ev.summary.order.resize(e.logits.size());
        std::iota(ev.summary.order.begin(), ev.summary.order.end(), 0);
        std::stable_sort(ev.summary.order.begin(), ev.summary.order.end(),
                         [&](int a, int b) { return e.logits[a] > e.logits[b]; });
        if (trace) {
            ev.labels.reserve(q.actions.size());
            for (const auto& a : q.actions) {
                std::ostringstream s;
                a.printDesc(s, q.state);
                ev.labels.push_back(s.str());
            }
        }
        return cache.emplace(key, std::move(ev)).first->second;
    }

    void record_node(int id, const CachedEval* ev, bool cached_hit, int born) {
        if (!trace)
            return;
        const auto& n = tree.node(id);
        trace->nodes.push_back({n.id, n.parent, n.parent_edge, n.world, born, n.turn, n.key, n.terminal, n.cycle,
                                cached_hit, float(n.nn_value), n.prior_sd,
                                ev ? ev->summary.logits : std::vector<float>{},
                                ev ? ev->thinking : std::vector<float>{},
                                ev ? ev->labels : std::vector<std::string>{},
                                std::vector<bool>(ev ? ev->summary.logits.size() : 0, false)});
    }

    void attach(const Inflight& p, const CachedEval* ev, bool cached_hit) {
        const bool first_child = tree.node(p.cand.parent).edges[p.cand.edge].world_children.empty();
        const auto [child_id, created] = tree.add_child(p.cand.parent, p.cand.edge, p.sample_idx, p.chance, p.key,
                                                        p.state, ev ? &ev->summary : nullptr);
        if (trace && first_child)
            trace->nodes[p.cand.parent].edge_chance[p.cand.edge] = p.chance;
        if (p.cand.parent == 0)
            ++root_trials;
        else
            ++interior_trials;
        tree.backup(mode == FilterMode::fixed_uncertainty);
        if (trace) {
            if (created)
                record_node(child_id, ev, cached_hit, int(trace->steps.size()) + 1);
            trace->steps.push_back({p.cand.parent, p.cand.edge, p.chance ? p.sample_idx : tree.node(child_id).world,
                                    child_id, spent, p.cand.world_sample, p.cand.voi, p.cand.weight, p.cand.gap,
                                    p.cand.innovation, tree.root_q()});
        }
        const auto& child = tree.node(child_id);
        const int target_turn = rollout ? rollout->target_turn : tree.node(p.cand.parent).turn + 1;
        const int rsteps = rollout ? rollout->steps + 1 : 1;
        if (created && mode != FilterMode::root_only && !child.terminal && !child.cycle && !child.order.empty() &&
            child.turn < target_turn && rsteps < 16) {
            rollout = {child_id, target_turn, rsteps, p.cand.voi, p.cand.weight, p.cand.gap, p.cand.innovation};
        } else {
            rollout.reset();
        }
    }

    void advance() {
        const int step_guard = std::max(cap * 4 + 32, 64);
        while (!done) {
            if (spent >= cap) {
                if (trace)
                    trace->stop = cap == 0 ? "cap" : "request cap";
                done = true;
                break;
            }
            if (root_trials + interior_trials >= step_guard) {
                if (trace)
                    trace->stop = "step limit";
                done = true;
                break;
            }
            std::optional<Tree::Candidate> cand;
            if (rollout) {
                const auto& cur = tree.node(rollout->node_id);
                const int greedy = cur.order[0];
                const auto& edge = cur.edges[greedy];
                if (edge.world_children.empty() || (edge.chance && int(edge.world_children.size()) < stepper.world_count())) {
                    cand = {rollout->node_id, greedy, cur.depth + 1, !edge.world_children.empty(), rollout->voi,
                            rollout->weight, rollout->gap, rollout->innovation};
                } else {
                    rollout.reset();
                }
            }
            if (!cand) {
                cand = tree.best_candidate(stepper.world_count(), mode == FilterMode::root_only);
                const double top_voi = cand ? cand->voi : 0.;
                const bool overpriced = cand && price > 0 && top_voi <= price;
                if (trace) {
                    trace->last_voi = top_voi;
                    if (!cand)
                        trace->stop = tree.node(0).edges.size() <= 1 ? "single action" : "exhausted";
                    else if (overpriced)
                        trace->stop = "price";
                }
                if (!cand || overpriced) {
                    done = true;
                    break;
                }
                // Pair root-prefix world samples with the prior line's baseline on that same world w.
                if (mode != FilterMode::root_only && tree.node(cand->parent).world < 0) {
                    const int want_w = int(tree.node(cand->parent).edges[cand->edge].world_children.size());
                    if (const auto ref = tree.reference_step_for_world(want_w)) {
                        const auto& rn = tree.node(ref->first);
                        const auto& re = rn.edges[ref->second];
                        cand = {ref->first, ref->second, rn.depth + 1, !re.world_children.empty(), cand->voi,
                                cand->weight, cand->gap, cand->innovation};
                    }
                }
            }
            const int sample_idx = int(tree.node(cand->parent).edges[cand->edge].world_children.size());
            auto [next_state, chance] = stepper.step(tree, cand->parent, cand->edge, sample_idx);
            const Key next_key = combat::observation_key(next_state);
            Inflight pending{*cand, sample_idx, chance, std::move(next_state), {}, next_key};
            if (pending.state.outcome != sts::Outcome::UNDECIDED) {
                attach(pending, nullptr, false);
                continue;
            }
            if (auto it = cache.find(next_key); it != cache.end()) {
                attach(pending, &it->second, true);
                continue;
            }
            pending.actions = combat::legal_actions(pending.state);
            inflight = std::move(pending);
            queries.push_back({inflight->state, inflight->actions});
            break;
        }
    }

public:
    Filter(const State& state, int limit, int nw, double p, FilterMode mode, Key seed, FilterTrace* trace,
           double explore, Key explore_seed)
        : start(state), cap(limit), price(p), mode(mode), trace(trace), explore(explore), explore_seed(explore_seed),
          root_actions(combat::legal_actions(state)), stepper(state, nw, seed) {
        queries.push_back({start, root_actions});
    }

    std::span<const Query> pending() const override { return queries; }

    void evaluated(std::span<const Evaluation> eval) override {
        if (eval.size() != 1 || queries.empty())
            throw std::logic_error{"filter: wrong evaluation count"};
        const auto& e = eval[0];
        if (tree.empty()) {
            const Key root_key = combat::observation_key(start);
            const auto& ev = store_eval(root_key, queries[0], e);
            queries.clear();
            tree.add_root(root_key, start, ev.summary);
            tree.backup(mode == FilterMode::fixed_uncertainty);
            record_node(0, &ev, e.cached, 0);
            if (trace) {
                trace->opening = e;
                trace->initial_q = tree.root_q();
            }
        } else {
            if (!inflight)
                throw std::logic_error{"filter: unexpected evaluation"};
            const auto& ev = store_eval(inflight->key, queries[0], e);
            queries.clear();
            if (!e.cached)
                ++spent;
            if (trace)
                trace->spent = spent;
            auto p = std::move(*inflight);
            inflight.reset();
            attach(p, &ev, e.cached);
        }
        advance();
    }

    Result result() const override {
        const auto qs = tree.root_q();
        const int best = tree.leader(0);
        std::vector<float> pi(qs.size());
        double z = 0;
        if (root_trials == 0 && interior_trials == 0) {
            const auto& logits = cache.at(tree.node(0).key).summary.logits;
            const float top = logits[tree.node(0).order[0]];
            for (std::size_t i = 0; i < logits.size(); ++i)
                z += pi[i] = float(std::exp(double(logits[i] - top)));
        } else {
            const double top = qs[best];
            for (std::size_t i = 0; i < qs.size(); ++i)
                z += pi[i] = float(std::exp((qs[i] - top) / filter::kPolicyTemp));
        }
        for (auto& x : pi)
            x = float(double(x) / z);
        int play = best;
        if (explore > 0) {
            std::mt19937_64 random{explore_seed};
            std::extreme_value_distribution<double> gumbel;
            double top_key = -std::numeric_limits<double>::infinity();
            for (int i = 0; i < int(pi.size()); ++i)
                if (const double key = std::log(double(pi[i])) + explore * gumbel(random); key > top_key) {
                    top_key = key;
                    play = i;
                }
        }
        return {root_actions[play], root_actions, pi, float(std::clamp(qs[best], 0., 1.)), root_trials, interior_trials};
    }
};
}  // namespace

std::unique_ptr<CombatSearch> filter_search(const State& state, int cap, int worlds, double price, FilterMode mode,
                                            std::uint64_t stream, FilterTrace* trace, double explore,
                                            std::uint64_t explore_seed) {
    return std::make_unique<Filter>(state, cap, worlds, price, mode, stream, trace, explore, explore_seed);
}
}
