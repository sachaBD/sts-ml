// Root Sequential Halving with Gumbel (SearchSettings::root_halving): schedule and completed-Q fixtures from the pinned
// mctx 0.0.71 sources, exact simulation budgets, visit structure, determinism, hidden-information independence and
// fail-closed misuse. A deterministic fake evaluator stands in for the network.
#include "agents/combat/pv/search.hpp"
#include "environments/combat/record_v4.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

using namespace stsrl::pv;
namespace {
void require(bool value, const std::string& message) { if (!value) throw std::runtime_error{message}; }

// Deterministic fake network: value and logits from a hash of the inputs (0..100 value units).
std::vector<Prediction> fake(std::span<const Inputs> batch) {
    std::vector<Prediction> out;
    for (const auto& in : batch) {
        std::uint64_t h = 1469598103934665603ULL;
        for (const auto& v : in) for (float x : v) { h ^= std::uint64_t(std::int64_t(x * 1000)); h *= 1099511628211ULL; }
        const auto actions = in[5].size() / std::size_t(widths[5]);
        Prediction p{float(h % 1000) / 10.f, {}};
        for (std::size_t a = 0; a < actions; ++a) p.logits.push_back(float((h >> (a % 50)) % 7) / 3.f);
        out.push_back(p);
    }
    return out;
}

template<class S> std::vector<std::int64_t> root_visits(const S& search) {
    std::vector<std::int64_t> v; for (const auto& e : search.root().edges) v.push_back(e.visits); return v;
}

// Final visit counts implied by a schedule: any consistent tie choice gives the same multiset.
std::vector<std::int64_t> schedule_counts(const HalvingSchedule& s, int m) {
    std::vector<std::int64_t> v(std::size_t(m), 0);
    for (auto cv : s.considered_visit) {
        auto it = std::find(v.begin(), v.end(), cv); require(it != v.end(), "schedule entry unreachable"); ++*it;
    }
    std::sort(v.rbegin(), v.rend()); return v;
}
}  // namespace

int main() {
    try {
        nlohmann::json fx; { std::ifstream f2{PV_HALVING_FIXTURE}; f2 >> fx; }
        // 1. schedules equal mctx get_sequence_of_considered_visits
        for (const auto& [key, expected] : fx.at("schedules").items()) {
            const int m = std::stoi(key.substr(0, key.find(':'))); const auto n = std::stoll(key.substr(key.find(':') + 1));
            require(halving_schedule(m, n).considered_visit == expected.get<std::vector<std::int64_t>>(), "schedule " + key);
        }
        // 2. completed-Q sigma equals the mctx transcription
        for (const auto& c : fx.at("sigma_cases")) {
            const auto got = halving_completed_sigma(c.at("visits").get<std::vector<std::int64_t>>(), c.at("sums").get<std::vector<double>>(),
                                                     c.at("priors").get<std::vector<double>>(), c.at("raw").get<double>(), {});
            const auto want = c.at("sigma").get<std::vector<double>>();
            for (std::size_t i = 0; i < want.size(); ++i) require(std::abs(got[i] - want[i]) < 1e-9 * std::max(1.0, std::abs(want[i])), "sigma case");
        }

        std::ifstream file{PV_FIXTURE}; nlohmann::json fight; file >> fight;
        sts::BattleContext initial; initial.init(stsrl::combat_v4::start_game(fight.at("start")));
        const auto moves = legal_actions(initial); require(moves.size() > 2, "fixture root needs several actions");
        const std::vector<sts::BattleContext> particles(4, initial);
        const auto root_pred = fake(std::array<Inputs, 1>{encode(initial, moves)})[0];

        auto make = [&](bool on, double scale, int m, int batch, std::vector<sts::BattleContext> ps = {}) {
            SearchSettings s; s.batch_size = batch; s.root_halving.enabled = on; s.root_halving.gumbel_scale = scale; s.root_halving.max_considered = m;
            return Search<>{initial, ps.empty() ? particles : ps, root_pred, s};
        };
        // 3. exact budgets and halving visit structure (incl. tiny budgets, m=1, ties with scale 0)
        for (std::int64_t n : {1, 2, 3, 7, 16, 17, 100, 2000})
            for (int batch : {1, 32})
                for (double scale : {0.0, 1.0})
                    for (int m : {1, 2, 3, 16}) {
                        auto search = make(true, scale, m, batch); search.run(fake, n);
                        require(search.simulations() == n, "simulation budget overrun/underrun");
                        auto v = root_visits(search); std::int64_t total = 0; for (auto x : v) total += x;
                        require(total == n, "root visits differ from budget");
                        const int me = std::min<int>(m, int(v.size()));
                        auto sorted = v; std::sort(sorted.rbegin(), sorted.rend()); sorted.resize(std::size_t(me));
                        require(sorted == schedule_counts(halving_schedule(me, n), me), "visit multiset differs from the halving schedule");
                        require(std::count_if(v.begin(), v.end(), [](auto x) { return x > 0; }) <= me, "more actions visited than considered");
                        const auto chosen = search.selected_action();
                        std::int64_t most = *std::max_element(v.begin(), v.end());
                        bool ok = false; for (const auto& e : search.root().edges) ok |= e.action.bits == chosen.bits && e.visits == most;
                        require(ok, "recommendation not among most-visited");
                        const auto pi = search.improved_policy(); double sum = 0; for (double p : pi) sum += p;
                        require(std::abs(sum - 1) < 1e-12, "improved policy not normalized");
                        if (batch == 32 && n >= 100) require(search.telemetry().batches > 0, "batch telemetry");
                    }
        // 4. determinism and hidden-information independence of the Gumbel draw
        {
            auto a = make(true, 1.0, 16, 32); a.run(fake, 300); auto b = make(true, 1.0, 16, 32); b.run(fake, 300);
            require(root_visits(a) == root_visits(b) && a.selected_action().bits == b.selected_action().bits, "not deterministic");
            auto hidden = particles; for (auto& p : hidden) std::reverse(p.cards.drawPile.begin(), p.cards.drawPile.end());
            auto c = make(true, 1.0, 16, 32, hidden);
            require(c.root_gumbel() == a.root_gumbel(), "Gumbel depends on hidden draw order");
            auto z = make(true, 0.0, 16, 32); for (double g : z.root_gumbel()) require(g == 0, "scale 0 not deterministic plain halving");
        }
        // 5. flag off: unchanged default path (most-visited recommendation, no halving state)
        {
            auto off = make(false, 1.0, 16, 32); off.run(fake, 500);
            require(off.simulations() == 500 && off.telemetry().phase_flushes == 0 && off.root_gumbel().empty(), "default path touched");
            bool threw = false; try { (void)off.improved_policy(); } catch (const std::logic_error&) { threw = true; }
            require(threw, "improved_policy must require halving");
        }
        // 6. fail-closed misuse: second run, oracle
        {
            auto s = make(true, 1.0, 16, 32); s.run(fake, 10); bool threw = false;
            try { s.run(fake, 10); } catch (const std::logic_error&) { threw = true; }
            require(threw, "second run must be rejected");
            SearchSettings o; o.oracle = true; o.root_halving.enabled = true; threw = false;
            try { Search<> bad{initial, {initial}, root_pred, o}; } catch (const std::invalid_argument&) { threw = true; }
            require(threw, "oracle halving must be rejected");
        }
        // 7. root Dirichlet noise is rejected (evaluation-only variant)
        {
            SearchSettings o; o.noise_fraction = 0.25; o.root_halving.enabled = true; bool threw = false;
            try { Search<> bad{initial, particles, root_pred, o}; } catch (const std::invalid_argument&) { threw = true; }
            require(threw, "root noise with halving must be rejected");
        }
        // 8. the default is deterministic halving (scale 0): no Gumbel draws, identical across hidden particle orders
        {
            SearchSettings d; d.root_halving.enabled = true; require(d.root_halving.gumbel_scale == 0, "default gumbel_scale must be 0");
            Search<> a{initial, particles, root_pred, d}; for (double g : a.root_gumbel()) require(g == 0, "scale-0 Gumbel draw");
        }
        // 9. extreme finite logits, near-zero priors and equal Q: finite scores/policy, lowest-index ties
        {
            auto extreme = [&](std::span<const Inputs> batch) {
                auto out = fake(batch);
                for (auto& p : out) for (std::size_t a = 0; a < p.logits.size(); ++a) p.logits[a] = a == 0 ? 1e4f : (a % 2 ? -1e4f : 0.f);
                return out;
            };
            auto flat = [&](std::span<const Inputs> batch) {
                auto out = fake(batch);
                for (auto& p : out) { p.value = 50; for (auto& l : p.logits) l = 0; }
                return out;
            };
            for (int which = 0; which < 2; ++which) {
                SearchSettings s; s.root_halving.enabled = true;
                const auto pred = which == 0 ? extreme(std::array<Inputs, 1>{encode(initial, moves)})[0] : flat(std::array<Inputs, 1>{encode(initial, moves)})[0];
                Search<> search{initial, particles, pred, s};
                if (which == 0) search.run(extreme, 200); else search.run(flat, 200);
                require(search.simulations() == 200, "extreme/flat budget");
                const auto pi = search.improved_policy(); double sum = 0;
                for (double x : pi) { require(std::isfinite(x) && x >= 0, "non-finite improved policy"); sum += x; }
                require(std::abs(sum - 1) < 1e-9, "extreme policy normalization");
                if (which == 1) {  // all logits and values equal: every tie resolves to the lowest index
                    require(search.root().edges.front().visits > 0, "equal-score tie not lowest index");
                    require(search.selected_action().bits == search.root().edges.front().action.bits, "equal tie recommendation");
                }
            }
            // near-zero prior: mixed value stays finite when visited actions have underflowing probabilities
            const auto sigma = halving_completed_sigma({3, 0, 1}, {150, 0, 20}, {0.0, 1.0, 0.0}, 40, {});
            for (double x : sigma) require(std::isfinite(x), "near-zero prior sigma");
        }
        // 10. the recommendation is not the raw most-visited argmax: among tied most-visited actions it uses the score
        {
            bool differs = false;
            for (std::int64_t n : {16, 32, 48, 64, 100, 128, 200, 256, 300, 500})
                for (int m : {2, 3, 4, 8, 16}) {
                    auto search = make(true, 0.0, m, 32); search.run(fake, n);
                    const auto& edges = search.root().edges;
                    const auto raw = std::max_element(edges.begin(), edges.end(), [](const auto& a, const auto& b) { return a.visits < b.visits; });
                    differs |= search.selected_action().bits != raw->action.bits;
                }
            require(differs, "no fixture where the halving recommendation differs from the raw visit argmax");
        }
        // 11. eliminating root choices wait for in-flight backups (batch 32 ends early at least once)
        {
            auto search = make(true, 0.0, 16, 32); search.run(fake, 2000);
            require(search.telemetry().phase_flushes > 0, "no elimination flush at batch 32");
            auto one = make(true, 0.0, 16, 1); one.run(fake, 2000);
            require(one.telemetry().phase_flushes == 0, "batch 1 cannot have in-flight paths");
        }
        std::cout << "root halving tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
