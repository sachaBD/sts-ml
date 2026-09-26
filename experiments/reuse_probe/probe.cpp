// Probe: after END_TURN, does the true next observation appear among the particles' next observations?
// Compares ordered-hand observation (current publicObservation) vs hand-sorted observation.
#include "agents/teacher_search.hpp"
#include "scenarios/slime_boss.hpp"

#include <algorithm>
#include <iostream>
#include <set>
#include <tuple>

using namespace stsrl;
using sts::search::PublicBeliefCombatSearch;

static std::uint64_t sorted_obs(sts::BattleContext s) {
    auto* b = s.cards.hand.begin();
    std::sort(b, b + s.cards.cardsInHand, [](const auto& x, const auto& y) {
        return std::make_tuple(int(x.id), x.getUpgradeCount(), x.cost, x.costForTurn, x.specialData)
             < std::make_tuple(int(y.id), y.getUpgradeCount(), y.cost, y.costForTurn, y.specialData);
    });
    return PublicBeliefCombatSearch::publicObservation(s);
}

int main(int argc, char** argv) {
    const int fights = argc > 1 ? std::stoi(argv[1]) : 3;
    const auto run = teacher::guided_rollout_search(2000);
    const int Ns[] = {8, 64, 512};
    for (int seed = 1; seed <= fights; ++seed) {
        auto env = scenarios::slime_boss(seed);
        while (!env.done()) {
            const auto legal = env.decision().legal_actions.size();
            auto choice = teacher::search_decision(env, legal, run);
            const auto before = env.battle();
            const auto bits = env.action_bits(choice.chosen);
            env.step(choice.chosen);
            if (env.done() || sts::search::Action{bits}.getActionType() != sts::search::ActionType::END_TURN) continue;
            const auto& after = env.battle();
            const auto t_ord = PublicBeliefCombatSearch::publicObservation(after);
            const auto t_sort = sorted_obs(after);
            std::cout << "{\"seed\":" << seed << ",\"turn\":" << after.turn << ",\"draw_left\":" << before.cards.drawPile.size();
            for (int n : Ns) {
                auto search = teacher::make_search(before, false, n);
                std::set<std::uint64_t> ord, srt;
                for (auto p : search.particles) {
                    sts::search::Action{bits}.execute(p);
                    ord.insert(PublicBeliefCombatSearch::publicObservation(p));
                    srt.insert(sorted_obs(p));
                }
                std::cout << ",\"n" << n << "\":[" << ord.count(t_ord) << "," << srt.count(t_sort) << "," << ord.size()
                          << "," << srt.size() << "]";
            }
            std::cout << "}" << std::endl;
        }
    }
}
