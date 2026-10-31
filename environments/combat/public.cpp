#include "environments/combat/public.hpp"
#include "game/Random.h"
#include "sim/search/PublicBeliefCombatSearch.h"
#include <algorithm>
#include <set>
#include <stdexcept>
#include <tuple>

namespace stsrl::combat {
namespace {
using Public = sts::search::PublicBeliefCombatSearch;

std::uint64_t next_seed(std::uint64_t& state) {  // splitmix64
    state += 0x9E3779B97F4A7C15ULL;
    auto value = state;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

// Fresh RNG streams and a reshuffled draw pile. The pile is sorted before shuffling so that its sampled
// order does not depend on the hidden true order.
sts::BattleContext particle(const sts::BattleContext& observed, std::uint64_t seed) {
    sts::BattleContext sampled = observed;
    const auto draw_seed = seed;
    const auto next = [&] { return next_seed(seed); };
    sampled.seed = next();
    for (int i = 0; i < 14; ++i) next();
    sampled.aiRng = sts::Random(next());
    sampled.cardRandomRng = sts::Random(next());
    sampled.miscRng = sts::Random(next());
    sampled.monsterHpRng = sts::Random(next());
    sampled.potionRng = sts::Random(next());
    sampled.shuffleRng = sts::Random(next());
    if (sampled.inputState != sts::InputState::CARD_SELECT) {
        std::sort(sampled.cards.drawPile.begin(), sampled.cards.drawPile.end(), [](const auto& a, const auto& b) {
            return std::tie(a.id, a.upgraded, a.specialData, a.cost, a.costForTurn, a.uniqueId)
                 < std::tie(b.id, b.upgraded, b.specialData, b.cost, b.costForTurn, b.uniqueId);
        });
        java::Collections::shuffle(sampled.cards.drawPile.begin(), sampled.cards.drawPile.end(), java::Random(next()));
    }
    Public::resampleDraw(sampled, observed, draw_seed);
    return sampled;
}
}  // namespace

std::vector<Action> legal_actions(const sts::BattleContext& state) {
    std::vector<Action> actions;
    std::set<std::uint64_t> keys;
    for (const auto a : Action::getAllActionsInState(state))
        if (keys.insert(action_key(state, a)).second) actions.push_back(a);
    if (actions.empty()) throw std::runtime_error{"combat: no legal actions"};
    return actions;
}

std::uint64_t action_key(const sts::BattleContext& state, Action action) { return Public::publicActionKey(state, action); }
std::uint64_t observation_key(const sts::BattleContext& state) { return Public::observationKey(state); }

std::vector<sts::BattleContext> particles(const sts::BattleContext& observed, int count, std::uint64_t stream) {
    if (count < 1) throw std::invalid_argument{"particles: count must be >= 1"};
    std::vector<sts::BattleContext> states;
    states.reserve(count);
    for (int i = 0; i < count; ++i) states.push_back(particle(observed, next_seed(stream)));
    return states;
}

std::vector<sts::BattleContext> particles(const sts::BattleContext& observed, int count) {
    return particles(observed, count, Public::publicObservation(observed));
}
}  // namespace stsrl::combat
