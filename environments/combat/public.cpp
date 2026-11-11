#include "environments/combat/public.hpp"
#include "game/Random.h"
#include "sim/search/PublicBeliefCombatSearch.h"
#include <algorithm>
#include <set>
#include <span>
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

std::uint64_t fold(std::uint64_t hash, std::uint64_t value) { return next_seed(hash ^= value); }

// Never position or uniqueId.
std::uint64_t card_identity(const sts::CardInstance& card) {
    std::uint64_t hash = 0;
    for (const std::uint64_t field : {std::uint64_t(card.id), std::uint64_t(card.getUpgradeCount()),
                                      std::uint64_t(card.specialData), std::uint64_t(card.cost),
                                      std::uint64_t(card.costForTurn), std::uint64_t(card.freeToPlayOnce),
                                      std::uint64_t(card.retain)})
        hash = fold(hash, field);
    return hash;
}

// The cards a move's positions refer to; empty when it names none.
std::vector<std::uint64_t> chosen(const sts::BattleContext& state, Action action) {
    using enum sts::search::ActionType;
    const auto type = action.getActionType();
    if (type == CARD) return {card_identity(state.cards.hand[action.getSourceIdx()])};
    if (type != SINGLE_CARD_SELECT && type != MULTI_CARD_SELECT) return {};
    const auto pile = cards_in(state, selection_pile(state));
    if (pile.empty()) return {};  // offers have fixed slots, so their positions stay
    std::vector<std::uint64_t> cards;
    if (type == SINGLE_CARD_SELECT) cards.push_back(card_identity(pile[action.getSelectIdx()]));
    else for (const int i : action.getSelectedIdxs()) cards.push_back(card_identity(pile[i]));
    std::ranges::sort(cards);
    return cards;
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

std::uint64_t action_key(const sts::BattleContext& state, Action action) {
    const auto cards = chosen(state, action);
    auto key = fold(0, cards.empty() ? action.bits : action.bits & ~0xFFFFu);  // positions: the low 16 bits
    for (const auto card : cards) key = fold(key, card);
    return key;
}
std::uint64_t observation_key(const sts::BattleContext& state) { return Public::observationKey(state); }

Pile selection_pile(const sts::BattleContext& state) {
    using enum sts::CardSelectTask;
    switch (state.cardSelectInfo.cardSelectTask) {
    case CODEX: case DISCOVERY: case WISH: return Pile::offered;
    case HEADBUTT: case HOLOGRAM: case LIQUID_MEMORIES_POTION: case MEDITATE: return Pile::discard;
    case EXHUME: return Pile::exhaust;
    case SECRET_TECHNIQUE: case SECRET_WEAPON: case SEEK: case OMNISCIENCE: case SCRY: return Pile::draw;
    default: return Pile::hand;
    }
}

std::span<const sts::CardInstance> cards_in(const sts::BattleContext& state, Pile pile) {
    const auto& cards = state.cards;
    switch (pile) {
    case Pile::hand: return {cards.hand.data(), std::size_t(cards.cardsInHand)};
    case Pile::draw: return cards.drawPile;
    case Pile::discard: return cards.discardPile;
    case Pile::exhaust: return cards.exhaustPile;
    case Pile::offered: return {};
    }
    return {};
}

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
