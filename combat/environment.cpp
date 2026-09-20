#include "combat/environment.hpp"

#include "combat/BattleContext.h"
#include "sim/search/BattleScumSearcher2.h"

#include <algorithm>
#include <array>
#include <random>
#include <tuple>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace stsrl {
namespace {

struct CardMeta { CardType type; TargetType target; int damage; int block; int hits; int draws; };

CardMeta card_meta(const sts::CardInstance& card) {
    using enum sts::CardId;
    switch (card.id) {
    case STRIKE_RED: return {CardType::attack, TargetType::one_enemy, card.upgraded ? 9 : 6, 0, 1, 0};
    case DEFEND_RED: return {CardType::skill, TargetType::none, 0, card.upgraded ? 8 : 5, 0, 0};
    case BASH: return {CardType::attack, TargetType::one_enemy, card.upgraded ? 10 : 8, 0, 1, 0};
    case SLIMED: return {CardType::status, TargetType::none, 0, 0, 0, 0};
    case FLAME_BARRIER: return {CardType::skill, TargetType::none, 0, card.upgraded ? 16 : 12, 0, 0};
    case COMBUST: return {CardType::power, TargetType::none, 0, 0, 0, 0};
    case HEMOKINESIS: return {CardType::attack, TargetType::one_enemy, card.upgraded ? 20 : 15, 0, 1, 0};
    case BATTLE_TRANCE: return {CardType::skill, TargetType::none, 0, 0, 0, card.upgraded ? 4 : 3};
    default: throw std::runtime_error{"unsupported card in combat encoding"};
    }
}

CardToken encode_card(const sts::BattleContext& state, const sts::CardInstance& card, CardZone zone, bool playable_now) {
    const auto meta = card_meta(card);
    return {static_cast<int>(card.id), zone, meta.type, meta.target,
        {float(card.upgraded), card.cost / 3.f, card.costForTurn / 3.f, meta.damage / 50.f, meta.hits / 10.f,
         meta.block / 50.f, state.calculateCardBlock(meta.block) / 50.f, meta.draws / 10.f, card.specialData / 10.f,
         float(card.freeToPlayOnce), float(card.doesExhaust()), float(card.isEthereal()), float(card.retain), float(playable_now)}};
}

MonsterToken encode_monster(const sts::BattleContext& state, const sts::Monster& monster) {
    const auto damage = monster.getMoveBaseDamage(state);
    const int per_hit = monster.calculateDamageToPlayer(state, damage.damage);
    return {static_cast<int>(monster.id), static_cast<int>(monster.moveHistory[0]),
        {monster.curHp / 100.f, monster.maxHp ? monster.curHp / float(monster.maxHp) : 0.f, monster.block / 100.f,
         per_hit * damage.attackCount / 100.f, damage.attackCount / 10.f, monster.strength / 10.f,
         monster.weak / 10.f, monster.vulnerable / 10.f, float(monster.isTargetable())}};
}

bool card_less(const CardToken& a, const CardToken& b) { return std::tie(a.card_id, a.zone, a.card_type, a.target_type, a.numeric) < std::tie(b.card_id, b.zone, b.card_type, b.target_type, b.numeric); }
bool monster_less(const MonsterToken& a, const MonsterToken& b) { return std::tie(a.monster_id, a.move_id, a.numeric) < std::tie(b.monster_id, b.move_id, b.numeric); }

const sts::CardInstance* selected_card(const sts::BattleContext& state, const sts::search::Action& action,
                                       CardZone& zone) {
    const auto index = action.getSelectIdx();
    switch (state.cardSelectInfo.cardSelectTask) {
    case sts::CardSelectTask::CODEX:
    case sts::CardSelectTask::DISCOVERY:
        zone = CardZone::offered;
        return nullptr;
    case sts::CardSelectTask::HOLOGRAM: case sts::CardSelectTask::LIQUID_MEMORIES_POTION:
    case sts::CardSelectTask::HEADBUTT: case sts::CardSelectTask::MEDITATE:
        zone = CardZone::discard; return index >= 0 && index < static_cast<int>(state.cards.discardPile.size()) ? &state.cards.discardPile[index] : nullptr;
    case sts::CardSelectTask::EXHUME:
        zone = CardZone::exhaust; return index >= 0 && index < static_cast<int>(state.cards.exhaustPile.size()) ? &state.cards.exhaustPile[index] : nullptr;
    case sts::CardSelectTask::SECRET_TECHNIQUE: case sts::CardSelectTask::SECRET_WEAPON: case sts::CardSelectTask::SEEK:
        zone = CardZone::draw; return index >= 0 && index < static_cast<int>(state.cards.drawPile.size()) ? &state.cards.drawPile[index] : nullptr;
    default:
        zone = CardZone::hand; return index >= 0 && index < state.cards.cardsInHand ? &state.cards.hand[index] : nullptr;
    }
}

} // namespace

struct CombatEnvironment::Impl {
    explicit Impl(sts::BattleContext initial_state)
        : state{std::move(initial_state)}, enumerator{state} {}

    sts::BattleContext state;
    sts::search::BattleScumSearcher2 enumerator;
    std::vector<sts::search::Action> current_actions;
};

CombatEnvironment::CombatEnvironment(sts::BattleContext initial_state)
    : impl_{std::make_unique<Impl>(std::move(initial_state))} {}

CombatEnvironment::~CombatEnvironment() = default;
CombatEnvironment::CombatEnvironment(CombatEnvironment&&) noexcept = default;
CombatEnvironment& CombatEnvironment::operator=(CombatEnvironment&&) noexcept = default;

Decision CombatEnvironment::decision() {
    sts::search::BattleScumSearcher2::Node node;
    impl_->enumerator.enumerateActionsForNode(node, impl_->state);

    impl_->current_actions.clear();
    impl_->current_actions.reserve(node.edges.size());

    Decision result;
    const auto& player = impl_->state.player;
    const auto& enemy = impl_->state.monsters.arr.front();
    result.observation = {
        .turn = impl_->state.turn,
        .player_hp = player.curHp,
        .player_max_hp = player.maxHp,
        .player_block = player.block,
        .energy = player.energy,
        .enemy_hp = enemy.curHp,
        .enemy_max_hp = enemy.maxHp,
        .enemy_block = enemy.block,
    };
    int incoming = 0;
    for (const auto& monster : impl_->state.monsters.arr) if (monster.id != sts::MonsterId::INVALID && monster.isAlive()) {
        const auto damage = monster.getMoveBaseDamage(impl_->state);
        incoming += monster.calculateDamageToPlayer(impl_->state, damage.damage) * damage.attackCount;
    }
    result.encoding.global = {{impl_->state.turn / 20.f, player.curHp / 100.f,
        player.maxHp ? player.curHp / float(player.maxHp) : 0.f, player.block / 100.f, player.energy / 10.f,
        player.energyPerTurn / 10.f, player.strength / 10.f, player.dexterity / 10.f,
        player.getStatusRuntime(PlayerStatus::WEAK) / 10.f, player.getStatusRuntime(PlayerStatus::VULNERABLE) / 10.f,
        player.getStatusRuntime(PlayerStatus::FRAIL) / 10.f, player.cardsPlayedThisTurn / 20.f,
        impl_->state.cards.cardsInHand / 10.f, impl_->state.cards.drawPile.size() / 64.f,
        impl_->state.cards.discardPile.size() / 64.f, impl_->state.cards.exhaustPile.size() / 64.f,
        incoming / 100.f, std::max(0, incoming - player.block) / 100.f,
        player.getStatusRuntime(PlayerStatus::COMBUST) / 10.f, player.combustHpLoss / 10.f,
        player.getStatusRuntime(PlayerStatus::FLAME_BARRIER) / 50.f, float(player.hasStatus<PlayerStatus::NO_DRAW>())},
        static_cast<int>(impl_->state.inputState), static_cast<int>(impl_->state.cardSelectInfo.cardSelectTask)};
    struct PendingCard { CardToken token; int hand_index = -1; const sts::CardInstance* card = nullptr; };
    struct PendingMonster { MonsterToken token; int slot; const sts::Monster* monster; };
    std::vector<PendingCard> cards;
    for (int i = 0; i < impl_->state.cards.cardsInHand; ++i)
        cards.push_back({encode_card(impl_->state, impl_->state.cards.hand[i], CardZone::hand, impl_->state.cards.hand[i].canUseOnAnyTarget(impl_->state)), i, &impl_->state.cards.hand[i]});
    for (const auto& card : impl_->state.cards.drawPile) cards.push_back({encode_card(impl_->state, card, CardZone::draw, false), -1, &card});
    for (const auto& card : impl_->state.cards.discardPile) cards.push_back({encode_card(impl_->state, card, CardZone::discard, false), -1, &card});
    for (const auto& card : impl_->state.cards.exhaustPile) cards.push_back({encode_card(impl_->state, card, CardZone::exhaust, false), -1, &card});
    std::sort(cards.begin(), cards.end(), [](const auto& a, const auto& b) { return card_less(a.token, b.token); });
    for (const auto& card : cards) result.encoding.cards.push_back(card.token);
    std::vector<PendingMonster> monsters;
    for (int slot = 0; slot < static_cast<int>(impl_->state.monsters.arr.size()); ++slot) {
        const auto& monster = impl_->state.monsters.arr[slot];
        if (monster.id != sts::MonsterId::INVALID && monster.isAlive()) monsters.push_back({encode_monster(impl_->state, monster), slot, &monster});
    }
    std::sort(monsters.begin(), monsters.end(), [](const auto& a, const auto& b) { return monster_less(a.token, b.token); });
    for (const auto& monster : monsters) result.encoding.monsters.push_back(monster.token);
    for (std::size_t ci = 0; ci < cards.size(); ++ci) {
        const auto& source = cards[ci];
        const auto meta = card_meta(*source.card);
        if (source.hand_index < 0 || meta.damage == 0) continue;
        for (std::size_t mi = 0; mi < monsters.size(); ++mi) {
            const auto& target = monsters[mi];
            if (!target.monster->isTargetable()) continue;
            const int per_hit = impl_->state.calculateCardDamage(*source.card, target.slot, meta.damage);
            const int damage = per_hit * meta.hits;
            const int hp_damage = std::min(target.monster->curHp, std::max(0, damage - target.monster->block));
            const int remaining = target.monster->curHp - hp_damage;
            result.encoding.card_monster_interactions.push_back({std::uint16_t(ci), std::uint8_t(mi),
                {meta.hits / 10.f, damage / 100.f, hp_damage / 100.f, remaining / 100.f,
                 target.monster->maxHp ? remaining / float(target.monster->maxHp) : 0.f,
                 float(source.card->canUse(impl_->state, target.slot, false))}});
        }
    }
    result.legal_actions.reserve(node.edges.size());
    result.encoding.legal_actions.reserve(node.edges.size());

    for (auto& edge : node.edges) {
        const auto index = impl_->current_actions.size();
        std::ostringstream description;
        edge.action.printDesc(description, impl_->state);
        impl_->current_actions.push_back(edge.action);
        result.legal_actions.push_back({index, std::move(description).str()});
        ActionToken token{.kind = static_cast<EncodedActionKind>(edge.action.getActionType()),
                          .source_card = std::nullopt, .target_monster = std::nullopt, .potion_id = std::nullopt,
                          .card_selection_task = static_cast<int>(impl_->state.cardSelectInfo.cardSelectTask),
                          .skips_selection = edge.action.getActionType() == sts::search::ActionType::SINGLE_CARD_SELECT &&
                              impl_->state.cardSelectInfo.cardSelectTask == sts::CardSelectTask::CODEX && edge.action.getSelectIdx() == 3,
                          .execution_index = index};
        if (edge.action.getActionType() == sts::search::ActionType::CARD) {
            const auto source = edge.action.getSourceIdx();
            if (source >= 0 && source < impl_->state.cards.cardsInHand)
                token.source_card = encode_card(impl_->state, impl_->state.cards.hand[source], CardZone::hand, true);
        } else if (edge.action.getActionType() == sts::search::ActionType::POTION) {
            const auto source = edge.action.getSourceIdx();
            if (source >= 0 && source < static_cast<int>(impl_->state.potions.size()))
                token.potion_id = static_cast<int>(impl_->state.potions[source]);
        } else if (edge.action.getActionType() == sts::search::ActionType::SINGLE_CARD_SELECT) {
            const auto task = impl_->state.cardSelectInfo.cardSelectTask;
            const auto selected = edge.action.getSelectIdx();
            if ((task == sts::CardSelectTask::CODEX || task == sts::CardSelectTask::DISCOVERY) && selected >= 0 && selected < 3) {
                const auto id = task == sts::CardSelectTask::CODEX ? impl_->state.cardSelectInfo.codexCards()[selected]
                                                                    : impl_->state.cardSelectInfo.discovery_Cards()[selected];
                token.source_card = CardToken{.card_id = static_cast<int>(id), .zone = CardZone::offered};
            } else {
                CardZone zone{};
                if (const auto* card = selected_card(impl_->state, edge.action, zone)) token.source_card = encode_card(impl_->state, *card, zone, false);
            }
        }
        const bool requires_target = edge.action.getActionType() == sts::search::ActionType::CARD
            ? impl_->state.cards.hand[edge.action.getSourceIdx()].requiresTarget()
            : edge.action.getActionType() == sts::search::ActionType::POTION
                && potionRequiresTarget(impl_->state.potions[edge.action.getSourceIdx()]);
        if (requires_target && edge.action.getTargetIdx() >= 0 && edge.action.getTargetIdx() < static_cast<int>(impl_->state.monsters.arr.size()))
            token.target_monster = encode_monster(impl_->state, impl_->state.monsters.arr[edge.action.getTargetIdx()]);
        result.encoding.legal_actions.push_back(std::move(token));
    }
    std::sort(result.encoding.legal_actions.begin(), result.encoding.legal_actions.end(), [](const ActionToken& a, const ActionToken& b) {
        if (a.kind != b.kind) return a.kind < b.kind;
        if (a.source_card.has_value() != b.source_card.has_value()) return !a.source_card.has_value();
        if (a.source_card && *a.source_card != *b.source_card) return card_less(*a.source_card, *b.source_card);
        if (a.target_monster.has_value() != b.target_monster.has_value()) return !a.target_monster.has_value();
        if (a.target_monster && *a.target_monster != *b.target_monster) return monster_less(*a.target_monster, *b.target_monster);
        return std::tie(a.potion_id, a.card_selection_task, a.skips_selection, a.execution_index)
             < std::tie(b.potion_id, b.card_selection_task, b.skips_selection, b.execution_index);
    });

    return result;
}

std::vector<SearchAction> CombatEnvironment::search_actions() {
    sts::search::BattleScumSearcher2::Node node;
    impl_->enumerator.enumerateActionsForNode(node, impl_->state);
    impl_->current_actions.clear();
    std::vector<SearchAction> result;
    for (const auto& edge : node.edges) {
        const auto& action = edge.action;
        SearchActionKey key{.kind = static_cast<int>(action.getActionType()), .selection_task = static_cast<int>(impl_->state.cardSelectInfo.cardSelectTask)};
        bool target_required = false;
        if (action.getActionType() == sts::search::ActionType::CARD) {
            const auto& card = impl_->state.cards.hand[action.getSourceIdx()];
            key.card_id = static_cast<int>(card.id); key.upgraded = card.upgraded; key.cost = card.cost; key.cost_for_turn = card.costForTurn;
            key.special = card.specialData; key.free = card.freeToPlayOnce; key.retain = card.retain; target_required = card.requiresTarget();
        } else if (action.getActionType() == sts::search::ActionType::POTION) {
            key.potion = static_cast<int>(impl_->state.potions[action.getSourceIdx()]);
            target_required = potionRequiresTarget(impl_->state.potions[action.getSourceIdx()]);
        } else if (action.getActionType() == sts::search::ActionType::SINGLE_CARD_SELECT || action.getActionType() == sts::search::ActionType::MULTI_CARD_SELECT) {
            throw std::logic_error{"card selection is unsupported by lightweight search"};
        }
        if (target_required && action.getTargetIdx() >= 0 && action.getTargetIdx() < static_cast<int>(impl_->state.monsters.arr.size())) {
            const auto& monster = impl_->state.monsters.arr[action.getTargetIdx()];
            key.target_id = static_cast<int>(monster.id); key.target_hp = monster.curHp; key.target_max_hp = monster.maxHp; key.target_block = monster.block;
            key.target_move = static_cast<int>(monster.moveHistory[0]); key.target_strength = monster.strength;
            key.target_weak = monster.weak; key.target_vulnerable = monster.vulnerable;
        }
        if (std::ranges::any_of(result, [&key](const SearchAction& x) { return x.key == key; })) continue;
        const auto index = impl_->current_actions.size();
        impl_->current_actions.push_back(action);
        result.push_back({index, key});
    }
    return result;
}

std::string CombatEnvironment::action_description(const std::size_t action_index) const {
    if (action_index >= impl_->current_actions.size()) {
        throw std::out_of_range{"action index is outside the current legal action set"};
    }
    std::ostringstream description;
    impl_->current_actions[action_index].printDesc(description, impl_->state);
    return description.str();
}

CombatEnvironment CombatEnvironment::determinized(const std::uint64_t seed) const {
    auto state = impl_->state;
    std::mt19937_64 random{seed};

    // The remaining draw-pile order and future RNG states are not observable to
    // the player. Sample them independently for each information-set search.
    std::shuffle(state.cards.drawPile.begin(), state.cards.drawPile.end(), random);
    state.aiRng = sts::Random{random()};
    state.cardRandomRng = sts::Random{random()};
    state.miscRng = sts::Random{random()};
    state.monsterHpRng = sts::Random{random()};
    state.potionRng = sts::Random{random()};
    state.shuffleRng = sts::Random{random()};

    return CombatEnvironment{std::move(state)};
}

void CombatEnvironment::step(const std::size_t action_index) {
    if (done()) {
        throw std::logic_error{"cannot step a finished combat"};
    }
    if (action_index >= impl_->current_actions.size()) {
        throw std::out_of_range{"action index is outside the current legal action set"};
    }

    impl_->current_actions[action_index].execute(impl_->state);
    impl_->current_actions.clear();
}

bool CombatEnvironment::done() const noexcept {
    return impl_->state.outcome != sts::Outcome::UNDECIDED;
}

bool CombatEnvironment::won() const noexcept {
    return impl_->state.outcome == sts::Outcome::PLAYER_VICTORY;
}

int CombatEnvironment::player_hp() const noexcept { return impl_->state.player.curHp; }
int CombatEnvironment::player_max_hp() const noexcept { return impl_->state.player.maxHp; }

double CombatEnvironment::combat_value() const noexcept {
    const auto player_hp_fraction = static_cast<double>(impl_->state.player.curHp)
        / static_cast<double>(impl_->state.player.maxHp);

    if (won()) {
        return 0.5 + 0.5 * player_hp_fraction;
    }
    if (impl_->state.outcome == sts::Outcome::PLAYER_LOSS) {
        return -1.0;
    }

    int total_enemy_hp = 0;
    int total_enemy_max_hp = 0;
    for (const auto& monster : impl_->state.monsters.arr) {
        if (monster.isAlive() && monster.maxHp > 0) {
            total_enemy_hp += monster.curHp;
            total_enemy_max_hp += monster.maxHp;
        }
    }
    const auto enemy_hp_fraction = total_enemy_max_hp == 0 ? 0.0
        : static_cast<double>(total_enemy_hp) / static_cast<double>(total_enemy_max_hp);
    return 0.25 * (player_hp_fraction - enemy_hp_fraction);
}

}  // namespace stsrl
