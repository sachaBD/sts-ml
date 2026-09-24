#include "combat/environment.hpp"

#include "combat/BattleContext.h"
#include "constants/CardPools.h"
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

struct CardMeta { CardType type; TargetType target; int damage; int block; int hits; int draws; int effective_block = -1; };

CardType card_type(const sts::CardInstance& card) {
    switch (sts::getCardType(card.id)) {
    case sts::CardType::ATTACK: return CardType::attack;
    case sts::CardType::SKILL: return CardType::skill;
    case sts::CardType::POWER: return CardType::power;
    case sts::CardType::STATUS: return CardType::status;
    case sts::CardType::CURSE: return CardType::curse;
    default: throw std::runtime_error{"unsupported card type in combat encoding"};
    }
}

// The capability boundary is intentional: deck projection must reject cards
// outside Ironclad, Act-1 colorless, statuses, and curses rather than inventing
// zero mechanics.  `getBaseDamage` supplies the simulator's canonical values;
// dynamic attacks below override it at the live state.
bool supported_card(const sts::CardInstance& card) {
    const auto color = sts::getCardColor(card.id);
    if (color == sts::CardColor::RED || color == sts::CardColor::CURSE || card.getType() == sts::CardType::STATUS) return true;
    using enum sts::CardId;
    switch (card.id) {
    case ANGER: case ARMAMENTS: case BARRICADE: case BASH: case BATTLE_TRANCE: case BLUDGEON: case BODY_SLAM:
    case BRUTALITY: case CARNAGE: case CLEAVE: case CLOTHESLINE: case COMBUST: case CORRUPTION:
    case DARK_EMBRACE: case DEFEND_RED: case DEMON_FORM: case DISARM: case DROPKICK: case ENTRENCH:
    case EVOLVE: case EXHUME: case FEEL_NO_PAIN: case FIEND_FIRE: case FIRE_BREATHING: case FLAME_BARRIER:
    case FLEX: case GHOSTLY_ARMOR: case HEADBUTT: case HEAVY_BLADE: case IMMOLATE: case IMPERVIOUS:
    case INFERNAL_BLADE: case INFLAME: case IRON_WAVE: case METALLICIZE: case OFFERING: case POMMEL_STRIKE:
    case POWER_THROUGH: case PUMMEL: case RAGE: case RAMPAGE: case SECOND_WIND: case SHOCKWAVE:
    case SHRUG_IT_OFF: case SPOT_WEAKNESS: case STRIKE_RED: case SWORD_BOOMERANG: case THUNDERCLAP:
    case TRUE_GRIT: case UPPERCUT: case WHIRLWIND: case WILD_STRIKE:
    case BANDAGE_UP: case BLIND: case DARK_SHACKLES: case DEEP_BREATH: case DISCOVERY:
    case DRAMATIC_ENTRANCE: case ENLIGHTENMENT: case FINESSE: case FLASH_OF_STEEL:
    case HAND_OF_GREED: case IMPATIENCE: case JACK_OF_ALL_TRADES: case MADNESS:
    case MASTER_OF_STRATEGY: case METAMORPHOSIS: case MIND_BLAST: case PANACEA:
    case PANIC_BUTTON: case PURITY: case SECRET_TECHNIQUE: case SECRET_WEAPON:
    case THE_BOMB: case THINKING_AHEAD: case TRANSMUTATION: case TRIP: case VIOLENCE:
        return true;
    default: return std::ranges::find(sts::baseColorlessPool, card.id) != sts::baseColorlessPool.end();
    }
}

CardMeta card_meta(const sts::BattleContext& state, const sts::CardInstance& card) {
    if (!supported_card(card)) throw std::runtime_error{"unsupported card capability: " + std::string{sts::getCardEnumName(card.id)}};
    const bool up = card.upgraded;
    CardMeta meta{card_type(card), card.requiresTarget() ? TargetType::one_enemy : TargetType::none,
                  sts::getBaseDamage(card.id, up), 0, 1, 0};
    if (meta.damage < 0) meta.damage = 0;
    using enum sts::CardId;
    switch (card.id) {
    case CLEAVE: case IMMOLATE: case REAPER: case THUNDERCLAP: case DRAMATIC_ENTRANCE:
        meta.target = TargetType::all_enemies; break;
    case SWORD_BOOMERANG: meta.target = TargetType::random_enemy; meta.damage = 3; meta.hits = up ? 4 : 3; break;
    case WHIRLWIND: meta.target = TargetType::all_enemies; meta.damage = up ? 8 : 5; meta.hits = std::max(0, state.player.energy); break;
    case PUMMEL: meta.damage = 2; meta.hits = up ? 5 : 4; break;
    case TWIN_STRIKE: meta.damage = up ? 7 : 5; meta.hits = 2; break;
    case FIEND_FIRE: meta.damage = up ? 10 : 7; meta.hits = std::max(0, state.cards.cardsInHand - 1); break;
    case BODY_SLAM: meta.damage = state.player.block; break;
    case HEAVY_BLADE: meta.damage = 14 + (up ? 4 : 2) * state.player.getStatus<PS::STRENGTH>(); break;
    case PERFECTED_STRIKE: meta.damage = 6 + state.cards.strikeCount * (up ? 3 : 2); break;
    case RAMPAGE: meta.damage = 8 + card.specialData; break;
    case RITUAL_DAGGER: meta.damage = card.specialData; break;
    case SEARING_BLOW: { const int n = card.getUpgradeCount(); meta.damage = n * (n + 7) / 2 + 12; break; }
    case MIND_BLAST: meta.damage = static_cast<int>(state.cards.drawPile.size()); break;
    case DEFEND_RED: meta.block = up ? 8 : 5; break;
    case ARMAMENTS: meta.block = 5; break;
    case IRON_WAVE: meta.block = up ? 7 : 5; meta.effective_block = state.calculateCardBlock(state.calculateCardBlock(meta.block)); break;
    case FLAME_BARRIER: meta.block = up ? 16 : 12; break;
    case GHOSTLY_ARMOR: meta.block = up ? 13 : 10; break;
    case IMPERVIOUS: meta.block = up ? 40 : 30; break;
    case POWER_THROUGH: meta.block = up ? 20 : 15; break;
    case SECOND_WIND: meta.block = up ? 7 : 5; break;
    case SHRUG_IT_OFF: meta.block = up ? 11 : 8; meta.draws = 1; break;
    case TRUE_GRIT: meta.block = up ? 9 : 7; break;
    case BATTLE_TRANCE: meta.draws = up ? 4 : 3; break;
    case DEEP_BREATH: meta.draws = up ? 2 : 1; break;
    case OFFERING: meta.draws = up ? 5 : 3; break;
    case POMMEL_STRIKE: meta.draws = up ? 2 : 1; break;
    case DROPKICK: meta.draws = std::ranges::any_of(state.monsters.arr, [](const auto& monster) { return monster.isAlive() && monster.vulnerable > 0; }) ? 1 : 0; break;
    case BURNING_PACT: meta.draws = up ? 3 : 2; break;
    case WARCRY: meta.draws = up ? 2 : 1; break;
    case MASTER_OF_STRATEGY: meta.draws = up ? 4 : 3; break;
    case THINKING_AHEAD: meta.draws = 2; break;
    case IMPATIENCE: meta.draws = up ? 3 : 2; break; // conditional on no attack in hand.
    case SENTINEL: meta.block = up ? 8 : 5; break;
    case GOOD_INSTINCTS: meta.block = up ? 9 : 6; break;
    case PANIC_BUTTON: meta.block = up ? 40 : 30; break;
    case FLASH_OF_STEEL: meta.draws = 1; break;
    case FINESSE: meta.block = up ? 4 : 2; meta.draws = 1; break;
    default: break;
    }
    return meta;
}

CardToken encode_card(const sts::BattleContext& state, const sts::CardInstance& card, CardZone zone, bool playable_now) {
    const auto meta = card_meta(state, card);
    return {static_cast<int>(card.id), zone, meta.type, meta.target,
        {float(card.upgraded), card.cost / 3.f, card.costForTurn / 3.f, meta.damage / 50.f, meta.hits / 10.f,
         meta.block / 50.f, (meta.effective_block >= 0 ? meta.effective_block : state.calculateCardBlock(meta.block)) / 50.f, meta.draws / 10.f, card.specialData / 10.f,
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
    if (state.inputState != sts::InputState::CARD_SELECT) {
        zone = CardZone::hand;
        return nullptr;
    }
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
    const int select_task = impl_->state.inputState == sts::InputState::CARD_SELECT
        ? static_cast<int>(impl_->state.cardSelectInfo.cardSelectTask)
        : static_cast<int>(sts::CardSelectTask::INVALID);
    result.encoding.global = {{impl_->state.turn / 20.f, player.curHp / 100.f,
        player.maxHp ? player.curHp / float(player.maxHp) : 0.f, player.block / 100.f, player.energy / 10.f,
        player.energyPerTurn / 10.f, player.strength / 10.f, player.dexterity / 10.f,
        player.getStatusRuntime(PlayerStatus::WEAK) / 10.f, player.getStatusRuntime(PlayerStatus::VULNERABLE) / 10.f,
        player.getStatusRuntime(PlayerStatus::FRAIL) / 10.f, player.cardsPlayedThisTurn / 20.f,
        impl_->state.cards.cardsInHand / 10.f, impl_->state.cards.drawPile.size() / 64.f,
        impl_->state.cards.discardPile.size() / 64.f, impl_->state.cards.exhaustPile.size() / 64.f,
        incoming / 100.f, std::max(0, incoming - player.block) / 100.f,
        player.getStatusRuntime(PlayerStatus::COMBUST) / 10.f, player.combustHpLoss / 10.f,
        player.getStatusRuntime(PlayerStatus::FLAME_BARRIER) / 50.f, float(player.hasStatus<PlayerStatus::NO_DRAW>()),
        player.getStatusRuntime(PlayerStatus::INTANGIBLE) / 10.f, player.getStatusRuntime(PlayerStatus::ARTIFACT) / 10.f,
        float(player.hasStatus<PlayerStatus::BARRICADE>()), float(player.hasStatus<PlayerStatus::CORRUPTION>()),
        player.getStatusRuntime(PlayerStatus::BRUTALITY) / 10.f, player.getStatusRuntime(PlayerStatus::DEMON_FORM) / 10.f,
        player.getStatusRuntime(PlayerStatus::DARK_EMBRACE) / 10.f, player.getStatusRuntime(PlayerStatus::EVOLVE) / 10.f,
        player.getStatusRuntime(PlayerStatus::FEEL_NO_PAIN) / 10.f, player.getStatusRuntime(PlayerStatus::METALLICIZE) / 10.f,
        player.getStatusRuntime(PlayerStatus::RAGE) / 10.f, player.getStatusRuntime(PlayerStatus::DOUBLE_TAP) / 10.f,
        player.getStatusRuntime(PlayerStatus::VIGOR) / 10.f, player.bomb1 / 50.f, player.bomb2 / 50.f, player.bomb3 / 50.f,
        player.getStatusRuntime(PlayerStatus::NO_BLOCK) / 10.f, player.getStatusRuntime(PlayerStatus::LOSE_STRENGTH) / 10.f,
        player.getStatusRuntime(PlayerStatus::LOSE_DEXTERITY) / 10.f, player.getStatusRuntime(PlayerStatus::ENERGIZED) / 10.f,
        player.getStatusRuntime(PlayerStatus::FIRE_BREATHING) / 10.f, player.getStatusRuntime(PlayerStatus::JUGGERNAUT) / 10.f,
        player.getStatusRuntime(PlayerStatus::RUPTURE) / 10.f, player.getStatusRuntime(PlayerStatus::MAGNETISM) / 10.f,
        player.getStatusRuntime(PlayerStatus::MAYHEM) / 10.f, player.getStatusRuntime(PlayerStatus::PANACHE) / 10.f,
        player.panacheCounter / 20.f, player.getStatusRuntime(PlayerStatus::SADISTIC) / 10.f},
        static_cast<int>(impl_->state.inputState), select_task};
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
        const auto meta = card_meta(impl_->state, *source.card);
        if (source.hand_index < 0 || meta.damage == 0 || meta.target == TargetType::random_enemy) continue;
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
                          .card_selection_task = select_task,
                          .skips_selection = impl_->state.inputState == sts::InputState::CARD_SELECT &&
                              edge.action.getActionType() == sts::search::ActionType::SINGLE_CARD_SELECT &&
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
            const auto task = impl_->state.inputState == sts::InputState::CARD_SELECT
                ? impl_->state.cardSelectInfo.cardSelectTask
                : sts::CardSelectTask::INVALID;
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
        const int select_task = impl_->state.inputState == sts::InputState::CARD_SELECT
            ? static_cast<int>(impl_->state.cardSelectInfo.cardSelectTask)
            : static_cast<int>(sts::CardSelectTask::INVALID);
        SearchActionKey key{.kind = static_cast<int>(action.getActionType()), .selection_task = select_task};
        bool target_required = false;
        if (action.getActionType() == sts::search::ActionType::CARD) {
            const auto& card = impl_->state.cards.hand[action.getSourceIdx()];
            key.card_id = static_cast<int>(card.id); key.upgraded = card.upgraded; key.cost = card.cost; key.cost_for_turn = card.costForTurn;
            key.special = card.specialData; key.free = card.freeToPlayOnce; key.retain = card.retain; target_required = card.requiresTarget();
        } else if (action.getActionType() == sts::search::ActionType::POTION) {
            key.potion = static_cast<int>(impl_->state.potions[action.getSourceIdx()]);
            target_required = potionRequiresTarget(impl_->state.potions[action.getSourceIdx()]);
        } else if (action.getActionType() == sts::search::ActionType::SINGLE_CARD_SELECT) {
            CardZone zone{};
            key.selection_count = 1;
            if (const auto* card = selected_card(impl_->state, action, zone)) {
                key.card_id = static_cast<int>(card->id); key.upgraded = card->upgraded;
                key.cost = card->cost; key.cost_for_turn = card->costForTurn; key.special = card->specialData;
                key.free = card->freeToPlayOnce; key.retain = card->retain;
                key.selected_cards[0] = static_cast<int>(card->id);
            } else if (impl_->state.cardSelectInfo.cardSelectTask == sts::CardSelectTask::CODEX && action.getSelectIdx() >= 0 && action.getSelectIdx() < 3) {
                zone = CardZone::offered;
                key.card_id = static_cast<int>(impl_->state.cardSelectInfo.codexCards()[action.getSelectIdx()]);
                key.selected_cards[0] = key.card_id;
            } else if (impl_->state.cardSelectInfo.cardSelectTask == sts::CardSelectTask::DISCOVERY && action.getSelectIdx() >= 0 && action.getSelectIdx() < 3) {
                zone = CardZone::offered;
                key.card_id = static_cast<int>(impl_->state.cardSelectInfo.discovery_Cards()[action.getSelectIdx()]);
                key.selected_cards[0] = key.card_id;
            }
            key.selection_zone = static_cast<int>(zone);
        } else if (action.getActionType() == sts::search::ActionType::MULTI_CARD_SELECT) {
            const auto selected = action.getSelectedIdxs();
            key.selection_count = static_cast<int>(selected.size());
            key.selection_zone = static_cast<int>(CardZone::hand);
            for (int i = 0; i < key.selection_count && i < static_cast<int>(key.selected_cards.size()); ++i) {
                const int idx = selected[i];
                if (idx >= 0 && idx < impl_->state.cards.cardsInHand) {
                    const auto& card = impl_->state.cards.hand[idx];
                    key.selected_cards[i] = static_cast<int>(card.id);
                    key.selected_upgraded[i] = card.upgraded;
                    key.selected_cost[i] = card.costForTurn;
                    key.selected_special[i] = card.specialData;
                }
            }
            std::array<std::array<int, 4>, 10> selected_semantics{};
            for (int i = 0; i < key.selection_count && i < static_cast<int>(selected_semantics.size()); ++i)
                selected_semantics[i] = {key.selected_cards[i], key.selected_upgraded[i], key.selected_cost[i], key.selected_special[i]};
            std::sort(selected_semantics.begin(), selected_semantics.begin() + std::min(key.selection_count, static_cast<int>(selected_semantics.size())));
            for (int i = 0; i < key.selection_count && i < static_cast<int>(selected_semantics.size()); ++i) {
                key.selected_cards[i] = selected_semantics[i][0]; key.selected_upgraded[i] = selected_semantics[i][1];
                key.selected_cost[i] = selected_semantics[i][2]; key.selected_special[i] = selected_semantics[i][3];
            }
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

CombatEnvironment CombatEnvironment::clone() const { return CombatEnvironment{impl_->state}; }

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

const sts::BattleContext& CombatEnvironment::battle() const noexcept { return impl_->state; }

std::uint32_t CombatEnvironment::action_bits(const std::size_t action_index) const {
    return impl_->current_actions.at(action_index).bits;
}

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
