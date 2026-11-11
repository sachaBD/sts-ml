// Combat features (features.hpp).
//
// Coverage: Ironclad, act 1 (all act-1 monsters, Ironclad + colorless + event cards, potions Ironclad can hold, any
// relic in the combat state). Each table below says where to extend it for other characters / acts. Trained models
// read these values: any layout or scaling change is a new encoding, never an edit.
//
// Scaling: counts that can grow without bound use symlog (sign * log1p|x|), flags are 0 / 1, and potion
// magnitudes are divided by a fixed typical size.
#include "agents/combat/nn/features.hpp"
#include "environments/combat/public.hpp"

#include "combat/BattleContext.h"
#include "constants/MonsterStatusEffects.h"
#include "constants/PlayerStatusEffects.h"
#include "constants/Potions.h"
#include "constants/Relics.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include "constants/CardPools.h"
#include "sim/search/BattleScumSearcher2.h"
#include <optional>

namespace stsrl::nn {

namespace {

float symlog(float x) { return std::copysign(std::log1p(std::abs(x)), x); }

// Player status amount (boolean statuses, stored only as a bit, count 1).
int status(const sts::Player& player, PlayerStatus s) {
    if (!player.hasStatusRuntime(s)) return 0;
    if (const auto it = player.statusMap.find(s); it != player.statusMap.end()) return it->second;
    return 1;
}

}  // namespace

// ---- player: statuses act-1 Ironclad can get that the v3 global numeric lacks ----
// EXTEND HERE (and player_status_features): other characters' powers (focus, mantra, orbs, stances, ...) and
// later acts' debuffs (hex, constricted, draw reduction, fasting, ...).
void player_extras(const sts::BattleContext& state, GlobalFeatures& global) {
    const auto& p = state.player;
    int held = 0;
    for (int i = 0; i < state.potionCapacity; ++i)
        if (state.potions[i] != sts::Potion::EMPTY_POTION_SLOT && state.potions[i] != sts::Potion::INVALID) ++held;
    global.player = {
        float(status(p, PlayerStatus::CONFUSED) > 0),   // Snecko Oil
        symlog(status(p, PlayerStatus::ENTANGLED)),      // Red Slaver
        float(status(p, PlayerStatus::PEN_NIB) > 0),     // Pen Nib: the next attack deals double
        symlog(status(p, PlayerStatus::DUPLICATION)),    // Duplication Potion
        symlog(status(p, PlayerStatus::REGEN)),          // Regen Potion
        symlog(status(p, PlayerStatus::RITUAL)),         // Cultist Potion
        symlog(status(p, PlayerStatus::THORNS)),         // Liquid Bronze
        symlog(status(p, PlayerStatus::PLATED_ARMOR)),   // Essence of Steel
        symlog(status(p, PlayerStatus::NEXT_TURN_BLOCK)),  // Self-Forming Clay
        held / 5.f,
        std::max(0, state.potionCapacity - held) / 5.f,  // Entropic Brew fills these
        p.cardDrawPerTurn / 10.f,
    };
    global.max_hp = static_cast<float>(p.maxHp);
}

// ---- monsters: statuses of act-1 monsters (and what Ironclad cards / potions put on them) ----
// EXTEND HERE (and monster_status_features): later acts' powers (flight, malleable, intangible, invincible,
// reactive, thorns, regen, poison for Silent, ...).
void monster_extras(const sts::Monster& m, MonsterToken& token) {
    using MS = sts::MonsterStatus;
    const auto s = [&](MS x) { return symlog(static_cast<float>(m.getStatusInternal(x))); };
    token.previous_move_id = static_cast<int>(m.moveHistory[1]);
    token.status = {
        s(MS::ARTIFACT),     // Sentries
        s(MS::METALLICIZE),  // Lagavulin
        s(MS::SHACKLED),     // Dark Shackles: strength returned at end of turn
        s(MS::ANGRY),        // Mad Gremlin
        s(MS::CURL_UP),      // louses
        s(MS::ENRAGE),       // Gremlin Nob: strength per skill played
        s(MS::MODE_SHIFT),   // The Guardian
        s(MS::RITUAL),       // Cultist
        s(MS::SPORE_CLOUD),  // Fungi Beast
        s(MS::THIEVERY),     // Looter
        s(MS::SHARP_HIDE),   // The Guardian (defensive mode)
        float(m.hasStatusInternal(MS::ASLEEP)),  // Lagavulin
        float(m.doesEscapeNext()),               // Looter
        symlog(static_cast<float>(m.miscInfo)),     // per-monster counter (meaning by monster id): wizard charge,
                                                    // louse bite, hexaghost divider, stolen gold, ...
        symlog(static_cast<float>(m.uniquePower0)),  // per-monster: hexaghost orb count (else the unique power)
    };
}

// ---- potions ----
// numeric: 0 damage to one enemy /50, 1 damage to all /50, 2 block /50, 3 strength /10, 4 dexterity /10,
// 5 strength / dexterity lost at end of turn (flag), 6 energy /5, 7 cards drawn /10, 8 healing /50,
// 9 max HP /10, 10 enemy vulnerable /5, 11 enemy weak /5, 12 lasting per-turn buff /10 (ritual, metallicize,
// plated armor, thorns), 13 artifact /2, 14 cards gained or replayed /3, 15 revive heal /50 (Fairy),
// 16 random effect (flag), 17 needs a target (flag).
// Amounts follow the simulator's drinkPotion (Sacred Bark included), so they are what drinking now does.
// EXTEND HERE: other characters' potions (poison, shivs, focus, orbs, stances, miracles, ...) throw.
PotionToken encode_potion(const sts::BattleContext& state, sts::Potion potion) {
    using enum sts::Potion;
    const bool bark = state.player.hasRelic<sts::RelicId::SACRED_BARK>();
    const float k = bark ? 2.f : 1.f;
    const float max_hp = static_cast<float>(state.player.maxHp);
    PotionToken token{static_cast<int>(potion), {}};
    auto& x = token.numeric;
    switch (potion) {
    case FIRE_POTION: x[0] = 20 * k / 50; break;
    case EXPLOSIVE_POTION: x[1] = 10 * k / 50; break;
    case BLOCK_POTION: x[2] = 12 * k / 50; break;
    case STRENGTH_POTION: x[3] = 2 * k / 10; break;
    case FLEX_POTION: x[3] = 5 * k / 10; x[5] = 1; break;
    case DEXTERITY_POTION: x[4] = 2 * k / 10; break;
    case SPEED_POTION: x[4] = 5 * k / 10; x[5] = 1; break;
    case ENERGY_POTION: x[6] = 2 * k / 5; break;
    case SWIFT_POTION: x[7] = 3 * k / 10; break;
    case SNECKO_OIL: x[7] = 5 * k / 10; x[16] = 1; break;
    case BLOOD_POTION: x[8] = std::floor(max_hp * (bark ? 40 : 20) / 100.f) / 50; break;
    case REGEN_POTION: { const float r = 5 * k; x[8] = r * (r + 1) / 2 / 50; x[12] = r / 10; break; }
    case FRUIT_JUICE: x[8] = 5 * k / 50; x[9] = 5 * k / 10; break;
    case FEAR_POTION: x[10] = 3 * k / 5; break;
    case WEAK_POTION: x[11] = 3 * k / 5; break;
    case CULTIST_POTION: x[12] = 1 * k / 10; break;
    case HEART_OF_IRON: x[12] = 6 * k / 10; break;
    case ESSENCE_OF_STEEL: x[12] = 4 * k / 10; break;
    case LIQUID_BRONZE: x[12] = 3 * k / 10; break;
    case ANCIENT_POTION: x[13] = 1 * k / 2; break;
    case ATTACK_POTION: case SKILL_POTION: case POWER_POTION: case COLORLESS_POTION:
        x[14] = 1 * k / 3; x[16] = 1; break;  // discover one of three random cards (bark: two copies)
    case LIQUID_MEMORIES: x[14] = 1 * k / 3; break;
    case DUPLICATION_POTION: x[14] = 1 * k / 3; break;
    case DISTILLED_CHAOS: x[14] = 3 * k / 3; x[16] = 1; break;
    case BLESSING_OF_THE_FORGE: case ELIXIR_POTION: break;  // (upgrade / exhaust the hand: the id carries it)
    case ENTROPIC_BREW: case GAMBLERS_BREW: x[16] = 1; break;
    case FAIRY_POTION: x[15] = std::max(1.f, std::floor(max_hp * (bark ? 0.6f : 0.3f))) / 50; break;
    // Smoke Bomb: disabled in the simulator (never generated, never legal).
    default:
        throw std::runtime_error{"unsupported potion in combat features: " + std::string{sts::getPotionName(potion)}};
    }
    x[17] = float(sts::potionRequiresTarget(potion));
    return token;
}

std::vector<PotionToken> encode_potions(const sts::BattleContext& state, std::vector<int>* slots) {
    std::vector<std::pair<PotionToken, int>> potions;
    for (int i = 0; i < state.potionCapacity; ++i)
        if (state.potions[i] != sts::Potion::EMPTY_POTION_SLOT && state.potions[i] != sts::Potion::INVALID)
            potions.emplace_back(encode_potion(state, state.potions[i]), i);
    std::ranges::sort(potions, {}, [](const auto& p) { return std::tie(p.first.potion_id, p.first.numeric); });
    std::vector<PotionToken> tokens;
    if (slots) slots->clear();
    for (const auto& [token, slot] : potions) {
        tokens.push_back(token);
        if (slots) slots->push_back(slot);
    }
    return tokens;
}

// ---- relics ----
// numeric: 0 progress to the relic's next trigger (0-1), 1 the next qualifying event triggers it (flag),
// 2 armed / active now (flag). Relics without a counter or condition are all zeros: their id says it all.
// Every relic in the combat state's relic bits gets a token (RelicId < 128: the in-combat relics).
// EXTEND HERE: counters of other characters' relics (inserter, ...) and conditions of later-act relics.
RelicToken encode_relic(const sts::BattleContext& state, sts::RelicId relic) {
    using enum sts::RelicId;
    const auto& p = state.player;
    RelicToken token{static_cast<int>(relic) + 1, {}};
    auto& x = token.numeric;
    const auto counter = [&](int count, int period) {  // fires when count reaches period
        x[0] = static_cast<float>(count) / period;
        x[1] = float(count == period - 1);
    };
    const auto on_turn = [&](int turn) {  // fires on bc.turn == turn
        x[0] = std::min(1.f, static_cast<float>(state.turn) / turn);
        x[1] = float(state.turn == turn - 1);
        x[2] = float(state.turn < turn);
    };
    switch (relic) {
    case PEN_NIB:  // counter 0-8 attacks; -1: the next attack deals double (PEN_NIB status)
        if (p.penNibCounter < 0) { x[0] = 1; x[2] = 1; }
        else counter(p.penNibCounter, 9);
        break;
    case NUNCHAKU: counter(p.nunchakuCounter, 10); break;
    case INK_BOTTLE: counter(p.inkBottleCounter, 10); break;
    case INCENSE_BURNER: counter(p.incenseBurnerCounter, 6); break;
    case HAPPY_FLOWER: counter(p.happyFlowerCounter, 3); break;
    case SUNDIAL: counter(p.sundialCounter, 3); break;
    case KUNAI: case SHURIKEN: case ORNAMENTAL_FAN: counter(p.attacksPlayedThisTurn % 3, 3); break;
    case LETTER_OPENER: counter(p.skillsPlayedThisTurn % 3, 3); break;
    case ORANGE_PELLETS: counter(static_cast<int>(p.orangePelletsCardTypesPlayed.count()), 3); break;
    case VELVET_CHOKER:
        x[0] = std::min(1.f, p.cardsPlayedThisTurn / 6.f);
        x[1] = float(p.cardsPlayedThisTurn == 5);
        x[2] = float(p.cardsPlayedThisTurn >= 6);  // no more cards this turn
        break;
    case ART_OF_WAR: x[2] = float(p.attacksPlayedThisTurn == 0); break;
    case POCKETWATCH: x[2] = float(p.cardsPlayedThisTurn <= 3); break;
    case NECRONOMICON: x[2] = float(!p.haveUsedNecronomiconThisTurn); break;
    case RED_SKULL: x[2] = float(p.curHp <= p.maxHp / 2); break;
    case HORN_CLEAT: on_turn(1); break;
    case CAPTAINS_WHEEL: on_turn(2); break;
    case STONE_CALENDAR: on_turn(7); break;  // end of turn 6 (0-based) = start of turn 7
    case CENTENNIAL_PUZZLE: case LIZARD_TAIL: x[2] = 1; break;  // unused (the bit is cleared when used)
    default: break;
    }
    return token;
}

std::vector<RelicToken> encode_relics(const sts::BattleContext& state) {
    std::vector<RelicToken> tokens;
    for (int r = 0; r < 128; ++r)
        if (state.player.hasRelicRuntime(static_cast<sts::RelicId>(r)))
            tokens.push_back(encode_relic(state, static_cast<sts::RelicId>(r)));
    return tokens;  // in relic id order
}


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
    // Event cards found in Ironclad runs.
    if ((card.id == sts::CardId::APPARITION || card.id == sts::CardId::BITE ||
         card.id == sts::CardId::JAX || card.id == sts::CardId::RITUAL_DAGGER)) return true;
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
    MonsterToken token{static_cast<int>(monster.id), static_cast<int>(monster.moveHistory[0]),
        {monster.curHp / 100.f, monster.maxHp ? monster.curHp / float(monster.maxHp) : 0.f, monster.block / 100.f,
         per_hit * damage.attackCount / 100.f, damage.attackCount / 10.f, monster.strength / 10.f,
         monster.weak / 10.f, monster.vulnerable / 10.f, float(monster.isTargetable())}};
    monster_extras(monster, token);
    return token;
}

bool card_less(const CardToken& a, const CardToken& b) { return std::tie(a.card_id, a.zone, a.card_type, a.target_type, a.numeric) < std::tie(b.card_id, b.zone, b.card_type, b.target_type, b.numeric); }
bool monster_less(const MonsterToken& a, const MonsterToken& b) { return std::tie(a.monster_id, a.move_id, a.numeric, a.previous_move_id, a.status) < std::tie(b.monster_id, b.move_id, b.numeric, b.previous_move_id, b.status); }

CardZone zone_of(combat::Pile pile) {
    switch (pile) {
    case combat::Pile::hand: return CardZone::hand;
    case combat::Pile::draw: return CardZone::draw;
    case combat::Pile::discard: return CardZone::discard;
    case combat::Pile::exhaust: return CardZone::exhaust;
    case combat::Pile::offered: return CardZone::offered;
    }
    throw std::logic_error{"features: unknown pile"};
}

// The card x monster interaction numeric of `card` (in hand) played at monster slot `slot`; nullopt when it
// deals no targeted damage or the monster is not targetable (as encode_state's interaction tokens).
std::optional<std::array<float, 6>> interaction_numeric(const sts::BattleContext& state, const sts::CardInstance& card,
                                                        int slot) {
    const auto meta = card_meta(state, card);
    if (meta.damage == 0 || meta.target == TargetType::random_enemy) return std::nullopt;
    const auto& monster = state.monsters.arr[slot];
    if (!monster.isTargetable()) return std::nullopt;
    const int per_hit = state.calculateCardDamage(card, slot, meta.damage);
    const int damage = per_hit * meta.hits;
    const int hp_damage = std::min(monster.curHp, std::max(0, damage - monster.block));
    const int remaining = monster.curHp - hp_damage;
    return std::array<float, 6>{meta.hits / 10.f, damage / 100.f, hp_damage / 100.f, remaining / 100.f,
                                monster.maxHp ? remaining / float(monster.maxHp) : 0.f,
                                float(card.canUse(state, slot, false))};
}

}  // namespace

std::vector<CardToken> encode_cards(const sts::BattleContext& state, std::vector<CardLocation>* locations) {
    struct Entry { CardToken token; CardLocation location; };
    std::vector<Entry> cards;
    cards.reserve(state.cards.cardsInHand + state.cards.drawPile.size() + state.cards.discardPile.size()
                  + state.cards.exhaustPile.size());
    const auto append = [&](const auto& pile, int count, CardZone zone) {
        for (int i = 0; i < count; ++i)
            cards.push_back({encode_card(state, pile[i], zone,
                zone == CardZone::hand && pile[i].canUseOnAnyTarget(state)), {zone, i}});
    };
    append(state.cards.hand, state.cards.cardsInHand, CardZone::hand);
    append(state.cards.drawPile, int(state.cards.drawPile.size()), CardZone::draw);
    append(state.cards.discardPile, int(state.cards.discardPile.size()), CardZone::discard);
    append(state.cards.exhaustPile, int(state.cards.exhaustPile.size()), CardZone::exhaust);
    std::sort(cards.begin(), cards.end(), [](const auto& a, const auto& b) { return card_less(a.token, b.token); });
    std::vector<CardToken> tokens;
    tokens.reserve(cards.size());
    if (locations) locations->clear();
    for (const auto& card : cards) {
        tokens.push_back(card.token);
        if (locations) locations->push_back(card.location);
    }
    return tokens;
}

std::vector<MonsterToken> encode_monsters(const sts::BattleContext& state, std::vector<int>* slots) {
    struct Entry { MonsterToken token; int slot; };
    std::vector<Entry> monsters;
    for (int slot = 0; slot < int(state.monsters.arr.size()); ++slot) {
        const auto& monster = state.monsters.arr[slot];
        if (monster.id != sts::MonsterId::INVALID && monster.isAlive())
            monsters.push_back({encode_monster(state, monster), slot});
    }
    std::sort(monsters.begin(), monsters.end(), [](const auto& a, const auto& b) { return monster_less(a.token, b.token); });
    std::vector<MonsterToken> tokens;
    if (slots) slots->clear();
    for (const auto& monster : monsters) {
        tokens.push_back(monster.token);
        if (slots) slots->push_back(monster.slot);
    }
    return tokens;
}

int incoming_damage(const sts::BattleContext& state) {
    int damage = 0;
    for (const auto& monster : state.monsters.arr) if (monster.id != sts::MonsterId::INVALID && monster.isAlive()) {
        const auto move = monster.getMoveBaseDamage(state);
        damage += monster.calculateDamageToPlayer(state, move.damage) * move.attackCount;
    }
    return damage;
}

EncodedCombatState encode_state(const sts::BattleContext& state, std::vector<int>* monster_slots) {
    EncodedCombatState encoding;
    const auto& player = state.player;
    const int incoming = incoming_damage(state);
    const int select_task = state.inputState == sts::InputState::CARD_SELECT
        ? static_cast<int>(state.cardSelectInfo.cardSelectTask)
        : static_cast<int>(sts::CardSelectTask::INVALID);
    encoding.global = {{state.turn / 20.f, player.curHp / 100.f,
        player.maxHp ? player.curHp / float(player.maxHp) : 0.f, player.block / 100.f, player.energy / 10.f,
        player.energyPerTurn / 10.f, player.strength / 10.f, player.dexterity / 10.f,
        player.getStatusRuntime(PlayerStatus::WEAK) / 10.f, player.getStatusRuntime(PlayerStatus::VULNERABLE) / 10.f,
        player.getStatusRuntime(PlayerStatus::FRAIL) / 10.f, player.cardsPlayedThisTurn / 20.f,
        state.cards.cardsInHand / 10.f, state.cards.drawPile.size() / 64.f,
        state.cards.discardPile.size() / 64.f, state.cards.exhaustPile.size() / 64.f,
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
        static_cast<int>(state.inputState), select_task};
    player_extras(state, encoding.global);
    encoding.potions = encode_potions(state);
    encoding.relics = encode_relics(state);
    encoding.cards = encode_cards(state);
    encoding.monsters = encode_monsters(state, monster_slots);
    return encoding;
}

ActionToken encode_action(const sts::BattleContext& state, const sts::search::Action action) {
    const int select_task = state.inputState == sts::InputState::CARD_SELECT
        ? static_cast<int>(state.cardSelectInfo.cardSelectTask)
        : static_cast<int>(sts::CardSelectTask::INVALID);
    ActionToken token{.kind = static_cast<EncodedActionKind>(action.getActionType()),
                      .source_card = std::nullopt, .target_monster = std::nullopt, .potion_id = std::nullopt,
                      .card_selection_task = select_task,
                      .skips_selection = state.inputState == sts::InputState::CARD_SELECT &&
                          action.getActionType() == sts::search::ActionType::SINGLE_CARD_SELECT &&
                          state.cardSelectInfo.cardSelectTask == sts::CardSelectTask::CODEX && action.getSelectIdx() == 3,
                      .potion = std::nullopt, .interaction = std::nullopt,
                      .discards_potion = action.getActionType() == sts::search::ActionType::POTION
                                         && action.getTargetIdx() > 5};  // Action::execute: target > 5 discards
    if (action.getActionType() == sts::search::ActionType::CARD) {
        const auto source = action.getSourceIdx();
        if (source >= 0 && source < state.cards.cardsInHand)
            token.source_card = encode_card(state, state.cards.hand[source], CardZone::hand, true);
    } else if (action.getActionType() == sts::search::ActionType::POTION) {
        const auto source = action.getSourceIdx();
        if (source >= 0 && source < static_cast<int>(state.potions.size()))
            token.potion_id = static_cast<int>(state.potions[source]);
    } else if (action.getActionType() == sts::search::ActionType::SINGLE_CARD_SELECT) {
        const auto task = state.inputState == sts::InputState::CARD_SELECT
            ? state.cardSelectInfo.cardSelectTask
            : sts::CardSelectTask::INVALID;
        const auto selected = action.getSelectIdx();
        if ((task == sts::CardSelectTask::CODEX || task == sts::CardSelectTask::DISCOVERY) && selected >= 0 && selected < 3) {
            auto info = state.cardSelectInfo;  // (its accessors are not const)
            const auto id = task == sts::CardSelectTask::CODEX ? info.codexCards()[selected]
                                                                : info.discovery_Cards()[selected];
            token.source_card = CardToken{.card_id = static_cast<int>(id), .zone = CardZone::offered};
        } else {
            const auto pile = combat::selection_pile(state);
            const auto cards = combat::cards_in(state, pile);
            if (selected >= 0 && selected < static_cast<int>(cards.size()))
                token.source_card = encode_card(state, cards[selected], zone_of(pile), false);
        }
    }
    const bool requires_target = action.getActionType() == sts::search::ActionType::CARD
        ? state.cards.hand[action.getSourceIdx()].requiresTarget()
        : action.getActionType() == sts::search::ActionType::POTION
            && potionRequiresTarget(state.potions[action.getSourceIdx()]);
    if (requires_target && action.getTargetIdx() >= 0 && action.getTargetIdx() < static_cast<int>(state.monsters.arr.size()))
        token.target_monster = encode_monster(state, state.monsters.arr[action.getTargetIdx()]);
    if (action.getActionType() == sts::search::ActionType::POTION) {
        const auto source = action.getSourceIdx();
        if (source >= 0 && source < static_cast<int>(state.potions.size()))
            token.potion = encode_potion(state, state.potions[source]);
    }
    if (action.getActionType() == sts::search::ActionType::CARD && token.target_monster)
        token.interaction = interaction_numeric(state, state.cards.hand[action.getSourceIdx()], action.getTargetIdx());
    return token;
}

}  // namespace stsrl::nn
