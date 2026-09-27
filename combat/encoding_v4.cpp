// Encoding v4 additions (combat/encoding.hpp): what the v3 encoding left out.
//
// Coverage: Ironclad, act 1 (all act-1 monsters, Ironclad + colorless cards, potions Ironclad can hold, any
// relic in the combat state). Each table below says where to extend it for other characters / acts; a width
// change (encoding.hpp) is a new encoding version and a new value net kind.
//
// Scaling: counts that can grow without bound use symlog (sign * log1p|x|), flags are 0 / 1, and potion
// magnitudes are divided by a fixed typical size. Layout changes here must be mirrored in
// python/sts_combat_rl/topology/deep_sets_v3.py (the widths) and never made to a trained kind.
#include "combat/encoding_v4.hpp"

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

namespace stsrl::v4 {
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
// EXTEND HERE (and player_v4_features): other characters' powers (focus, mantra, orbs, stances, ...) and
// later acts' debuffs (hex, constricted, draw reduction, fasting, ...).
void encode_player(const sts::BattleContext& state, GlobalFeatures& global) {
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
void encode_monster(const sts::Monster& m, MonsterToken& token) {
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
        throw std::runtime_error{"unsupported potion in encoding v4: " + std::string{sts::getPotionName(potion)}};
    }
    x[17] = float(sts::potionRequiresTarget(potion));
    return token;
}

std::vector<PotionToken> encode_potions(const sts::BattleContext& state) {
    std::vector<PotionToken> tokens;
    for (int i = 0; i < state.potionCapacity; ++i)
        if (state.potions[i] != sts::Potion::EMPTY_POTION_SLOT && state.potions[i] != sts::Potion::INVALID)
            tokens.push_back(encode_potion(state, state.potions[i]));
    std::ranges::sort(tokens, {}, [](const PotionToken& t) { return std::tie(t.potion_id, t.numeric); });
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

}  // namespace stsrl::v4
