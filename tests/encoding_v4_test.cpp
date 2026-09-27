// Encoding v4 (combat/encoding_v4.cpp) and deep_sets_v3 inference.
//   encoding_v4_test                        checks that the v4 fields see what the v3 encoding missed
//   encoding_v4_test WEIGHTS OUT.json       writes [{state: state_row (a combat_v3 row's state columns with
//                                           legal_actions), value, logits (policy nets)}] for the sample
//                                           decisions (the Python / C++ parity test, tests/test_deep_sets_v3.py)
#include "combat/encoding_v4.hpp"
#include "combat/environment.hpp"
#include "topology/value_net.hpp"

#include "combat/BattleContext.h"
#include "constants/CharacterClasses.h"
#include "constants/MonsterEncounters.h"
#include "constants/MonsterStatusEffects.h"
#include "constants/Potions.h"
#include "constants/Relics.h"
#include "constants/Rooms.h"
#include "game/GameContext.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

using stsrl::encode_state;
using MS = sts::MonsterStatus;

void check(bool ok, const char* what) {
    if (!ok) {
        std::cerr << "FAILED: " << what << '\n';
        std::exit(1);
    }
}

float symlog(float x) { return std::copysign(std::log1p(std::abs(x)), x); }

sts::BattleContext fight(sts::MonsterEncounter encounter, std::uint64_t seed = 11) {
    sts::GameContext game{sts::CharacterClass::IRONCLAD, seed, 20};
    game.floorNum = 6;
    game.curRoom = sts::Room::ELITE;
    sts::BattleContext battle;
    battle.init(game, encounter);
    return battle;
}

// Sample states for the parity test: every v4 field non-trivial somewhere.
std::vector<sts::BattleContext> samples() {
    std::vector<sts::BattleContext> out;
    auto sentries = fight(sts::MonsterEncounter::THREE_SENTRIES);
    sentries.obtainPotion(sts::Potion::FIRE_POTION);
    sentries.obtainPotion(sts::Potion::FAIRY_POTION);
    sentries.player.setHasRelic<sts::RelicId::PEN_NIB>(true);
    sentries.player.penNibCounter = 8;
    sentries.player.setHasRelic<sts::RelicId::KUNAI>(true);
    sentries.player.buff<PlayerStatus::REGEN>(3);
    out.push_back(sentries);
    auto stripped = sentries;
    stripped.debuffEnemy<MS::VULNERABLE>(1, 2, false);  // Bash into artifact: artifact gone, no vulnerable
    out.push_back(stripped);
    auto nob = fight(sts::MonsterEncounter::GREMLIN_NOB, 12);
    nob.monsters.arr[0].buff<MS::ENRAGE>(3);
    nob.obtainPotion(sts::Potion::BLOOD_POTION);
    nob.obtainPotion(sts::Potion::ATTACK_POTION);
    nob.obtainPotion(sts::Potion::FLEX_POTION);
    nob.player.setHasRelic<sts::RelicId::SACRED_BARK>(true);
    out.push_back(nob);
    out.push_back(fight(sts::MonsterEncounter::LAGAVULIN, 13));
    out.push_back(fight(sts::MonsterEncounter::JAW_WORM, 14));
    return out;
}

void encoding_checks() {
    // Sentries: artifact is visible, and stripping it changes the encoding.
    const auto s = samples();
    const auto sentries = encode_state(s[0]), stripped = encode_state(s[1]);
    check(sentries.monsters.size() == 3, "three sentries");
    for (const auto& m : sentries.monsters) check(m.status[0] == symlog(1), "sentry artifact 1");
    int artifact = 0;
    for (const auto& m : stripped.monsters) artifact += m.status[0] > 0;
    check(artifact == 2, "one sentry's artifact stripped");
    check(sentries.monsters != stripped.monsters, "stripping artifact changes the encoding");
    for (const auto& m : stripped.monsters) check(m.numeric[7] == 0, "artifact blocked the vulnerable");

    // Potions: one token per held potion; count in the player numeric.
    check(sentries.potions.size() == 2, "two potion tokens");
    check(sentries.global.player[9] == 2 / 5.f, "potions held");
    check(sentries.global.player[10] == (s[0].potionCapacity - 2) / 5.f, "empty slots");
    const auto fire = stsrl::v4::encode_potion(s[0], sts::Potion::FIRE_POTION);
    check(fire.numeric[0] == 20 / 50.f && fire.numeric[17] == 1, "fire potion damage and target");
    const auto fairy = stsrl::v4::encode_potion(s[0], sts::Potion::FAIRY_POTION);
    check(fairy.numeric[15] == std::floor(s[0].player.maxHp * 0.3f) / 50, "fairy revive heal");
    check(stsrl::v4::encode_potion(s[2], sts::Potion::FLEX_POTION).numeric[3] == 1.f, "sacred bark doubles flex");
    check(sentries.global.player[4] == symlog(3), "regen status");
    check(sentries.global.max_hp == static_cast<float>(s[0].player.maxHp), "raw max hp");

    // Every potion Ironclad can hold encodes; other characters' potions throw (the Act-1 Ironclad boundary).
    using enum sts::Potion;
    for (const auto p : {ANCIENT_POTION, ATTACK_POTION, BLESSING_OF_THE_FORGE, BLOCK_POTION, BLOOD_POTION,
                         COLORLESS_POTION, CULTIST_POTION, DEXTERITY_POTION, DISTILLED_CHAOS, DUPLICATION_POTION,
                         ELIXIR_POTION, ENERGY_POTION, ENTROPIC_BREW, ESSENCE_OF_STEEL, EXPLOSIVE_POTION,
                         FAIRY_POTION, FEAR_POTION, FIRE_POTION, FLEX_POTION, FRUIT_JUICE, GAMBLERS_BREW,
                         HEART_OF_IRON, LIQUID_BRONZE, LIQUID_MEMORIES, POWER_POTION, REGEN_POTION, SKILL_POTION,
                         SNECKO_OIL, SPEED_POTION, STRENGTH_POTION, SWIFT_POTION, WEAK_POTION})
        (void)stsrl::v4::encode_potion(s[0], p);
    bool threw = false;
    try { (void)stsrl::v4::encode_potion(s[0], POISON_POTION); } catch (const std::exception&) { threw = true; }
    check(threw, "Silent's poison potion is outside the encoding");

    // Relics: Pen Nib at 8 attacks fires on the next attack; Kunai at 0 of 3; Burning Blood (starter) present.
    const auto relic = [&](const stsrl::EncodedCombatState& e, sts::RelicId r) {
        for (const auto& t : e.relics) if (t.relic_id == static_cast<int>(r) + 1) return t;
        check(false, "relic token present");
        return stsrl::RelicToken{};
    };
    check(relic(sentries, sts::RelicId::PEN_NIB).numeric[1] == 1, "pen nib fires next attack");
    check(relic(sentries, sts::RelicId::KUNAI).numeric[0] == 0, "kunai progress 0");

    // Nob's enrage, Lagavulin asleep with metallicize.
    const auto nob = encode_state(s[2]);
    check(nob.monsters[0].status[5] == symlog(3), "nob enrage");
    const auto laga = encode_state(s[3]);
    check(laga.monsters[0].status[11] == 1 && laga.monsters[0].status[1] == symlog(8), "lagavulin asleep, metallicize");

    // Previous move.
    auto moved = s[4];
    moved.monsters.arr[0].moveHistory[1] = moved.monsters.arr[0].moveHistory[0];
    check(encode_state(moved).monsters[0].previous_move_id == static_cast<int>(moved.monsters.arr[0].moveHistory[0]),
          "previous move");

    // Action tokens: potions carry their token, targeted attacks their card x target interaction.
    {
        stsrl::CombatEnvironment env{s[0]};
        const auto d = env.decision();
        int potions = 0, interactions = 0;
        for (const auto& a : d.encoding.legal_actions) {
            if (a.kind == stsrl::EncodedActionKind::potion) {
                check(a.potion.has_value() && a.potion_id && a.potion->potion_id == *a.potion_id, "potion action token");
                potions += a.potion->potion_id == static_cast<int>(sts::Potion::FIRE_POTION) && !a.discards_potion;
                if (a.potion->potion_id == static_cast<int>(sts::Potion::FAIRY_POTION))
                    check(a.discards_potion, "fairy's only action is a discard");
            }
            if (a.interaction) {
                check(a.source_card && a.target_monster, "interaction has card and target");
                ++interactions;
            }
        }
        check(potions == 3, "fire potion at each of three sentries");
        check(interactions > 0, "targeted attacks have interactions");
        const auto again = stsrl::encode_action(s[0], env.action_bits(d.encoding.legal_actions[0].execution_index),
                                                d.encoding.legal_actions[0].execution_index);
        check(again == d.encoding.legal_actions[0], "encode_action matches the decision's token");
    }

    // Drinking a potion removes its token.
    stsrl::CombatEnvironment env{s[0]};
    const auto decision = env.decision();
    for (const auto& a : decision.legal_actions)
        if (a.description.find("Fire Potion") != std::string::npos) {
            env.step(a.index);
            check(env.decision().encoding.potions.size() == 1, "drunk potion's token gone");
            return;
        }
    check(false, "fire potion is drinkable");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 1) {
        encoding_checks();
        return 0;
    }
    if (argc != 3) {
        std::cerr << "usage: encoding_v4_test [WEIGHTS OUT.json]\n";
        return 2;
    }
    const stsrl::ValueNet net{argv[1]};
    auto out = nlohmann::json::array();
    for (const auto& battle : samples()) {
        stsrl::CombatEnvironment env{battle};
        const auto e = env.decision().encoding;
        nlohmann::json row = {{"state", stsrl::state_row(e, true)}};
        if (net.has_policy()) {
            std::vector<float> logits;
            row["value"] = net.evaluate(e, e.legal_actions, logits);
            row["logits"] = logits;  // in e.legal_actions order
            check(row["value"].get<float>() == net.evaluate(e), "value is the same with the policy");
        } else {
            row["value"] = net.evaluate(e);
        }
        out.push_back(std::move(row));
    }
    std::ofstream{argv[2]} << out.dump();
    return 0;
}
