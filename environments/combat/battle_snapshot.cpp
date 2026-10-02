#include "environments/combat/battle_snapshot.hpp"
#include <stdexcept>
#include <type_traits>
namespace stsrl {
namespace {
using Json = nlohmann::json;
#define P(field) j[#field] = v.field
#define G(field) v.field = j.at(#field).get<std::remove_cvref_t<decltype(v.field)>>()
Json raw_battle(const sts::BattleContext& v) { Json j; P(haveUsedDiscoveryAction); P(undefinedBehaviorEvoked); P(seed); P(floorNum); P(encounter); P(loopCount); P(energyWasted); P(cardsDrawn); P(ascension); P(outcome); P(unsupportedEffectKind); P(unsupportedEffectId); P(inputState); P(monsterTurnIdx); P(isBattleOver); P(escapedCombat); P(endTurnQueued); P(endTurnAfterCurrentCard); P(turnHasEnded); P(skipMonsterTurn); P(potionCount); P(potionCapacity); P(turn); return j; }
void restore_battle(sts::BattleContext& v, const Json& j) { G(haveUsedDiscoveryAction); G(undefinedBehaviorEvoked); G(seed); G(floorNum); G(encounter); G(loopCount); G(energyWasted); G(cardsDrawn); G(ascension); G(outcome); G(unsupportedEffectKind); G(unsupportedEffectId); G(inputState); G(monsterTurnIdx); G(isBattleOver); G(escapedCombat); G(endTurnQueued); G(endTurnAfterCurrentCard); G(turnHasEnded); G(skipMonsterTurn); G(potionCount); G(potionCapacity); G(turn); }
Json raw_player(const sts::Player& v) { Json j; P(cc); P(gold); P(curHp); P(maxHp); P(energy); P(energyPerTurn); P(cardDrawPerTurn); P(stance); P(orbSlots); P(lightningOrbsChanneled); P(frostOrbsChanneled); P(lastTargetedMonster); P(block); P(artifact); P(dexterity); P(focus); P(strength); P(justAppliedBits); P(statusBits0); P(statusBits1); P(relicBits0); P(relicBits1); P(happyFlowerCounter); P(incenseBurnerCounter); P(inkBottleCounter); P(inserterCounter); P(nunchakuCounter); P(penNibCounter); P(sundialCounter); P(haveUsedNecronomiconThisTurn); P(combustHpLoss); P(devaFormEnergyPerTurn); P(echoFormCardsDoubled); P(panacheCounter); P(cardsPlayedThisTurn); P(attacksPlayedThisTurn); P(skillsPlayedThisTurn); P(cardsDiscardedThisTurn); P(lastAttackUnblockedDamage); P(timesDamagedThisCombat); P(bomb1); P(bomb2); P(bomb3); P(selfRepairHeal); P(creativeAi); P(helloWorld); P(heatsinks); P(nirvanaBlock); P(rushdownDraw); P(studyInsights); P(mentalFortressBlock); P(machineLearning); P(storm); return j; }
void restore_player(sts::Player& v, const Json& j) { G(cc); G(gold); G(curHp); G(maxHp); G(energy); G(energyPerTurn); G(cardDrawPerTurn); G(stance); G(orbSlots); G(lightningOrbsChanneled); G(frostOrbsChanneled); G(lastTargetedMonster); G(block); G(artifact); G(dexterity); G(focus); G(strength); G(justAppliedBits); G(statusBits0); G(statusBits1); G(relicBits0); G(relicBits1); G(happyFlowerCounter); G(incenseBurnerCounter); G(inkBottleCounter); G(inserterCounter); G(nunchakuCounter); G(penNibCounter); G(sundialCounter); G(haveUsedNecronomiconThisTurn); G(combustHpLoss); G(devaFormEnergyPerTurn); G(echoFormCardsDoubled); G(panacheCounter); G(cardsPlayedThisTurn); G(attacksPlayedThisTurn); G(skillsPlayedThisTurn); G(cardsDiscardedThisTurn); G(lastAttackUnblockedDamage); G(timesDamagedThisCombat); G(bomb1); G(bomb2); G(bomb3); G(selfRepairHeal); G(creativeAi); G(helloWorld); G(heatsinks); G(nirvanaBlock); G(rushdownDraw); G(studyInsights); G(mentalFortressBlock); G(machineLearning); G(storm); }
Json raw_monster(const sts::Monster& v) { Json j; P(idx); P(id); P(curHp); P(maxHp); P(block); P(isEscapingB); P(halfDead); P(escapeNext); P(statusBits); P(artifact); P(blockReturn); P(choked); P(corpseExplosion); P(lockOn); P(mark); P(metallicize); P(platedArmor); P(poison); P(regen); P(shackled); P(strength); P(vulnerable); P(weak); P(uniquePower0); P(uniquePower1); P(miscInfo); return j; }
void restore_monster(sts::Monster& v, const Json& j) { G(idx); G(id); G(curHp); G(maxHp); G(block); G(isEscapingB); G(halfDead); G(escapeNext); G(statusBits); G(artifact); G(blockReturn); G(choked); G(corpseExplosion); G(lockOn); G(mark); G(metallicize); G(platedArmor); G(poison); G(regen); G(shackled); G(strength); G(vulnerable); G(weak); G(uniquePower0); G(uniquePower1); G(miscInfo); }
Json raw_counts(const sts::CardManager& v) { Json j; P(nextUniqueCardId); P(handNormalityCount); P(handPainCount); P(strikeCount); P(handBloodCardCount); P(drawPileBloodCardCount); P(discardPileBloodCardCount); return j; }
void restore_counts(sts::CardManager& v, const Json& j) { G(nextUniqueCardId); G(handNormalityCount); G(handPainCount); G(strikeCount); G(handBloodCardCount); G(drawPileBloodCardCount); G(discardPileBloodCardCount); }
#undef P
#undef G
Json card_json(const sts::CardInstance& c) {
    return {{"id",int(c.id)},{"unique_id",c.uniqueId},{"special_data",c.specialData},{"cost",c.cost},
        {"cost_for_turn",c.costForTurn},{"upgraded",c.upgraded},{"free_to_play_once",c.freeToPlayOnce},{"retain",c.retain}};
}
sts::CardInstance card_restore(const Json& j) {
    sts::CardInstance c{}; c.id=static_cast<sts::CardId>(j.at("id").get<int>());
    c.uniqueId=j.at("unique_id"); c.specialData=j.at("special_data"); c.cost=j.at("cost");
    c.costForTurn=j.at("cost_for_turn"); c.upgraded=j.at("upgraded"); c.freeToPlayOnce=j.at("free_to_play_once"); c.retain=j.at("retain"); return c;
}
template<class Container> Json card_list(const Container& c) {Json a=Json::array();for(const auto& v:c)a.push_back(card_json(v));return a;}
Json rng_json(const sts::Random& r) { return {{"counter",r.counter},{"seed0",r.seed0},{"seed1",r.seed1}}; }
void rng_restore(sts::Random& r,const Json& j){r.counter=j.at("counter");r.seed0=j.at("seed0");r.seed1=j.at("seed1");}
}
Json battle_snapshot(const sts::BattleContext& b) {
    // std::function action queues cannot be persisted losslessly. Reject unsupported initial boundaries.
    // At terminal boundaries stale pending actions are irrelevant to replay; snapshot compares factual endpoint.
    if (b.outcome == sts::Outcome::UNDECIDED &&
        (b.inputState != sts::InputState::PLAYER_NORMAL || b.actionQueue.size || b.cardQueue.size))
        throw std::invalid_argument("snapshot requires ordinary player boundary with empty queues");
    Json j, raw;
    raw["battle"]=raw_battle(b); raw["player"]=raw_player(b.player); raw["counts"]=raw_counts(b.cards);
    raw["player"]["status_map"]=Json::array();
    for(const auto& [s,v]:b.player.statusMap)raw["player"]["status_map"].push_back({int(s),v});
    raw["player"]["orange_pellets"]=b.player.orangePelletsCardTypesPlayed.to_ulong();
    raw["monsters"]=Json::array();for(const auto& m:b.monsters.arr)raw["monsters"].push_back(raw_monster(m));
    j["extra_state_json"]=raw.dump();
    j["seed"]=b.seed;
    j["floor_num"]=b.floorNum;
    j["encounter"]=b.encounter;
    j["ascension"]=b.ascension;
    j["turn"]=b.turn;
    j["loop_count"]=b.loopCount;
    j["energy_wasted"]=b.energyWasted;
    j["cards_drawn"]=b.cardsDrawn;
    j["have_used_discovery_action"]=b.haveUsedDiscoveryAction;
    j["undefined_behavior_evoked"]=b.undefinedBehaviorEvoked;
    j["unsupported_effect_kind"]=b.unsupportedEffectKind;
    j["unsupported_effect_id"]=b.unsupportedEffectId;
    j["monster_turn_idx"]=b.monsterTurnIdx;
    j["is_battle_over"]=b.isBattleOver;
    j["escaped_combat"]=b.escapedCombat;
    j["end_turn_queued"]=b.endTurnQueued;
    j["end_turn_after_current_card"]=b.endTurnAfterCurrentCard;
    j["turn_has_ended"]=b.turnHasEnded;
    j["skip_monster_turn"]=b.skipMonsterTurn;
    j["misc_bits"]=b.miscBits.to_ulong();
    j["rng"]["ai"]=rng_json(b.aiRng);
    j["rng"]["card_random"]=rng_json(b.cardRandomRng);
    j["rng"]["misc"]=rng_json(b.miscRng);
    j["rng"]["monster_hp"]=rng_json(b.monsterHpRng);
    j["rng"]["potion"]=rng_json(b.potionRng);
    j["rng"]["shuffle"]=rng_json(b.shuffleRng);
    j["player"]["character"]=b.player.cc;
    j["player"]["gold"]=b.player.gold;
    j["player"]["hp"]=b.player.curHp;
    j["player"]["max_hp"]=b.player.maxHp;
    j["player"]["energy"]=b.player.energy;
    j["player"]["block"]=b.player.block;
    j["player"]["energy_per_turn"]=b.player.energyPerTurn;
    j["player"]["card_draw_per_turn"]=b.player.cardDrawPerTurn;
    j["player"]["stance"]=b.player.stance;
    j["player"]["orb_slots"]=b.player.orbSlots;
    j["player"]["lightning_orbs_channeled"]=b.player.lightningOrbsChanneled;
    j["player"]["frost_orbs_channeled"]=b.player.frostOrbsChanneled;
    j["player"]["last_targeted_monster"]=b.player.lastTargetedMonster;
    j["player"]["artifact"]=b.player.artifact;
    j["player"]["dexterity"]=b.player.dexterity;
    j["player"]["focus"]=b.player.focus;
    j["player"]["strength"]=b.player.strength;
    j["player"]["happy_flower_counter"]=b.player.happyFlowerCounter;
    j["player"]["incense_burner_counter"]=b.player.incenseBurnerCounter;
    j["player"]["ink_bottle_counter"]=b.player.inkBottleCounter;
    j["player"]["inserter_counter"]=b.player.inserterCounter;
    j["player"]["nunchaku_counter"]=b.player.nunchakuCounter;
    j["player"]["pen_nib_counter"]=b.player.penNibCounter;
    j["player"]["sundial_counter"]=b.player.sundialCounter;
    j["player"]["used_necronomicon_this_turn"]=b.player.haveUsedNecronomiconThisTurn;
    j["player"]["combust_hp_loss"]=b.player.combustHpLoss;
    j["player"]["deva_form_energy_per_turn"]=b.player.devaFormEnergyPerTurn;
    j["player"]["echo_form_cards_doubled"]=b.player.echoFormCardsDoubled;
    j["player"]["panache_counter"]=b.player.panacheCounter;
    j["player"]["cards_played_this_turn"]=b.player.cardsPlayedThisTurn;
    j["player"]["attacks_played_this_turn"]=b.player.attacksPlayedThisTurn;
    j["player"]["skills_played_this_turn"]=b.player.skillsPlayedThisTurn;
    j["player"]["cards_discarded_this_turn"]=b.player.cardsDiscardedThisTurn;
    j["player"]["last_attack_unblocked_damage"]=b.player.lastAttackUnblockedDamage;
    j["player"]["times_damaged_this_combat"]=b.player.timesDamagedThisCombat;
    j["player"]["bomb1"]=b.player.bomb1;
    j["player"]["bomb2"]=b.player.bomb2;
    j["player"]["bomb3"]=b.player.bomb3;
    j["player"]["self_repair_heal"]=b.player.selfRepairHeal;
    j["player"]["creative_ai"]=b.player.creativeAi;
    j["player"]["hello_world"]=b.player.helloWorld;
    j["player"]["heatsinks"]=b.player.heatsinks;
    j["player"]["nirvana_block"]=b.player.nirvanaBlock;
    j["player"]["rushdown_draw"]=b.player.rushdownDraw;
    j["player"]["study_insights"]=b.player.studyInsights;
    j["player"]["mental_fortress_block"]=b.player.mentalFortressBlock;
    j["player"]["machine_learning"]=b.player.machineLearning;
    j["player"]["storm"]=b.player.storm;
    j["player"]["orange_pellets_card_types_played"]=b.player.orangePelletsCardTypesPlayed.to_ulong();
    j["player"]["orbs"]=Json::array();for(int i=0;i<b.player.orbSlots;++i)j["player"]["orbs"].push_back({{"orb",int(b.player.orbs[i])},{"data",b.player.orbData[i]}});
    j["player"]["statuses"]=Json::array();
    for(int k=1;k<=int(PlayerStatus::THE_BOMB);++k) {
        const auto status=static_cast<PlayerStatus>(k);
        if(!b.player.hasStatusRuntime(status)) continue;
        const auto it=b.player.statusMap.find(status);
        const int value=(status==PlayerStatus::STRENGTH || status==PlayerStatus::DEXTERITY || status==PlayerStatus::FOCUS || status==PlayerStatus::ARTIFACT)
            ? b.player.getStatusRuntime(status) : (it==b.player.statusMap.end() ? 1 : it->second);
        const bool applied=k<=int(PlayerStatus::WEAK) && (b.player.justAppliedBits & (1u<<k));
        j["player"]["statuses"].push_back({{"status",k},{"value",value},{"just_applied",applied}});
    }
    j["player"]["relics"]=Json::array();for(int i=0;i<128;++i)if(b.player.hasRelicRuntime(static_cast<sts::RelicId>(i)))j["player"]["relics"].push_back(i);
    j["monsters"]={{"monsters_alive",b.monsters.monstersAlive},{"monster_count",b.monsters.monsterCount},
        {"extra_roll_move_on_turn",b.monsters.extraRollMoveOnTurn.to_ulong()},{"skip_turn",b.monsters.skipTurn.to_ulong()},{"slots",Json::array()}};
    for(const auto& m:b.monsters.arr)j["monsters"]["slots"].push_back({{"index",m.idx},{"id",int(m.id)},
        {"hp",m.curHp},{"max_hp",m.maxHp},{"block",m.block},{"is_escaping",m.isEscapingB},{"half_dead",m.halfDead},
        {"escape_next",m.escapeNext},{"move_history",{int(m.moveHistory[0]),int(m.moveHistory[1])}},
        {"statuses",Json::array()},{"unique_power_0",m.uniquePower0},{"unique_power_1",m.uniquePower1},{"misc_info",m.miscInfo}});
    for(int i=0;i<5;++i) {
        const auto& m=b.monsters.arr[i];
        for(int k=0;k<int(sts::MonsterStatus::INVALID);++k) {
            const auto status=static_cast<sts::MonsterStatus>(k);
            if(!m.hasStatusInternal(status) && !(status==sts::MonsterStatus::STRENGTH && m.strength)) continue;
            bool applied=(status==sts::MonsterStatus::VULNERABLE && (m.statusBits & (1ull<<63)))
                || (status==sts::MonsterStatus::WEAK && (m.statusBits & (1ull<<62)))
                || (status==sts::MonsterStatus::RITUAL && (m.statusBits & (1ull<<61)));
            j["monsters"]["slots"][i]["statuses"].push_back({{"status",k},{"value",m.getStatusInternal(status)},{"just_applied",applied}});
        }
    }
    j["cards"]={{"next_unique_card_id",b.cards.nextUniqueCardId},{"hand",Json::array()},
        {"draw",card_list(b.cards.drawPile)},{"discard",card_list(b.cards.discardPile)},{"exhaust",card_list(b.cards.exhaustPile)},
        {"stasis",card_list(b.cards.stasisCards)}};
    for(int i=0;i<b.cards.cardsInHand;++i)j["cards"]["hand"].push_back(card_json(b.cards.hand[i]));
    j["nightmare_cards"]=card_list(b.nightmareCards);j["nightmare_copy_counts"]=Json::array();for(int v:b.nightmareCopyCounts)j["nightmare_copy_counts"].push_back(v);
    j["potions"]={{"count",b.potionCount},{"capacity",b.potionCapacity},{"slots",Json::array()}};
    for(auto p:b.potions)j["potions"]["slots"].push_back(int(p));
    return j;
}
sts::BattleContext battle_restore(const Json& j) {
    sts::BattleContext b{};const auto raw=Json::parse(j.at("extra_state_json").get<std::string>());
    restore_battle(b,raw.at("battle"));restore_player(b.player,raw.at("player"));restore_counts(b.cards,raw.at("counts"));
    for(const auto& pair:raw.at("player").at("status_map"))b.player.statusMap[static_cast<PlayerStatus>(pair[0].get<int>())]=pair[1].get<int>();
    b.player.orangePelletsCardTypesPlayed=raw.at("player").at("orange_pellets").get<unsigned long>();
    int i=0;for(const auto& orb:j.at("player").at("orbs")){b.player.orbs[i]=static_cast<Orb>(orb.at("orb").get<int>());b.player.orbData[i++]=orb.at("data");}
    b.miscBits=j.at("misc_bits").get<unsigned long>();
    rng_restore(b.aiRng,j.at("rng").at("ai"));
    rng_restore(b.cardRandomRng,j.at("rng").at("card_random"));
    rng_restore(b.miscRng,j.at("rng").at("misc"));
    rng_restore(b.monsterHpRng,j.at("rng").at("monster_hp"));
    rng_restore(b.potionRng,j.at("rng").at("potion"));
    rng_restore(b.shuffleRng,j.at("rng").at("shuffle"));
    const auto& mg=j.at("monsters");b.monsters.monstersAlive=mg.at("monsters_alive");b.monsters.monsterCount=mg.at("monster_count");
    b.monsters.extraRollMoveOnTurn=mg.at("extra_roll_move_on_turn").get<unsigned long>();b.monsters.skipTurn=mg.at("skip_turn").get<unsigned long>();
    for(int k=0;k<5;++k){restore_monster(b.monsters.arr[k],raw.at("monsters").at(k));for(int h=0;h<2;++h)b.monsters.arr[k].moveHistory[h]=static_cast<sts::MMID>(mg.at("slots").at(k).at("move_history").at(h).get<int>());}
    const auto& cards=j.at("cards");b.cards.cardsInHand=cards.at("hand").size();i=0;for(const auto& c:cards.at("hand"))b.cards.hand[i++]=card_restore(c);
    for(const auto& c:cards.at("draw"))b.cards.drawPile.push_back(card_restore(c));
    for(const auto& c:cards.at("discard"))b.cards.discardPile.push_back(card_restore(c));
    for(const auto& c:cards.at("exhaust"))b.cards.exhaustPile.push_back(card_restore(c));
    for(int k=0;k<2;++k)b.cards.stasisCards[k]=card_restore(cards.at("stasis").at(k));
    for(const auto& c:j.at("nightmare_cards"))b.nightmareCards.push_back(card_restore(c));
    for(const auto& c:j.at("nightmare_copy_counts"))b.nightmareCopyCounts.push_back(c.get<int>());
    for(int k=0;k<5;++k)b.potions[k]=static_cast<sts::Potion>(j.at("potions").at("slots").at(k).get<int>());
    return b;
}
} // namespace stsrl
