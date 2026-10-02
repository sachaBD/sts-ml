// Public observation contract: probabilities/history and no future RNG/schedule leakage.
#include "environments/overworld/observation.hpp"
#include "environments/overworld/macro_sim.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
void check(bool ok, const char* what) { if (!ok) throw std::runtime_error{what}; }
void near(double a,double b,const char* what) { check(std::abs(a-b)<1e-6,what); }
void probabilities() {
    using R=sts::RelicId;
    sts::GameContext gc{sts::CharacterClass::IRONCLAD,42,20};
    auto obs=stsrl::overworld::observation_json(gc);
    check(obs["version"]==1 && obs["ascension"]==20,"version and difficulty");
    near(obs["card_rarity"]["hallway"]["rare"],0,"initial rarity modifier");
    gc.cardRarityFactor=-10;
    obs=stsrl::overworld::observation_json(gc);
    near(obs["card_rarity"]["hallway"]["rare"],.13,"hallway rarity");
    near(obs["card_rarity"]["elite"]["rare"],.20,"elite rarity");
    near(obs["card_rarity"]["shop"]["rare"],.19,"shop rarity");
    gc.relics.add({R::NLOTHS_GIFT,0});
    near(stsrl::overworld::observation_json(gc)["card_rarity"]["hallway"]["rare"],.19,"N'loth modifier");
    gc.potionChance=20;
    near(stsrl::overworld::observation_json(gc)["potion_roll_probability"],.6,"potion history modifier");
    gc.relics.add({R::WHITE_BEAST_STATUE,0});
    near(stsrl::overworld::observation_json(gc)["potion_roll_probability"],1,"white beast");
    gc.relics.add({R::SOZU,0});
    near(stsrl::overworld::observation_json(gc)["potion_obtain_probability"],0,"Sozu distinction");
    auto q=stsrl::overworld::question_probabilities(gc,false);
    // Mirrors integer bucket boundaries in the simulator, including float-to-int truncation.
    const auto fight=static_cast<int>(gc.monsterChance*100)/100.;
    const auto shop=static_cast<int>(gc.shopChance*100)/100.;
    near(q["fight"],fight,"question combat"); near(q["shop"],shop,"question shop");
    near(stsrl::overworld::question_probabilities(gc,true)["shop"],0,"no shop immediately after shop");
    gc.relics.add({R::JUZU_BRACELET,0});
    near(stsrl::overworld::question_probabilities(gc,false)["fight"],0,"Juzu");
    gc.relics.add({R::TINY_CHEST,3});
    near(stsrl::overworld::question_probabilities(gc,false)["treasure"],1,"Tiny Chest");
}
void history_and_leaks() {
    using ME=sts::MonsterEncounter;
    sts::GameContext gc{sts::CharacterClass::IRONCLAD,42,20};
    gc.monsterListOffset=1;gc.monsterList[0]=ME::CULTIST;
    gc.eliteMonsterListOffset=1;gc.eliteMonsterList[0]=ME::GREMLIN_NOB;
    const auto before=stsrl::overworld::observation_json(gc);
    check(before["encounters"]["possible_elites"].size()==2,"elite excludes last");
    for (auto id:before["encounters"]["possible_elites"]) check(id!=static_cast<int>(ME::GREMLIN_NOB),"last elite unavailable");
    for (auto q:before["encounters"]["next_hallway"]) check(q["id"]!=static_cast<int>(ME::CULTIST),"last hallway unavailable");
    // Change only secret future order/membership and RNG state: observations must not change.
    for (int i=1;i<static_cast<int>(gc.monsterList.size());++i) gc.monsterList[i]=ME::JAW_WORM;
    for (int i=1;i<static_cast<int>(gc.eliteMonsterList.size());++i) gc.eliteMonsterList[i]=ME::LAGAVULIN;
    gc.commonRelicPool.clear();gc.rareRelicPool.clear();gc.bossRelicPool.clear();
    gc.secondBoss=ME::THE_HEART;gc.relicRng=sts::Random(123);gc.monsterRng=sts::Random(456);
    check(before==stsrl::overworld::observation_json(gc),"must not leak secret future lists/RNG");
    check(before["relic_candidates_exact"]==false,"public candidates must not claim exact pool");
    const auto count=gc.eventList.size();
    gc.eventList.erase(gc.eventList.begin());
    check(stsrl::overworld::observation_json(gc)["events"]["normal"].size()+1==count,"consumed event excluded");
    gc.monsterListOffset=3;gc.monsterList[2]=ME::SMALL_SLIMES;
    for(auto q:stsrl::overworld::public_encounters(gc)["next_hallway"])
        check(q["id"]!=static_cast<int>(ME::LARGE_SLIME) && q["id"]!=static_cast<int>(ME::LOTS_OF_SLIMES),"first strong exclusions");
    gc.transitionToAct(4);
    const auto fourth=stsrl::overworld::observation_json(gc);
    check(fourth["encounters"]["possible_elites"]==nlohmann::json::array({static_cast<int>(ME::SHIELD_AND_SPEAR)}),"Act4 elite");
    check(fourth["encounters"]["next_hallway"].empty(),"no Act4 hallway schedule");
    check(stsrl::macro_sim::state_json(gc).contains("overworld"),"state export includes observation");
}
}
int main(int argc, char** argv) {
    if (argc == 2 && (std::string(argv[1]) == "--dump" || std::string(argv[1]) == "--dump4")) {
        sts::GameContext gc{sts::CharacterClass::IRONCLAD, 42, 20};
        if (std::string(argv[1]) == "--dump4") gc.transitionToAct(4);
        std::cout << stsrl::macro_sim::state_json(gc).dump() << '\n';
        return 0;
    }
    probabilities();history_and_leaks();
    std::cout<<"public overworld observation tests passed\n";
}
