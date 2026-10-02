#include "environments/combat/battle_snapshot.hpp"
#include "agents/combat/search/teacher_search.hpp"
#include "constants/CharacterClasses.h"
#include "constants/Rooms.h"
#include "game/GameContext.h"
#include <iostream>
#include <stdexcept>

int main() {
    namespace t=stsrl::teacher;
    const sts::MonsterEncounter encounters[]={sts::MonsterEncounter::JAW_WORM,sts::MonsterEncounter::GREMLIN_NOB,
        sts::MonsterEncounter::LAGAVULIN,sts::MonsterEncounter::THREE_SENTRIES,sts::MonsterEncounter::SLIME_BOSS,
        sts::MonsterEncounter::THE_GUARDIAN,sts::MonsterEncounter::HEXAGHOST};
    int count=0;
    for(auto encounter:encounters)for(int seed=11;seed<14;++seed) {
        sts::GameContext gc{sts::CharacterClass::IRONCLAD,static_cast<std::uint64_t>(seed),20};
        gc.floorNum=6;gc.curRoom=sts::Room::ELITE;
        sts::BattleContext b{};b.init(gc,encounter);
        b.obtainPotion(sts::Potion::FIRE_POTION);
        b.player.setHasRelic<sts::RelicId::PEN_NIB>(true);b.player.penNibCounter=8;
        const auto initial=stsrl::battle_snapshot(b);
        const auto restored=stsrl::battle_restore(initial);
        if(stsrl::battle_snapshot(restored)!=initial)throw std::runtime_error("snapshot roundtrip mismatch");
        std::vector<nlohmann::json> rows;
        const auto search=t::leaf_search({"guided_rollout",0,0},nullptr,{500,8});
        const auto end=t::play_fight(b,{{"episode_id",count}},rows,search,true,false,8,false,false);
        stsrl::CombatEnvironment replay{restored};
        for(const auto& row:rows) {
            const auto n=replay.legal_action_count();auto chosen=n;
            for(std::size_t i=0;i<n;++i)if(replay.action_bits(i)==row.at("executed_action_bits")){chosen=i;break;}
            if(chosen==n)throw std::runtime_error("illegal replay action");
            replay.step(chosen);
        }
        if(!replay.done() || stsrl::battle_snapshot(replay.battle())!=stsrl::battle_snapshot(end))
            throw std::runtime_error("final replay mismatch");
        ++count;
    }
    std::cout<<"Exact initial roundtrip and executed-action replay: "<<count<<" fights\n";
}
