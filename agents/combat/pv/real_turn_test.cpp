#include "agents/combat/pv/real_turn.hpp"
#include "agents/combat/pv/search.hpp"
#include "environments/combat/record_v4.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
using namespace stsrl::pv;
void require(bool ok,const char* why){if(!ok)throw std::runtime_error{why};}
int main(){try{
    std::ifstream file{PV_FIXTURE};nlohmann::json fight;file>>fight;
    sts::BattleContext state;state.init(stsrl::combat_v4::start_game(fight.at("start")));
    state.player=sts::Player{};state.player.cc=sts::CharacterClass::IRONCLAD;state.player.curHp=state.player.maxHp=100;
    state.player.energy=3;state.potionCount=state.potionCapacity=0;state.potions.fill(sts::Potion::EMPTY_POTION_SLOT);
    state.cards=sts::CardManager{};state.cards.cardsInHand=1;
    state.cards.hand[0]=sts::CardInstance{sts::CardId::POMMEL_STRIKE};state.cards.hand[0].uniqueId=0;
    state.cards.drawPile={sts::CardInstance{sts::CardId::DEFEND_RED},sts::CardInstance{sts::CardId::STRIKE_RED}};
    state.cards.drawPile[0].uniqueId=1;state.cards.drawPile[1].uniqueId=2;state.cards.nextUniqueCardId=3;
    state.monsters.arr[0].curHp=1000;
    auto other=state;std::swap(other.cards.drawPile[0],other.cards.drawPile[1]);
    require(real_turn_public_signature(state)==real_turn_public_signature(other),"signature leaks hidden order");
    std::array<sts::BattleContext,2> particles{state,other};
    std::size_t max_batch=0;
    auto evaluate=[&](std::span<const Inputs> batch){
        max_batch=std::max(max_batch,batch.size());std::vector<Prediction> result;
        for(std::size_t i=0;i<batch.size();++i)result.push_back({i%2?120.f:-20.f,{}});return result;
    };
    RealTurnCaps caps;caps.max_seconds=10;
    auto draw=decide_real_turn(state,particles,evaluate,caps);
    require(!draw.stats.fallback && draw.value==50,"leaf values not clamped and averaged before maximization");
    require(draw.stats.reveal_leaves>0 && draw.stats.mean_leaf_depth==1,"continued after information reveal");
    require(draw.action.isValidAction(state) && max_batch<=32,"bad real action/batch cap");
    auto limited=caps;limited.max_leaves=1;
    require(decide_real_turn(state,particles,evaluate,limited).stats.reason=="leaves","partial leaf-cap result published");
    limited=caps;limited.max_sequences=1;
    require(decide_real_turn(state,particles,evaluate,limited).stats.reason=="sequences","sequence cap not atomic");
    limited=caps;limited.max_actions=1;
    auto quiet=state;quiet.cards.hand[0]=sts::CardInstance{sts::CardId::DEFEND_RED};
    std::array<sts::BattleContext,2> quiets{quiet,quiet};
    require(decide_real_turn(quiet,quiets,evaluate,limited).stats.reason=="actions","path cap not atomic");
    limited=caps;limited.max_bytes=1;
    require(decide_real_turn(state,particles,evaluate,limited).stats.reason=="memory","memory cap not atomic");
    limited=caps;limited.max_seconds=1e-12;
    require(decide_real_turn(state,particles,evaluate,limited).stats.reason=="seconds","time cap not atomic");
    auto win=quiet;win.cards.hand[0]=sts::CardInstance{sts::CardId::STRIKE_RED};win.monsters.arr[0].curHp=1;
    std::array<sts::BattleContext,2> wins{win,win};
    auto terminal=decide_real_turn(win,wins,evaluate,caps);
    require(terminal.value==100 && terminal.stats.terminal_leaves>0,"terminal scoring wrong");
    auto codex=quiet;codex.cards.cardsInHand=0;codex.player.setHasRelic<sts::RelicId::NILRYS_CODEX>(true);
    std::array<sts::BattleContext,2> same{codex,codex};
    auto choices=decide_real_turn(codex,same,evaluate,caps);
    require(!choices.stats.fallback && choices.stats.end_turn_leaves>0 && choices.stats.mean_leaf_depth>1,
            "mandatory END_TURN choices not completed");
    same[1].cardRandomRng=sts::Random{891};
    auto callback=decide_real_turn(codex,same,evaluate,caps);
    require(callback.stats.fallback && callback.stats.reason=="callback","reveal with callbacks did not fall back");
    auto menu=codex;menu.inputState=sts::InputState::CARD_SELECT;menu.cardSelectInfo.cardSelectTask=sts::CardSelectTask::DISCOVERY;
    menu.cardSelectInfo.cards[0]=sts::CardId::DEFEND_RED;menu.cardSelectInfo.cards[1]=sts::CardId::STRIKE_RED;menu.cardSelectInfo.cards[2]=sts::CardId::BASH;
    auto different=menu;different.cardSelectInfo.cards[0]=sts::CardId::POMMEL_STRIKE;
    require(observation_key(menu)==observation_key(different) && real_turn_public_signature(menu)!=real_turn_public_signature(different),
            "augmented signature missed public choice reveal");
    std::cout<<"real-turn tests passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
