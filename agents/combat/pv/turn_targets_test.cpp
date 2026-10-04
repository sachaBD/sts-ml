#include "agents/combat/pv/turn_targets.hpp"
#include "environments/combat/record_v4.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
using namespace stsrl::pv;
void require(bool ok, const char* message) { if(!ok) throw std::runtime_error{message}; }
int main() {
    try {
        std::ifstream file{PV_FIXTURE}; nlohmann::json fight; file>>fight;
        sts::BattleContext state; state.init(stsrl::combat_v4::start_game(fight.at("start")));
        state.player=sts::Player{}; state.player.cc=sts::CharacterClass::IRONCLAD;
        state.player.curHp=state.player.maxHp=100; state.player.energy=3;
        state.cards=sts::CardManager{}; state.cards.cardsInHand=2;
        for(int i=0;i<2;++i) {state.cards.hand[i]=sts::CardInstance{sts::CardId::DEFEND_RED};state.cards.hand[i].uniqueId=i;}
        state.cards.nextUniqueCardId=2; state.potionCount=state.potionCapacity=0;
        state.potions.fill(sts::Potion::EMPTY_POTION_SLOT);
        const auto source=state;
        const sts::search::Action card{sts::search::ActionType::CARD,0,0}, end{sts::search::ActionType::END_TURN};
        TurnNode root;
        auto add=[&](std::vector<std::uint32_t> sequence,std::size_t visits,double q) {
            TurnChild child;child.sequence=std::move(sequence);child.visits=visits;child.sum=visits*q;root.children.push_back(std::move(child));
        };
        add({card.bits,card.bits,end.bits},3,30);add({card.bits,end.bits},1,90);add({end.bits},2,60);
        auto start=turn_prefix_target(root,{},state);
        require(start.visits==6 && start.value==50 && start.children.size()==2,"root weighted target wrong");
        std::vector<std::uint32_t> prefix{card.bits};card.execute(state);
        auto next=turn_prefix_target(root,prefix,state);
        require(next.visits==4 && next.value==45 && next.children.size()==2,"prefix filtering/weighted Q wrong");
        prefix.push_back(card.bits);card.execute(state);
        auto forced=turn_prefix_target(root,prefix,state);
        require(forced.visits==3 && forced.value==30 && forced.children.size()==1 && forced.children[0].action==end.bits,
                "forced decision target wrong");
        root.children[0].sum=3*123.;require(turn_prefix_target(root,prefix,state).value==100,"target not clamped");
        bool rejected=false;try{turn_prefix_target(root,std::array<std::uint32_t,1>{end.bits},state);}catch(const std::runtime_error&){rejected=true;}
        require(rejected,"zero-mass prefix invented target");
        TurnSearchCaps caps;caps.max_seconds=10;
        auto evaluate=[](std::span<const Inputs> batch){return std::vector<Prediction>(batch.size(),{40,{}});};
        std::mt19937_64 rng{91};TurnSearch tree{source,caps};
        const auto chosen=tree.decide(8,evaluate,.25,&rng,true);
        bool visited=false;for(const auto& child:tree.root().children)if(child.sequence==chosen)visited|=child.visits>0;
        require(visited,"sampled unvisited child");
        double sum=0;bool changed=false;for(const auto& child:tree.root().children){sum+=child.prior;changed|=std::abs(child.prior-1.0/tree.root().children.size())>1e-9;}
        require(changed,"Dirichlet noise not applied");
        require(std::abs(sum-1)<1e-9,"noisy priors not normalized");
        std::cout<<"turn targets tests passed\n";return 0;
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
