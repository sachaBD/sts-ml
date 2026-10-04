#include "agents/combat/pv/turn_pimc.hpp"
#include "agents/combat/pv/search.hpp"
#include "agents/combat/search/teacher_leaves.hpp"
#include "environments/combat/record_v4.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
using namespace stsrl::pv;
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error{why}; }
int main() {
    try {
        const std::array<std::uint64_t, 3> keys{11,22,33};
        const std::array<ParticleActionValues, 2> values{{{10,{{11,90}}}, {20,{{22,50}}}}};
        const auto scores = mean_particle_scores(keys, values);
        require(scores == std::vector<double>({55,30,15}), "mean/absent-root-value aggregation wrong");
        std::ifstream file{PV_FIXTURE}; nlohmann::json fight; file >> fight;
        sts::BattleContext state; state.init(stsrl::combat_v4::start_game(fight.at("start")));
        state.player = sts::Player{}; state.player.cc = sts::CharacterClass::IRONCLAD;
        state.player.curHp = state.player.maxHp = 100; state.player.energy = 1;
        state.potionCount = state.potionCapacity = 0; state.potions.fill(sts::Potion::EMPTY_POTION_SLOT);
        state.cards = sts::CardManager{}; state.cards.cardsInHand = 2;
        for (int i=0; i<2; ++i) { state.cards.hand[i] = sts::CardInstance{sts::CardId::STRIKE_RED}; state.cards.hand[i].uniqueId=i; }
        state.cards.nextUniqueCardId = 3; state.cards.strikeCount = 2;
        const sts::search::Action a{sts::search::ActionType::CARD,0,0}, b{sts::search::ActionType::CARD,1,0};
        auto menu=state; menu.inputState=sts::InputState::CARD_SELECT;
        menu.cardSelectInfo.cardSelectTask=sts::CardSelectTask::SECRET_TECHNIQUE;
        menu.cards.drawPile.push_back(sts::CardInstance{sts::CardId::DEFEND_RED});
        menu.cards.drawPile.push_back(sts::CardInstance{sts::CardId::STRIKE_RED});
        auto reordered=menu; std::swap(reordered.cards.drawPile[0],reordered.cards.drawPile[1]);
        const sts::search::Action pick0{sts::search::ActionType::SINGLE_CARD_SELECT,0};
        const sts::search::Action pick1{sts::search::ActionType::SINGLE_CARD_SELECT,1};
        require(pick0.bits!=pick1.bits && action_key(menu,pick0)==action_key(reordered,pick1),
                "public draw-selection mapping depends on hidden position");
        TurnNode root;
        TurnChild first; first.sequence={a.bits}; first.value=90; first.visits=2; first.sum=60;
        TurnChild second; second.sequence={a.bits, b.bits}; second.value=70;
        root.children.push_back(std::move(first)); root.children.push_back(std::move(second));
        auto grouped=turn_first_action_values(state,root,12);
        require(grouped.best.size()==1 && grouped.best.begin()->second==70, "not max Q over semantic first actions");
        state.monsters.arr[0].curHp=1;
        auto particles=stsrl::teacher::public_particles(state,2);
        auto evaluate=[](std::span<const Inputs> batch) {
            std::vector<Prediction> result;
            for (const auto& input:batch) result.push_back({80,std::vector<float>(input[5].size()/widths[5],0)});
            return result;
        };
        TurnSearchCaps caps; caps.max_seconds=10;
        auto chosen=decide_turn_particles(state,particles,2,evaluate,caps);
        require(chosen.action.isValidAction(state) && chosen.action.getActionType()==sts::search::ActionType::CARD,
                "particle turn searches did not select real legal winning move");
        require(chosen.particles==2 && chosen.fallback_count==0, "particle telemetry wrong");
        caps.root_max_children=1;
        auto fallback=decide_turn_particles(state,particles,2,evaluate,caps);
        require(fallback.fallback_count==2 && fallback.action.isValidAction(state), "particle cap did not use oracle fallback");
        auto timed=caps; timed.max_seconds=1e-12;
        auto timeout=decide_turn_particles(state,particles,2,evaluate,timed);
        require(timeout.time_fallback_count==2 && timeout.time_overshoot_count==2, "time-cap particles not separately counted");
        auto hidden=state; hidden.shuffleRng=sts::Random{712}; std::swap(hidden.aiRng, hidden.miscRng);
        auto alternate=stsrl::teacher::public_particles(hidden,2);
        require(observation_key(hidden)==observation_key(state), "fixture private mutation changed public state");
        require(alternate[0].seed==particles[0].seed, "belief uses true private seed");
        require(decide_turn_particles(hidden,alternate,2,evaluate,caps).action.bits==fallback.action.bits,
                "PIMC choice leaked private RNG");
        std::cout << "turn PIMC tests passed\n"; return 0;
    } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
