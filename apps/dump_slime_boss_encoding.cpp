#include "scenarios/slime_boss.hpp"
#include <iostream>

namespace {
template <size_t N> void numeric(const std::array<float, N>& x) { std::cout << '['; for (size_t i=0;i<N;++i) { if(i) std::cout << ','; std::cout << x[i]; } std::cout << ']'; }
void card(const stsrl::CardToken& x) { std::cout << "{\"card_id\":" << x.card_id << ",\"zone\":" << int(x.zone) << ",\"card_type\":" << int(x.card_type) << ",\"target_type\":" << int(x.target_type) << ",\"numeric\":"; numeric(x.numeric); std::cout << '}'; }
void monster(const stsrl::MonsterToken& x) { std::cout << "{\"monster_id\":" << x.monster_id << ",\"move_id\":" << x.move_id << ",\"numeric\":"; numeric(x.numeric); std::cout << '}'; }
}
int main(int argc, char** argv) {
 const auto state=stsrl::scenarios::slime_boss(argc > 1 ? std::stoull(argv[1]) : 1).decision().encoding;
 std::cout << "{\"version\":" << state.version << ",\"global_numeric\":"; numeric(state.global.numeric);
 std::cout << ",\"input_state\":" << state.global.input_state << ",\"card_selection_task\":" << state.global.card_selection_task << ",\"cards\":[";
 for(size_t i=0;i<state.cards.size();++i){if(i)std::cout<<',';card(state.cards[i]);} std::cout<<"],\"monsters\":[";
 for(size_t i=0;i<state.monsters.size();++i){if(i)std::cout<<',';monster(state.monsters[i]);} std::cout<<"],\"card_monster_interactions\":[";
 for(size_t i=0;i<state.card_monster_interactions.size();++i){if(i)std::cout<<',';auto& x=state.card_monster_interactions[i];std::cout<<"{\"card_index\":"<<x.card_index<<",\"monster_index\":"<<int(x.monster_index)<<",\"numeric\":";numeric(x.numeric);std::cout<<'}';} std::cout<<"]}\n";
}
