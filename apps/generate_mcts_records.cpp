#include "agents/mcts_agent.hpp"
#include "scenarios/slime_boss.hpp"
#include <nlohmann/json.hpp>
#include <iostream>
#include <vector>

using json = nlohmann::json;
namespace {
template <size_t N> json array(const std::array<float, N>& x) { return x; }
json encode(const stsrl::EncodedCombatState& s) {
 json j{{"encoding_version",s.version},{"global_numeric",array(s.global.numeric)},{"input_state",s.global.input_state},{"card_selection_task",s.global.card_selection_task},{"cards",json::array()},{"monsters",json::array()},{"card_monster_interactions",json::array()}};
 for(auto& x:s.cards) j["cards"].push_back({{"card_id",x.card_id},{"zone",int(x.zone)},{"card_type",int(x.card_type)},{"target_type",int(x.target_type)},{"numeric",array(x.numeric)}});
 for(auto& x:s.monsters) j["monsters"].push_back({{"monster_id",x.monster_id},{"move_id",x.move_id},{"numeric",array(x.numeric)}});
 for(auto& x:s.card_monster_interactions) j["card_monster_interactions"].push_back({{"card_index",x.card_index},{"monster_index",x.monster_index},{"numeric",array(x.numeric)}});
 return j;
}
}
int main(int argc,char**argv) {
 if(argc != 6) { std::cerr<<"usage: generate_mcts_records seed count simulations rollout exploration\n"; return 2; }
 const auto first=std::stoull(argv[1]), count=std::stoull(argv[2]);
 const stsrl::MctsConfig config{std::stoull(argv[3]),std::stoull(argv[4]),std::stod(argv[5])};
 for(std::uint64_t episode=0;episode<count;++episode) {
  const auto seed=first+episode; auto env=stsrl::scenarios::slime_boss(seed); stsrl::MctsAgent agent{seed,config}; std::vector<json> rows; int index=0;
  while(!env.done()) { auto decision=env.decision(); auto result=agent.search(env); rows.push_back(encode(decision.encoding)); auto& row=rows.back(); row["episode_id"]=episode;row["seed"]=seed;row["decision_index"]=index++;row["mcts_value"]=result.root_value;row["root_visits"]=result.root_visits;row["chosen_action"]=result.chosen_action; env.step(result.chosen_action); }
  const int outcome=env.won() ? 1 : -1; const auto value=env.combat_value();
  for(auto& row:rows) { row["terminal_outcome"]=outcome;row["final_player_hp"]=env.player_hp();row["final_player_max_hp"]=env.player_max_hp();row["terminal_value"]=value; auto bytes=json::to_msgpack(row); std::cout.write(reinterpret_cast<const char*>(bytes.data()),bytes.size()); }
  std::cerr<<"episode "<<episode+1<<'/'<<count<<"\n";
 }
}
