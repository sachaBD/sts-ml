#include "agents/combat/pv/real_turn.hpp"
#include "agents/combat/pv/search.hpp"
#include "agents/combat/pv/turn_state_key.hpp"
#include <chrono>
#include <cmath>
#include <memory>
#include <map>
#include <unordered_set>

namespace stsrl::pv {
namespace {
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point start) { return std::chrono::duration<double>(Clock::now()-start).count(); }
template<class T> void append(std::string& key,const T& value) { key.append(reinterpret_cast<const char*>(&value),sizeof(value)); }
std::map<std::uint64_t,sts::search::Action> menu(const sts::BattleContext& state) {
    std::map<std::uint64_t,sts::search::Action> result;
    for(auto action:legal_actions(state))
        if(!result.emplace(action_key(state,action),action).second)
            throw std::runtime_error{"real-turn: non-bijective public menu"};
    return result;
}
std::size_t state_bytes(const sts::BattleContext& state) {
    return sizeof(state)+65536+4*sizeof(sts::CardInstance)*(state.cards.drawPile.size()+state.cards.discardPile.size()+state.cards.exhaustPile.size());
}
}
std::string real_turn_public_signature(const sts::BattleContext& state) {
    std::string key;append(key,observation_key(state));
    std::vector<sts::search::Action> moves;
    const auto choices=menu(state);append(key,choices.size());
    for(const auto& [id,action]:choices) {append(key,id);moves.push_back(action);}
    auto inputs=encode(state,moves);
    for(std::size_t i=0;i<inputs.size();++i) {
        // Sets are permutation-invariant public inputs; hidden draw-order permutations are not a reveal.
        if(i>0 && i<5) {
            std::vector<std::vector<float>> rows;
            for(std::size_t j=0;j<inputs[i].size();j+=widths[i])
                rows.emplace_back(inputs[i].begin()+j,inputs[i].begin()+j+widths[i]);
            std::sort(rows.begin(),rows.end());inputs[i].clear();
            for(const auto& row:rows) inputs[i].insert(inputs[i].end(),row.begin(),row.end());
        }
        append(key,inputs[i].size());
        key.append(reinterpret_cast<const char*>(inputs[i].data()),inputs[i].size()*sizeof(float));
    }
    return key;
}
RealTurnResult decide_real_turn(const sts::BattleContext& observed,std::span<const sts::BattleContext> particles,
                                const RealTurnEvaluate& evaluate,RealTurnCaps caps) {
    if(particles.empty() || !caps.max_sequences || !caps.max_leaves || !caps.max_actions || caps.max_actions>512 ||
       !caps.max_bytes || !std::isfinite(caps.max_seconds) || caps.max_seconds<=0)
        throw std::invalid_argument{"real-turn: invalid particles/caps"};
    const auto start=Clock::now();RealTurnResult result;auto& stats=result.stats;
    validate_root_state(observed);
    const auto root_signature=real_turn_public_signature(observed);
    for(const auto& state:particles) {
        validate_root_state(state);
        if(real_turn_public_signature(state)!=root_signature)
            throw std::runtime_error{"real-turn: particle changed public root"};
    }
    auto cap=[&](const char* reason){stats.fallback=true;stats.reason=reason;};
    auto timed=[&](){if(elapsed(start)>caps.max_seconds){cap("seconds");return true;}return false;};
    struct Frame {std::vector<sts::BattleContext> states;std::vector<sts::search::Action> moves;std::size_t next=0,bytes=0;bool ending=false;};
    struct Leaf {std::uint32_t first=0;std::size_t depth=0;double sum=0;};
    std::vector<std::unique_ptr<Frame>> stack;std::vector<Leaf> leaves;
    std::unordered_set<std::string> seen;std::vector<std::uint32_t> path;
    std::vector<Inputs> pending;std::vector<std::size_t> indices;
    std::size_t stack_bytes=0,leaf_bytes=0,input_bytes=0;
    auto flush=[&](){
        if(pending.empty() || stats.fallback || timed())return;
        ++stats.network_calls;stats.evaluated_states+=pending.size();
        const auto predictions=evaluate(pending);
        if(predictions.size()!=pending.size())throw std::runtime_error{"real-turn: prediction batch mismatch"};
        for(std::size_t i=0;i<predictions.size();++i){
            if(!std::isfinite(predictions[i].value))throw std::runtime_error{"real-turn: nonfinite value"};
            leaves[indices[i]].sum+=std::clamp(double(predictions[i].value),0.0,100.0);
        }
        pending.clear();indices.clear();input_bytes=0;timed();
    };
    auto push=[&](std::vector<sts::BattleContext> states,bool ending){
        std::size_t bytes=sizeof(Frame)+128;for(const auto& state:states)bytes+=state_bytes(state);
        if(stack_bytes+leaf_bytes+4*input_bytes+bytes>caps.max_bytes){cap("memory");return;}
        auto frame=std::make_unique<Frame>();frame->states=std::move(states);frame->ending=ending;frame->bytes=bytes;
        frame->moves=legal_actions(frame->states.front());
        stack_bytes+=bytes;stack.push_back(std::move(frame));
    };
    push({particles.begin(),particles.end()},observed.endTurnQueued || observed.turnHasEnded);
    while(!stack.empty() && !stats.fallback){
        auto& frame=*stack.back();
        if(frame.next==frame.moves.size()) {stack_bytes-=frame.bytes;stack.pop_back();if(!path.empty())path.pop_back();continue;}
        if(stats.sequences>=caps.max_sequences){cap("sequences");break;}
        if(path.size()>=caps.max_actions){cap("actions");break;}
        if(timed())break;
        const auto action=frame.moves[frame.next++];const auto id=action_key(frame.states.front(),action);
        auto next=frame.states;bool terminal=false,ending=frame.ending || action.getActionType()==sts::search::ActionType::END_TURN;
        for(auto& state:next){
            const auto choices=menu(state);const auto found=choices.find(id);
            if(found==choices.end())throw std::runtime_error{"real-turn: common action has no public match"};
            if(!found->second.isValidAction(state))throw std::runtime_error{"real-turn: mapped action illegal"};
            found->second.execute(state);
            if(state.unsupportedEffectKind!=sts::UnsupportedEffectKind::NONE)throw std::runtime_error{"real-turn: unsupported simulator effect"};
            terminal|=state.outcome!=sts::Outcome::UNDECIDED;ending|=state.turn!=observed.turn;
        }
        // Root first action must be the real menu representative, not a particle's hidden pile index.
        const auto real_action=path.empty()?menu(observed).at(id):action;
        path.push_back(real_action.bits);
        if(timed())break;
        bool reveal=false;
        if(!terminal){
            const auto signature=real_turn_public_signature(next.front());
            for(std::size_t i=1;i<next.size();++i)reveal|=real_turn_public_signature(next[i])!=signature;
        }
        bool boundary=ending;
        for(const auto& state:next)boundary&=state.inputState==sts::InputState::PLAYER_NORMAL;
        if(!terminal && !reveal && !boundary){push(std::move(next),ending);continue;}
        ++stats.sequences;
        std::string key;
        for(const auto& state:next){
            if(state.outcome==sts::Outcome::UNDECIDED && state.actionQueue.size){cap("callback");break;}
            const auto part=state.outcome==sts::Outcome::UNDECIDED?turn_state_key(state):
                std::string(state.outcome==sts::Outcome::PLAYER_VICTORY?"terminal:win":"terminal:loss");
            append(key,part.size());key+=part;
        }
        if(stats.fallback)break;
        if(seen.find(key)==seen.end()){
            if(leaves.size()>=caps.max_leaves){cap("leaves");break;}
            leaf_bytes+=2*(key.capacity()+sizeof(Leaf)+256);
            if(stack_bytes+leaf_bytes+4*input_bytes>caps.max_bytes){cap("memory");break;}
            seen.insert(std::move(key));leaves.push_back({path.front(),path.size(),0});stats.leaves=leaves.size();
            if(terminal)++stats.terminal_leaves;else if(reveal)++stats.reveal_leaves;else ++stats.end_turn_leaves;
            const auto index=leaves.size()-1;
            for(const auto& state:next){
                if(state.outcome!=sts::Outcome::UNDECIDED){leaves[index].sum+=state.outcome==sts::Outcome::PLAYER_VICTORY?100:0;continue;}
                auto inputs=encode(state,legal_actions(state));
                for(const auto& rows:inputs)input_bytes+=rows.capacity()*sizeof(float);
                if(stack_bytes+leaf_bytes+4*input_bytes>caps.max_bytes){cap("memory");break;}
                pending.push_back(std::move(inputs));indices.push_back(index);
                if(pending.size()==32)flush();
                if(stats.fallback)break;
            }
        }
        path.pop_back();
    }
    if(!stats.fallback)flush();
    if(!stats.fallback)timed();
    stats.seconds=elapsed(start);stats.time_overshoot=stats.seconds>caps.max_seconds;
    if(!stats.fallback){
        if(leaves.empty())throw std::runtime_error{"real-turn: no complete leaves"};
        double best=-1,depth=0;
        for(const auto& leaf:leaves){const double value=leaf.sum/particles.size();depth+=leaf.depth;
            if(value>best){best=value;result.action=sts::search::Action{leaf.first};}}
        result.value=best;stats.mean_leaf_depth=depth/leaves.size();
    }
    return result;
}
} // namespace stsrl::pv
