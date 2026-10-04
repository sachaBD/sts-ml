#include "agents/combat/pv/turn_targets.hpp"
#include "agents/combat/pv/search.hpp"
namespace stsrl::pv {
TurnPrefixTarget turn_prefix_target(const TurnNode& root, std::span<const std::uint32_t> prefix,
                                   const sts::BattleContext& state) {
    struct Group { std::size_t visits = 0; double sum = 0; };
    std::map<std::uint64_t, Group> groups;
    TurnPrefixTarget result;
    double sum = 0;
    for (const auto& child : root.children) {
        if (!child.visits || child.sequence.size() <= prefix.size() ||
            !std::equal(prefix.begin(), prefix.end(), child.sequence.begin())) continue;
        const sts::search::Action next{child.sequence[prefix.size()]};
        if (!next.isValidAction(state)) throw std::runtime_error{"turn targets: prefix next action illegal"};
        auto& group = groups[action_key(state, next)];
        const double value = std::clamp(child.q(), 0.0, 100.0);
        group.visits += child.visits; group.sum += child.visits * value;
        result.visits += child.visits; sum += child.visits * value;
    }
    if (!result.visits) throw std::runtime_error{"turn targets: no visit mass for played prefix"};
    for (auto move : legal_actions(state)) {
        const auto found = groups.find(action_key(state, move));
        if (found == groups.end()) continue;
        const auto& group = found->second;
        result.children.push_back({move.bits, group.visits, group.sum / group.visits});
        groups.erase(found);
    }
    if (!groups.empty()) throw std::runtime_error{"turn targets: action not in canonical legal menu"};
    result.value = std::clamp(sum / result.visits, 0.0, 100.0);
    return result;
}
} // namespace stsrl::pv
