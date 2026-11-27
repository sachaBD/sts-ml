#pragma once
#include "agents/combat/player/targets.hpp"
#include "agents/combat/search/search.hpp"
#include <iosfwd>
#include <string>

namespace stsrl::player {
// Owns one fight's replay rows, not the decision of which roots to keep.
class FightSamples {
public:
    void record(const sts::BattleContext&, std::vector<search::Action>, std::vector<float> policy,
                ThinkingTargets);
    void write(std::ostream&, std::span<const Vitals> turn_starts, const sts::BattleContext& end) const;
    std::size_t size() const { return samples_.size(); }

private:
    struct Sample {
        std::string state;
        std::vector<search::Action> actions;
        std::vector<float> policy;
        Vitals vitals;
        ThinkingTargets gains;
    };
    static void write_sample(std::ostream&, const Sample&, std::span<const Vitals>, const sts::BattleContext&);
    std::vector<Sample> samples_;
};
}  // namespace stsrl::player
