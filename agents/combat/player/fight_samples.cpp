#include "agents/combat/player/fight_samples.hpp"
#include "environments/combat/state_codec.hpp"
#include <ostream>

namespace stsrl::player {
void FightSamples::record(const sts::BattleContext& state, std::vector<search::Action> actions, std::vector<float> policy) {
    Sample sample{{}, std::move(actions), std::move(policy), vitals(state)};
    combat::pack(state, sample.state);
    samples_.push_back(std::move(sample));
}

void FightSamples::write(std::ostream& out, std::span<const Vitals> turn_starts, const sts::BattleContext& end) const {
    for (const auto& sample : samples_) write_sample(out, sample, turn_starts, end);
}

void FightSamples::write_sample(std::ostream& out, const Sample& sample, std::span<const Vitals> turn_starts,
                                const sts::BattleContext& end) {
    const auto target = targets(sample.vitals, turn_starts, end);
    const std::uint32_t header[]{combat::state_codec_version, std::uint32_t(sample.state.size()),
                                std::uint32_t(sample.actions.size())};
    out.write(reinterpret_cast<const char*>(header), sizeof header);
    out.write(reinterpret_cast<const char*>(target.data()), sizeof target);
    out.write(sample.state.data(), std::streamsize(sample.state.size()));
    for (const auto action : sample.actions)
        out.write(reinterpret_cast<const char*>(&action.bits), sizeof action.bits);
    out.write(reinterpret_cast<const char*>(sample.policy.data()), std::streamsize(sample.policy.size() * sizeof(float)));
}
}  // namespace stsrl::player
