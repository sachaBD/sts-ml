#include "agents/combat/nn/training/sample.hpp"
#include "environments/combat/state_codec.hpp"
#include <cstring>
#include <stdexcept>
#include <utility>

namespace stsrl::nn::training {
void write_sample(std::ostream& out, std::string_view state, std::span<const sts::search::Action> actions,
                  std::span<const float> policy, const Targets& targets) {
    if (actions.empty() || actions.size() != policy.size()) throw std::invalid_argument{"sample: policy/action mismatch"};
    if (!std::in_range<std::uint32_t>(state.size()) || !std::in_range<std::uint32_t>(actions.size()))
        throw std::length_error{"sample: record too large"};
    const SampleHeader header{combat::state_codec_version, std::uint32_t(state.size()), std::uint32_t(actions.size()), targets};
    out.write(reinterpret_cast<const char*>(&header), sizeof header);
    out.write(state.data(), std::streamsize(state.size()));
    for (const auto action : actions) out.write(reinterpret_cast<const char*>(&action.bits), sizeof action.bits);
    out.write(reinterpret_cast<const char*>(policy.data()), std::streamsize(policy.size_bytes()));
}

infra::Tokens sample_tokens(std::span<const char> record, Encoding encoding) {
    if (record.size() < sizeof(SampleHeader)) throw std::invalid_argument{"sample: truncated header"};
    SampleHeader header;
    std::memcpy(&header, record.data(), sizeof header);
    if (header.state_version != combat::state_codec_version)
        throw std::invalid_argument{"sample: unsupported state version"};
    record = record.subspan(sizeof header);
    if (!header.actions || header.state_bytes > record.size() ||
        header.actions > (record.size() - header.state_bytes) / (sizeof(std::uint32_t) + sizeof(float)) ||
        record.size() != header.state_bytes + std::size_t(header.actions) * (sizeof(std::uint32_t) + sizeof(float)))
        throw std::invalid_argument{"sample: invalid record size"};
    const auto state = combat::unpack(record.first(header.state_bytes));
    record = record.subspan(header.state_bytes);
    std::vector<sts::search::Action> actions;
    actions.reserve(header.actions);
    for (std::uint32_t i = 0; i < header.actions; ++i) {
        std::uint32_t bits;
        std::memcpy(&bits, record.data() + i * sizeof bits, sizeof bits);
        actions.emplace_back(bits);
    }
    return encode(state, actions, encoding);
}
}  // namespace stsrl::nn::training
