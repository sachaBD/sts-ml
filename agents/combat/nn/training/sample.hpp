#pragma once
#include "agents/combat/nn/tokens.hpp"
#include <array>
#include <cstdint>
#include <ostream>
#include <span>
#include <string_view>

namespace stsrl::nn::training {
using Targets = std::array<float, layout::sample::targets::count>;
struct SampleHeader {
    std::uint32_t state_version, state_bytes, actions;
    Targets targets;
};
static_assert(sizeof(SampleHeader) == 12 + sizeof(Targets));

// Action order is the search's policy order, not a newly enumerated order.
void write_sample(std::ostream&, std::string_view state, std::span<const sts::search::Action>,
                  std::span<const float> policy, const Targets&);
infra::Tokens sample_tokens(std::span<const char> record, Encoding);
}  // namespace stsrl::nn::training
