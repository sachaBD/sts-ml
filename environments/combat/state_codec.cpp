#include "environments/combat/state_codec.hpp"
#include "combat/StateFields.h"
#include <cstring>
#include <stdexcept>
#include <type_traits>

namespace stsrl::combat {
namespace {
// RNGs are leaves in the simulator's walk; write their fields, not their padding.
template <class State, class Scalar> void state_scalars(State& state, Scalar scalar) {
    sts::forEachField(state, [&](auto& leaf) {
        if constexpr (std::is_same_v<std::decay_t<decltype(leaf)>, sts::Random>) {
            scalar(leaf.seed0);
            scalar(leaf.seed1);
            scalar(leaf.counter);
        } else {
            scalar(leaf);
        }
    });
}
}  // namespace

bool packable(const sts::BattleContext& state) { return state.actionQueue.size == 0; }

void pack(const sts::BattleContext& state, std::string& out) {
    if (!packable(state)) throw std::invalid_argument{"pack: queued actions cannot be stored"};
    state_scalars(state, [&](const auto& scalar) {
        out.append(reinterpret_cast<const char*>(&scalar), sizeof scalar);
    });
}

sts::BattleContext unpack(std::span<const char> bytes) {
    sts::BattleContext state;
    state_scalars(state, [&](auto& scalar) {
        if (bytes.size() < sizeof scalar) throw std::invalid_argument{"unpack: truncated state"};
        std::memcpy(&scalar, bytes.data(), sizeof scalar);
        bytes = bytes.subspan(sizeof scalar);
    });
    if (!bytes.empty()) throw std::invalid_argument{"unpack: trailing bytes"};
    return state;
}
}  // namespace stsrl::combat
