#include "agents/combat/nn/tokens.hpp"
#include "environments/combat/public.hpp"
#include "environments/combat/scenarios/jaw_worm.hpp"
#include "environments/combat/state_codec.hpp"
#include <algorithm>
#include <iostream>

namespace {
void buffer(const char* data, std::size_t size) {
    const auto length = std::uint32_t(size);
    std::cout.write(reinterpret_cast<const char*>(&length), sizeof length);
    std::cout.write(data, std::streamsize(size));
}
}

int main() {
    const auto state = stsrl::scenarios::jaw_worm(1);
    auto actions = stsrl::combat::legal_actions(state);
    std::ranges::reverse(actions);
    std::string packed;
    stsrl::combat::pack(state, packed);
    buffer(packed.data(), packed.size());
    std::string order;
    for (const auto action : actions) order.append(reinterpret_cast<const char*>(&action.bits), sizeof action.bits);
    buffer(order.data(), order.size());
    for (const auto encoding : {stsrl::nn::Encoding::entity, stsrl::nn::Encoding::inventory_v1}) {
        const auto tokens = stsrl::nn::encode(state, actions, encoding);
        buffer(reinterpret_cast<const char*>(tokens.data.data()), tokens.data.size() * sizeof(float));
    }
    return std::cout ? 0 : 1;
}
