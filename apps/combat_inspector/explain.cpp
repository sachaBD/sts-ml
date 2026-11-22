#include "apps/combat_inspector/inspection.hpp"
#include "agents/combat/nn/network.hpp"
#include <iostream>

int main() try {
    const auto request = nlohmann::json::parse(std::cin);
    const auto network = stsrl::nn::load_network(request.at("checkpoint").get<std::string>(), stsrl::nn::Backends::cpu);
    const auto response = stsrl::inspector::inspect(request, network.encoding, [&](auto requests) {
        std::vector<stsrl::infra::Input> inputs;
        for (const auto& tokens : requests) inputs.push_back({&tokens, {}});
        return network.backends.front()->evaluate(inputs);
    });
    std::cout << response.dump() << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "combat_explain: " << error.what() << '\n';
    return 1;
}
