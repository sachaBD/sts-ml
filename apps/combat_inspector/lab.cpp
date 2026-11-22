// Native GPU/CPU session server: JSON commands on stdin, one JSON response per line on stdout (see inspection.hpp).
#include "agents/combat/nn/network.hpp"
#include "apps/combat_inspector/inspection.hpp"
#include <CLI/CLI.hpp>
#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace {
using namespace stsrl;
using namespace stsrl::inspection;

// Evaluates a batch of token requests across the checkpoint's loaded GPU/CPU backends.
std::vector<infra::Prediction> evaluate(nn::Network& network, std::span<const infra::Tokens> requests) {
    std::vector<infra::Prediction> out(requests.size());
    std::vector<std::vector<std::size_t>> routed(network.backends.size());
    for (std::size_t i = 0; i < requests.size(); ++i) {
        const auto backend = std::ranges::find_if(network.backends, [&](const auto& b) { return b->accepts(requests[i]); });
        if (backend == network.backends.end()) throw std::runtime_error{"no backend accepts a request"};
        routed[backend - network.backends.begin()].push_back(i);
    }
    for (std::size_t b = 0; b < routed.size(); ++b) {
        auto& backend = *network.backends[b];
        const auto& rows = routed[b];
        for (std::size_t first = 0; first < rows.size(); first += backend.max_batch()) {
            const auto last = std::min(rows.size(), first + backend.max_batch());
            std::vector<std::vector<std::uint16_t>> packed(last - first);
            std::vector<infra::Input> inputs;
            inputs.reserve(last - first);
            for (std::size_t r = first; r < last; ++r) {
                backend.prepare(requests[rows[r]], packed[r - first]);
                inputs.push_back({&requests[rows[r]], packed[r - first]});
            }
            auto predictions = backend.evaluate(inputs);
            for (std::size_t r = first; r < last; ++r) out[rows[r]] = std::move(predictions[r - first]);
        }
    }
    return out;
}
}  // namespace

int main(int argc, char** argv) try {
    CLI::App app{"An interactive combat inspection session: JSON commands on stdin, one JSON response per line on "
                 "stdout (see inspection.hpp).", "combat_lab"};
    std::filesystem::path models;
    bool cpu = false;
    app.add_option("MODELS", models, "exported model directory")->required();
    app.add_flag("--cpu", cpu, "evaluate on the CPU only");
    CLI11_PARSE(app, argc, argv);
    auto network = nn::load_network(models, cpu ? nn::Backends::cpu : nn::Backends::all);
    Session session{network.encoding, [&](std::span<const infra::Tokens> requests) { return evaluate(network, requests); }};
    for (std::string line; std::getline(std::cin, line);) {
        if (line.empty()) continue;
        Json response;
        try {
            response = session.run(Json::parse(line));
        } catch (const std::exception& error) {
            response = {{"error", error.what()}};
        }
        std::cout << response.dump() << std::endl;
    }
    return 0;
} catch (const std::exception& error) {
    std::cerr << "combat_lab: " << error.what() << '\n';
    return 1;
}
