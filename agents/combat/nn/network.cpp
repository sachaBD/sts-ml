#include "agents/combat/nn/network.hpp"
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

namespace stsrl::nn {
Network load_network(const std::filesystem::path& models) {
    std::ifstream file{models / "network.json"};
    if (!file) throw std::invalid_argument{"no network.json in " + models.string()};
    const auto manifest = nlohmann::json::parse(file);
    Network network{encoding_from_name(manifest.at("encoding").get<std::string>()), {}};
    for (const auto& deployment : manifest.at("deployments")) {
        std::vector<std::filesystem::path> files;
        for (const auto& name : deployment.at("files")) files.push_back(models / name.get<std::string>());
        network.backends.push_back(infra::gpu_backend(files, deployment.at("lanes").get<int>()));
    }
    if (network.backends.empty()) throw std::invalid_argument{"network.json has no deployments"};
    const auto& cpu = manifest.at("cpu");
    network.backends.push_back(infra::cpu_backend(models / cpu.at("file").get<std::string>(),
                                                  cpu.at("lanes").get<int>(), cpu.at("batch").get<int>()));
    for (const auto& backend : network.backends)
        if (backend->width() != width) throw std::invalid_argument{backend->name() + " model token width does not match the encoder"};
    return network;
}
}  // namespace stsrl::nn
