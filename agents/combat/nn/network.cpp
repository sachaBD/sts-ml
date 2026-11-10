#include "agents/combat/nn/network.hpp"
#include <stdexcept>
#include <string>
#include <toml++/toml.hpp>

namespace stsrl::nn {
namespace {
template <class T> T required(toml::node_view<const toml::node> node, const std::string& what) {
    if (const auto value = node.value<T>()) return *value;
    throw std::invalid_argument{"network.toml: missing or mistyped " + what};
}

const toml::array& array(toml::node_view<const toml::node> node, const std::string& what) {
    if (const auto* value = node.as_array()) return *value;
    throw std::invalid_argument{"network.toml: missing or mistyped " + what};
}
}  // namespace

Network load_network(const std::filesystem::path& models) {
    const auto path = models / "network.toml";
    if (!std::filesystem::exists(path)) throw std::invalid_argument{"no network.toml in " + models.string()};
    const toml::table parsed = toml::parse_file(path.string());
    const toml::node_view<const toml::node> manifest{parsed};
    Network network{encoding_from_name(required<std::string>(manifest["encoding"], "encoding")), {}};
    for (const auto& deployment : array(manifest["deployments"], "deployments")) {
        const toml::node_view<const toml::node> d{deployment};
        std::vector<std::filesystem::path> files;
        for (const auto& file : array(d["files"], "deployment files"))
            files.push_back(models / required<std::string>(toml::node_view<const toml::node>{file}, "deployment file"));
        network.backends.push_back(infra::gpu_backend(files, required<int>(d["lanes"], "deployment lanes")));
    }
    if (network.backends.empty()) throw std::invalid_argument{"network.toml has no deployments"};
    const auto cpu = manifest["cpu"];
    network.backends.push_back(infra::cpu_backend(models / required<std::string>(cpu["file"], "cpu file"),
                                                  required<int>(cpu["lanes"], "cpu lanes"),
                                                  required<int>(cpu["batch"], "cpu batch")));
    for (const auto& backend : network.backends)
        if (backend->width() != width) throw std::invalid_argument{backend->name() + " model token width does not match the encoder"};
    return network;
}
}  // namespace stsrl::nn
