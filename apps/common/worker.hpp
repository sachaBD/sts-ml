// The command line and result file of every app worker; apps/common/worker.py runs them:
//   WORKER REQUEST.json OUTPUT_DIR [WEIGHTS]
// play(request, net) -> OUTPUT_DIR/result.msgpack; net is the value net from WEIGHTS, or null without it.
#pragma once

#include "models/value_net.hpp"

#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

namespace stsrl::worker {

using PlayFn = std::function<nlohmann::json(const nlohmann::json& request, const ValueNet* net)>;

inline void write_result(const std::filesystem::path& directory, const nlohmann::json& result) {
    std::filesystem::create_directories(directory);
    const auto bytes = nlohmann::json::to_msgpack(result);
    const auto path = directory / "result.msgpack";
    std::ofstream output{path, std::ios::binary};
    if (!output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size())))
        throw std::runtime_error{"failed to write " + path.string()};
    output.close();
    if (!output) throw std::runtime_error{"failed to close " + path.string()};
}

// A worker's main(): exit 0 on success, 1 on an error (message on stderr), 2 on bad usage.
inline int main(int argc, char* argv[], std::string_view name, const PlayFn& play) {
    if (argc != 3 && argc != 4) {
        std::cerr << "usage: " << name << " REQUEST.json OUTPUT_DIR [WEIGHTS]\n";
        return 2;
    }
    try {
        std::optional<ValueNet> net;
        if (argc == 4) net.emplace(argv[3]);
        std::ifstream file{argv[1]};
        if (!file) throw std::runtime_error{std::string{"cannot read "} + argv[1]};
        write_result(argv[2], play(nlohmann::json::parse(file), net ? &*net : nullptr));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << name << ": " << error.what() << '\n';
        return 1;
    }
}

}  // namespace stsrl::worker
