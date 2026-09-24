// Result writing shared by the fight workers (apps/bootstrap/fight_worker.cpp, apps/value_play/worker.cpp).
#pragma once

#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace stsrl::worker {

// directory/fight.msgpack
inline void write_result(const std::filesystem::path& directory, const nlohmann::json& result) {
    std::filesystem::create_directories(directory);
    const auto bytes = nlohmann::json::to_msgpack(result);
    const auto path = directory / "fight.msgpack";
    std::ofstream output{path, std::ios::binary};
    if (!output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size())))
        throw std::runtime_error{"failed to write " + path.string()};
    output.close();
    if (!output) throw std::runtime_error{"failed to close " + path.string()};
}

}  // namespace stsrl::worker
