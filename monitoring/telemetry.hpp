#pragma once
#include "infra/workers/actors.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace stsrl::monitoring {
// Each run_actors Counters as one JSON line; monitoring/telemetry.py reads them.
class TelemetryFile {
public:
    explicit TelemetryFile(const std::filesystem::path& path) : out_{path} {
        if (!out_) throw std::runtime_error{"cannot write telemetry " + path.string()};
    }

    void write(const infra::Counters& counters) {
        nlohmann::json actors = nlohmann::json::array();
        for (const auto& a : counters.actors) actors.push_back({{"requests", a.requests}, {"blocked_seconds", a.blocked.count()}});
        const std::chrono::duration<double> time = counters.time.time_since_epoch();
        out_ << nlohmann::json{{"time", time.count()}, {"actors", actors}}.dump() << std::endl;
    }

private:
    std::ofstream out_;
};
}  // namespace stsrl::monitoring
