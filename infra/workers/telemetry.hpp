#pragma once
#include "infra/workers/actors.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace stsrl::infra {
// Each run_actors Counters as one JSON line; infra/workers/telemetry.py reads them.
class TelemetryFile {
public:
    explicit TelemetryFile(const std::filesystem::path& path) : out_{path} {
        if (!out_) throw std::runtime_error{"cannot write telemetry " + path.string()};
    }

    void write(const infra::Counters& counters) {
        nlohmann::json actors = nlohmann::json::array();
        for (const auto& a : counters.actors) actors.push_back({{"requests", a.requests}, {"blocked_seconds", a.blocked.count()}});
        nlohmann::json queues = nlohmann::json::array();
        for (const auto& q : counters.queues)
            queues.push_back({{"backend", q.backend}, {"requests", q.requests}, {"waited_seconds", q.waited.count()},
                              {"depth", q.depth}, {"peak_depth", q.peak_depth}});
        nlohmann::json lanes = nlohmann::json::array();
        for (const auto& l : counters.lanes)
            lanes.push_back({{"queue", l.queue}, {"requests", l.requests}, {"capacity", l.capacity}, {"busy_seconds", l.busy.count()}});
        const std::chrono::duration<double> time = counters.time.time_since_epoch();
        out_ << nlohmann::json{{"time", time.count()}, {"actors", actors}, {"queues", queues}, {"lanes", lanes}}.dump() << std::endl;
    }

private:
    std::ofstream out_;
};
}  // namespace stsrl::infra
