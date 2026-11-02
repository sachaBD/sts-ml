#pragma once
// App-owned serialization: infra has no JSON, filesystem or dashboard dependency.
#include "infra/workers/actors.hpp"
#include <fstream>
#include <nlohmann/json.hpp>
#include <unistd.h>

namespace stsrl::nn {
inline nlohmann::json runtime_json(const infra::RuntimeSnapshot& s, const std::string& instance, int pid) {
    using Json = nlohmann::json;
    Json out{{"schema_version", 1}, {"worker_instance_id", instance}, {"pid", pid},
             {"sequence", s.sequence}, {"captured_at", s.captured_unix_seconds}, {"elapsed_seconds", s.elapsed_seconds}};
    for (const auto* key : {"actors", "queues", "dispatchers", "backends"}) out[key] = Json::array();
    for (const auto& a : s.actors) out["actors"].push_back({
        {"id", a.id}, {"slots", a.slots}, {"requests", a.requests}, {"completed", a.completed},
        {"slot_seconds", {{"ready", a.slot_seconds[0]}, {"awaiting_inference", a.slot_seconds[1]}, {"empty", a.slot_seconds[2]}}},
        {"working_seconds", a.working_seconds}});
    for (const auto& q : s.queues) out["queues"].push_back({
        {"id", q.id}, {"depth", q.depth}, {"peak_depth_interval", q.peak_depth},
        {"depth_seconds", q.depth_seconds}, {"oldest_seconds", q.oldest_seconds},
        {"wait_histogram", {{"upper_bounds_seconds", infra::queue_wait_bounds_seconds}, {"counts", q.wait_counts}}}});
    for (const auto& d : s.dispatchers) out["dispatchers"].push_back({
        {"id", d.id}, {"backend_id", d.backend_id}, {"requests", d.requests}, {"batches", d.batches}, {"capacity_rows", d.capacity_rows}});
    for (const auto& b : s.backends) {
        Json backend{{"id", b.id}, {"kind", b.kind}, {"max_batch", b.max_batch}, {"tokens", b.tokens},
                     {"attended", b.attended}, {"oversized_rows", b.oversized_rows}, {"precision_retries", b.precision_retries},
                     {"lanes", Json::array()}};
        for (const auto& lane : b.lanes) backend["lanes"].push_back({
            {"id", lane.id}, {"rows", lane.rows}, {"batches", lane.batches}, {"capacity_rows", lane.capacity_rows},
            {"time_seconds", {{"idle", lane.time_seconds[0]}, {"packing", lane.time_seconds[1]}, {"h2d", lane.time_seconds[2]},
                              {"inference", lane.time_seconds[3]}, {"d2h", lane.time_seconds[4]}, {"cpu_retry", lane.time_seconds[5]}}}});
        out["backends"].push_back(std::move(backend));
    }
    return out;
}

inline infra::RuntimeObserver runtime_observer(const char* destination) {
    if (!destination || !*destination) return {};
    const std::filesystem::path path{destination};
    const int pid = int(::getpid());
    const std::string instance = std::to_string(pid) + ":" + std::to_string(infra::TelemetryClock::now().time_since_epoch().count());
    return [path, instance, pid](const infra::RuntimeSnapshot& snapshot) {
        auto temporary = path; temporary += ".tmp." + std::to_string(pid);
        {
            std::ofstream stream;
            stream.exceptions(std::ios::badbit | std::ios::failbit);
            stream.open(temporary, std::ios::binary | std::ios::trunc);
            stream << runtime_json(snapshot, instance, pid).dump() << '\n';
            stream.close();
        }
        std::filesystem::rename(temporary, path);
    };
}
} // namespace stsrl::nn
