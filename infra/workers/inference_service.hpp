#pragma once
#include "infra/inference/backend.hpp"
#include <chrono>
#include <compare>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace stsrl::infra {
struct EvaluationRequest {
    std::uint64_t id;
    Tokens tokens;
};

struct SlotId {
    std::size_t value;
    auto operator<=>(const SlotId&) const = default;
};

struct Completion {
    SlotId slot;
    std::uint64_t id;
    Prediction prediction;
};

struct QueueCounters {
    long requests = 0;
    std::chrono::duration<double> waited{};
    std::size_t depth = 0, peak_depth = 0;
};
struct LaneCounters {
    int queue = 0;
    long requests = 0, capacity = 0;
    std::chrono::duration<double> busy{};
};
struct InferenceCounters {
    std::vector<QueueCounters> queues;
    std::vector<LaneCounters> lanes;
};

class InferenceService {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    // Owned by one actor. Payloads stay here until their completions arrive; inference only borrows views.
    // Connections must outlive lane calls that may still access their payloads, including on failure.
    class Connection {
        struct Impl;
        std::unique_ptr<Impl> impl_;
        explicit Connection(InferenceService&);
        friend class InferenceService;
    public:
        ~Connection();
        Connection(const Connection&) = delete;
        Connection& operator=(const Connection&) = delete;
        void submit(SlotId, EvaluationRequest, std::string label);
        std::vector<Completion> receive();
        bool pending(SlotId) const;
        std::optional<SlotId> timed_out(std::chrono::milliseconds) const;
        void wait(std::chrono::milliseconds);
    };

    InferenceService(const std::vector<Backend*>&, int batch, std::chrono::microseconds batch_wait);
    ~InferenceService();
    InferenceService(const InferenceService&) = delete;
    InferenceService& operator=(const InferenceService&) = delete;
    Connection connect();
    int lanes() const;
    void run(int lane);  // one caller thread per lane; returns on close/drain or stop
    void close();       // reject new submissions and drain accepted requests
    void stop();        // stop accepting/dispatching; in-progress backend calls must still return
    struct LaneStatus { long long heartbeat_ms; bool done; };
    LaneStatus status(int lane) const;
    InferenceCounters counters() const;
    struct Totals { long batches, requests; double cpu_seconds; };
    Totals totals() const;  // after lane threads have joined
};
}  // namespace stsrl::infra
