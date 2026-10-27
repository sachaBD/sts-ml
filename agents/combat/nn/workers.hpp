#pragma once
// Worker-local simulations. Only immutable token views and predictions cross threads.
#include "agents/combat/nn/net.hpp"
#include <chrono>
#include <functional>
#include <string>

namespace stsrl::nn {
class Simulation {
public:
    virtual ~Simulation() = default;
    // nullptr means finished. Otherwise the returned tokens stay immutable until evaluated().
    virtual const Tokens* advance() = 0;
    virtual void evaluated(const Prediction&) = 0;
    virtual void finish() = 0;
    virtual std::string label() const = 0;
};

struct WorkerConfig {
    int workers = 9;
    int slots = 224;              // per worker; also bounds each request/completion queue
    int batch = 512;
    std::chrono::microseconds batch_wait{100};
    std::chrono::milliseconds timeout{60'000};
    std::chrono::milliseconds shutdown_timeout{5'000};
};

using NextSimulation = std::function<std::unique_ptr<Simulation>()>; // thread-safe shared start queue
using Evaluate = std::function<std::vector<Prediction>(std::vector<const Tokens*>)>; // synchronous
using Prepare = std::function<void(const Tokens&, std::vector<std::uint16_t>&)>;
using PreparedEvaluate = std::function<std::vector<Prediction>(std::vector<InferenceInput>)>;

// Creates N SimWorkers + one Coordinator. The calling thread supervises timeouts.
// Throws on failure; outstanding token owners survive until the coordinator exits.
// If native inference/simulation cannot stop within shutdown_timeout, exits nonzero rather
// than hanging forever or freeing state still in use. No timeout is turned into a game loss.
void run_workers(const WorkerConfig&, NextSimulation, Evaluate);
void run_workers(const WorkerConfig&, NextSimulation, PreparedEvaluate, Prepare);
}  // namespace stsrl::nn
