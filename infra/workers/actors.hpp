#pragma once
// Actors play episodes; InferenceServer lanes batch their network requests onto a Backend.
// Episodes and their lifecycle methods stay on their Actor; only immutable token views and predictions cross threads.
// On failure, episode/request storage stays alive until inference has stopped accessing it.
#include "infra/inference/backend.hpp"
#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace stsrl::infra {
struct EvaluationRequest {
    std::uint64_t id;
    const Tokens* tokens;
};

class Episode {
public:
    virtual ~Episode() = default;
    // Claims one request. Null means blocked or done(); claiming never returns the same outstanding request twice.
    // The episode bounds outstanding work and owns immutable tokens until completed(id).
    // Issued IDs strictly increase within an episode, including across completions. Completion order is unrestricted.
    virtual std::optional<EvaluationRequest> next_request() = 0;
    virtual void completed(std::uint64_t id, const Prediction&) = 0;
    // True only when no requests remain outstanding and no further work can be produced.
    virtual bool done() const = 0;
    virtual void finish() = 0;                // record the result; called once, on the owning Actor
    virtual std::string label() const = 0;    // names the episode in errors
};

// Counters since run_actors began; queue depth is sampled at publication.
struct ActorCounters {
    long requests = 0;                        // inference requests submitted
    std::chrono::duration<double> blocked{};  // waiting for inference replies
};
struct QueueCounters {
    long requests = 0;                       // requests that left in a batch
    std::chrono::duration<double> waited{};  // their total time from submit to leaving
    std::size_t depth = 0;                  // pending requests, excluding gathered batches
    std::size_t peak_depth = 0;             // largest pending count since run_actors began
};
struct LaneCounters {
    int queue = 0;                         // index into Counters::queues
    long requests = 0;                     // requests evaluated
    long capacity = 0;                     // batch cap summed over batches
    std::chrono::duration<double> busy{};  // evaluating and replying
};
struct Counters {
    std::chrono::system_clock::time_point time;  // when captured
    std::vector<ActorCounters> actors;
    std::vector<QueueCounters> queues;           // one per backend, in order
    std::vector<LaneCounters> lanes;             // one per InferenceServer
};

struct ActorConfig {
    int actors = 9;
    int slots = 224;                           // episodes in flight per Actor
    int batch = 512;                           // most requests per InferenceServer batch
    std::chrono::microseconds batch_wait{100}; // how long a partial batch waits for more requests
    std::chrono::milliseconds timeout{60'000}; // an Actor, request or the server stalled this long is an error
    std::chrono::milliseconds shutdown_timeout{5'000};
    // Called on the supervising thread about once a second and once at the end; must return promptly and not throw.
    std::function<void(const Counters&)> on_counters;
};

// Thread-safe source of episodes; nullptr when there are none left. Runs on the Actor that will own the episode.
using NextEpisode = std::function<std::unique_ptr<Episode>()>;

// Runs the Actors and the InferenceServer lanes until every episode has finished; the calling thread supervises.
// Throws the first failure; a timeout is a failure, never a lost episode. If a native call cannot be stopped
// within shutdown_timeout the process exits nonzero rather than hang or free state still in use.
// First accepting backend wins; the last handles requests none accept. Backends outlive this call.
void run_actors(const ActorConfig&, NextEpisode, const std::vector<Backend*>&);
void run_actors(const ActorConfig&, NextEpisode, Backend&);
}  // namespace stsrl::infra
