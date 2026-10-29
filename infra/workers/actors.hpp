#pragma once
// Actors play episodes; one InferenceServer batches their network requests onto a Backend.
// Episodes never leave their Actor: only immutable token views and predictions cross threads.
#include "infra/inference/backend.hpp"
#include <chrono>
#include <functional>
#include <memory>
#include <string>

namespace stsrl::infra {
class Episode {
public:
    virtual ~Episode() = default;
    // Play on until the network is needed (tokens stay immutable until evaluated()) or the episode is over (nullptr).
    virtual const Tokens* advance() = 0;
    virtual void evaluated(const Prediction&) = 0;
    virtual void finish() = 0;                // record the result; called once, on the owning Actor
    virtual std::string label() const = 0;    // names the episode in errors
};

struct ActorConfig {
    int actors = 9;
    int slots = 224;                           // episodes in flight per Actor
    int batch = 512;                           // most requests per InferenceServer batch
    std::chrono::microseconds batch_wait{100}; // how long a partial batch waits for more requests
    std::chrono::milliseconds timeout{60'000}; // an Actor, request or the server stalled this long is an error
    std::chrono::milliseconds shutdown_timeout{5'000};
};

// Thread-safe source of episodes; nullptr when there are none left. Runs on the Actor that will own the episode.
using NextEpisode = std::function<std::unique_ptr<Episode>()>;

// Runs the Actors and the InferenceServer until every episode has finished; the calling thread supervises.
// Throws the first failure; a timeout is a failure, never a lost episode. If a native call cannot be stopped
// within shutdown_timeout the process exits nonzero rather than hang or free state still in use.
void run_actors(const ActorConfig&, NextEpisode, Backend&);
}  // namespace stsrl::infra
