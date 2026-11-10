#pragma once
#include "agents/combat/nn/evaluator.hpp"
#include "agents/combat/search/search.hpp"
#include "infra/workers/actors.hpp"
#include <atomic>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <ostream>
#include <vector>

namespace stsrl::player {
// Fights as infra episodes. Each finished fight appends its samples to the sample file and writes one result
// line, under one lock, so a result's sample range is exactly its fight's samples.
class Episodes {
public:
    Episodes(std::vector<nlohmann::json> fights, search::SearchFactory, nn::NetworkEvaluator,
             const std::filesystem::path& samples, std::ostream& results);
    std::unique_ptr<infra::Episode> next();  // thread-safe; nullptr once every fight has started
    std::size_t completed() const { return completed_; }
    std::size_t decisions() const { return decisions_; }

private:
    struct Game;
    class Episode;
    void finish(const Game&);

    const std::vector<nlohmann::json> fights_;
    const search::SearchFactory search_;
    const nn::NetworkEvaluator network_;
    std::atomic<std::size_t> next_{0}, completed_{0}, decisions_{0};
    std::mutex output_;
    std::ofstream samples_;
    const std::string sample_file_;
    std::ostream& results_;
};
}  // namespace stsrl::player
