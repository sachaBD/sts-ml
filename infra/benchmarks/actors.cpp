// Actor + InferenceServer routing capacity: free episodes and a free backend, so only the plumbing costs anything.
// Production shape: 9 Actors x 224 slots, batches of up to 512, 100 us batch wait.
// infra_bench_actors [SECONDS]  ->  one JSON line on stdout
#include "infra/workers/actors.hpp"
#include <chrono>
#include <iostream>

using namespace stsrl::infra;
using Clock = std::chrono::steady_clock;

namespace {
constexpr int token_width = 68, count = 37, actions = 4, steps = 100;

class FreeEpisode final : public Episode {
    Tokens tokens_{std::vector<float>(count * token_width, 0.f), count, actions};
    int left_ = steps;
public:
    const Tokens* advance() override { return left_ ? &tokens_ : nullptr; }
    void evaluated(const Prediction&) override { --left_; }
    void finish() override {}
    std::string label() const override { return "free"; }
};

// Counts only until `stop`, so the drain of in-flight episodes at the end is not measured.
class FreeBackend final : public Backend {
public:
    explicit FreeBackend(Clock::time_point stop) : stop_{stop} {}
    long batches = 0, requests = 0;
    int width() const override { return token_width; }
    int max_batch() const override { return 512; }
    std::vector<Prediction> evaluate(std::span<const Input> inputs) override {
        if (Clock::now() < stop_) { ++batches; requests += long(inputs.size()); }
        return std::vector<Prediction>(inputs.size(), Prediction{0.5f, std::vector<float>(actions, 0.f)});
    }
private:
    Clock::time_point stop_;
};
}  // namespace

int main(int argc, char** argv) try {
    const double seconds = argc > 1 ? std::stod(argv[1]) : 5;
    const auto stop = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
    FreeBackend backend{stop};
    run_actors(ActorConfig{}, [&]() -> std::unique_ptr<Episode> {
        return Clock::now() < stop ? std::make_unique<FreeEpisode>() : nullptr;
    }, backend);
    std::cout << "{\"requests_per_second\":" << backend.requests / seconds
              << ",\"mean_batch\":" << double(backend.requests) / backend.batches << "}" << std::endl;
    return 0;
} catch (const std::exception& e) {
    std::cerr << "infra_bench_actors: " << e.what() << '\n';
    return 1;
}
