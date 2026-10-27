// Production TensorRT throughput, without simulator/search. See docs/INFERENCE_BENCHMARK.md.
#include "agents/combat/nn/net.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <mutex>
#include <thread>
#include <cuda_profiler_api.h>

using namespace stsrl::nn;
using Clock = std::chrono::steady_clock;
using Seconds = std::chrono::duration<double>;

struct Batch {
    std::vector<Tokens> tokens;
    std::vector<std::vector<std::uint16_t>> packed;
    explicit Batch(int n) : tokens(n), packed(n) {}
    std::vector<InferenceInput> inputs() const {
        std::vector<InferenceInput> out;
        out.reserve(tokens.size());
        for (std::size_t i = 0; i < tokens.size(); ++i) out.push_back({&tokens[i], packed[i]});
        return out;
    }
};

// Valid encoder schema/ranges, not a realistic combat-state distribution.
void prepare(Batch& batch, Evaluator& evaluator, int count, unsigned& rng) {
    auto random = [&] { rng = rng * 1664525u + 1013904223u; return rng; };
    for (auto& t : batch.tokens) {
        t.count = count;
        t.actions = std::min(4, count - 1);
        t.data.resize(std::size_t(count) * width);
        for (int row = 0; row < count; ++row) {
            auto* p = t.data.data() + row * width;
            p[0] = row == 0 ? 1 : row >= count - t.actions ? 6 : 2 + random() % 4;
            p[1] = random() % 128;
            p[2] = random() % 128;
            for (int j = 3; j < width; ++j) p[j] = float(random() % 1000) / 1000.f;
        }
    }
    for (std::size_t i = 0; i < batch.tokens.size(); ++i)
        evaluator.prepare(batch.tokens[i], batch.packed[i]);
}

void evaluate(Evaluator& evaluator, const Batch& batch) {
    const auto out = evaluator.submit_prepared(batch.inputs()).get();
    if (out.size() != batch.tokens.size()) throw std::runtime_error{"wrong prediction count"};
    for (std::size_t i = 0; i < out.size(); ++i) {
        if (!std::isfinite(out[i].value) || out[i].value < 0 || out[i].value > 1 ||
            out[i].logits.size() != std::size_t(batch.tokens[i].actions))
            throw std::runtime_error{"invalid prediction"};
        for (float x : out[i].logits) if (!std::isfinite(x)) throw std::runtime_error{"invalid logit"};
    }
}

int main(int argc, char** argv) try {
    if (argc < 3 || argc % 2 == 0) {
        std::cerr << "usage: nn_bench_inference CPU.onnx GPU.onnx [--batch 512] [--tokens 48] "
                     "[--state-tokens 32] [--seconds 3] [--queue 3] [--limit-seconds 55] [--profile-prepared 0]\n";
        return 2;
    }
    int batch_size = 512, tokens = 48, state_tokens = 32, depth = 3, limit = 55;
    double seconds = 3;
    bool profile_prepared = false;
    for (int i = 3; i < argc; i += 2) {
        const std::string key = argv[i];
        if (key == "--batch") batch_size = std::stoi(argv[i + 1]);
        else if (key == "--tokens") tokens = std::stoi(argv[i + 1]);
        else if (key == "--state-tokens") state_tokens = std::stoi(argv[i + 1]);
        else if (key == "--seconds") seconds = std::stod(argv[i + 1]);
        else if (key == "--queue") depth = std::stoi(argv[i + 1]);
        else if (key == "--limit-seconds") limit = std::stoi(argv[i + 1]);
        else if (key == "--profile-prepared") profile_prepared = std::stoi(argv[i + 1]) != 0;
        else throw std::invalid_argument{"unknown option: " + key};
    }
    if (batch_size < 1 || batch_size > 8192 || tokens < 2 || tokens > 512 ||
        state_tokens < 2 || state_tokens > tokens || depth < 1 || depth > 16 ||
        !std::isfinite(seconds) || seconds <= 0 || limit < 1 || limit > 60 || 2 * seconds + 2 >= limit)
        throw std::invalid_argument{"invalid sizes/duration; two measurements must fit within hard limit <=60s"};

    // Native engine setup/calls cannot be safely interrupted. Bound the entire process.
    std::mutex timer_mutex;
    std::condition_variable timer_wake;
    std::jthread watchdog{[&](std::stop_token stop) {
        std::unique_lock lock{timer_mutex};
        std::stop_callback notify{stop, [&] { timer_wake.notify_all(); }};
        if (!timer_wake.wait_for(lock, std::chrono::seconds(limit), [&] { return stop.stop_requested(); })) {
            std::cerr << "benchmark hard time limit reached (including setup)\n";
            std::_Exit(124);
        }
    }};
    const auto setup = Clock::now();
    auto evaluator = gpu_evaluator(argv[1], argv[2], batch_size, tokens, {}, 0, true);
    std::cout << "setup_seconds " << Seconds(Clock::now() - setup).count() << std::endl;
    unsigned rng = 1;
    Batch prepared(batch_size);
    prepare(prepared, *evaluator, state_tokens, rng);
    for (int i = 0; i < 3; ++i) evaluate(*evaluator, prepared);

    auto report = [&](const char* mode, long calls, double wall, double wait, double producer_wait) {
        std::cout << "{\"mode\":\"" << mode << "\",\"seconds\":" << wall
                  << ",\"batches\":" << calls << ",\"evaluations_per_second\":" << calls * batch_size / wall
                  << ",\"mean_batch_ms\":" << wall * 1000 / calls
                  << ",\"batch_fill\":1,\"batch\":" << batch_size << ",\"tokens\":" << tokens
                  << ",\"state_tokens\":" << state_tokens << ",\"consumer_wait_seconds\":" << wait
                  << ",\"producer_wait_seconds\":" << producer_wait << "}" << std::endl;
    };
    {
        if (profile_prepared && cudaProfilerStart() != cudaSuccess)
            throw std::runtime_error{"cudaProfilerStart failed"};
        long calls = 0;
        const auto start = Clock::now();
        do { evaluate(*evaluator, prepared); ++calls; } while (Seconds(Clock::now() - start).count() < seconds);
        report("prepared", calls, Seconds(Clock::now() - start).count(), 0, 0);
        if (profile_prepared && cudaProfilerStop() != cudaSuccess)
            throw std::runtime_error{"cudaProfilerStop failed"};
    }
    {
        std::mutex mutex;
        std::condition_variable wake;
        std::deque<std::unique_ptr<Batch>> free, ready;
        for (int i = 0; i < depth; ++i) free.push_back(std::make_unique<Batch>(batch_size));
        double producer_wait = 0;
        std::exception_ptr error;
        std::jthread producer{[&](std::stop_token stop) {
            std::stop_callback notify{stop, [&] { wake.notify_all(); }};
            try {
                while (!stop.stop_requested()) {
                    std::unique_ptr<Batch> batch;
                    {
                        const auto start = Clock::now();
                        std::unique_lock lock{mutex};
                        wake.wait(lock, [&] { return stop.stop_requested() || !free.empty(); });
                        producer_wait += Seconds(Clock::now() - start).count();
                        if (stop.stop_requested()) return;
                        batch = std::move(free.front()); free.pop_front();
                    }
                    prepare(*batch, *evaluator, state_tokens, rng);
                    { std::lock_guard lock{mutex}; ready.push_back(std::move(batch)); }
                    wake.notify_all();
                }
            } catch (...) {
                { std::lock_guard lock{mutex}; error = std::current_exception(); }
                wake.notify_all();
            }
        }};
        // Consumer starts with an empty queue: include initial producer preparation.
        long calls = 0;
        double consumer_wait = 0;
        const auto start = Clock::now();
        do {
            std::unique_ptr<Batch> batch;
            {
                const auto begin = Clock::now();
                std::unique_lock lock{mutex};
                wake.wait(lock, [&] { return error || !ready.empty(); });
                consumer_wait += Seconds(Clock::now() - begin).count();
                if (error) std::rethrow_exception(error);
                batch = std::move(ready.front()); ready.pop_front();
            }
            evaluate(*evaluator, *batch);
            ++calls;
            { std::lock_guard lock{mutex}; free.push_back(std::move(batch)); }
            wake.notify_all();
        } while (Seconds(Clock::now() - start).count() < seconds);
        const double wall = Seconds(Clock::now() - start).count();
        producer.request_stop(); producer.join();
        if (error) std::rethrow_exception(error);
        report("cpu_fed", calls, wall, consumer_wait, producer_wait);
    }
    return 0;
} catch (const std::exception& e) {
    std::cerr << "benchmark: " << e.what() << '\n';
    return 1;
}
