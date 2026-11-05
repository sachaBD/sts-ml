// GPU backend throughput at the supplied model's fixed batch size.
// Optional raw float32 tokens let callers benchmark their own input schema without teaching infra about it.
#include "infra/inference/backend.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>

using namespace stsrl::infra;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) try {
    if (argc < 3 || argc > 6 || argc == 5) {
        std::cerr << "usage: infra_bench_inference MODEL.onnx MODEL-GPU.onnx [SECONDS [INPUT.f32 ACTIONS]]\n";
        return 2;
    }
    const double seconds = argc > 3 ? std::stod(argv[3]) : 5;
    if (!std::isfinite(seconds) || seconds <= 0) throw std::invalid_argument{"positive finite duration required"};
    auto gpu = gpu_backend(argv[1], {argv[2]});
    const int width = gpu->width(), batch = gpu->max_batch();
    Tokens supplied;
    if (argc == 6) {
        const auto bytes = std::filesystem::file_size(argv[4]);
        const auto stride = sizeof(float) * width;
        if (!bytes || bytes % stride || bytes / stride > std::numeric_limits<int>::max())
            throw std::invalid_argument{"input must contain complete float32 token rows"};
        supplied.count = int(bytes / stride);
        supplied.actions = std::stoi(argv[5]);
        supplied.attended = supplied.count;
        if (supplied.actions <= 0 || supplied.actions > supplied.count || !gpu->accepts(supplied))
            throw std::invalid_argument{"input actions or token count exceed model capacity"};
        supplied.data.resize(bytes / sizeof(float));
        std::ifstream input{argv[4], std::ios::binary};
        input.read(reinterpret_cast<char*>(supplied.data.data()), std::streamsize(bytes));
        if (!input) throw std::runtime_error{"could not read input tokens"};
    }

    // Legacy default: bounded synthetic tokens, context first and four actions last.
    std::mt19937 rng{1};
    std::vector<Tokens> states(batch);
    std::vector<std::vector<std::uint16_t>> packed(batch);
    std::vector<Input> inputs;
    for (int b = 0; b < batch; ++b) {
        auto& t = states[b];
        if (argc == 6) {
            t = supplied;
        } else {
            t.count = 37; t.actions = 4; t.data.resize(std::size_t(t.count) * width);
            for (int row = 0; row < t.count; ++row) {
                float* p = t.data.data() + std::size_t(row) * width;
                p[0] = row == 0 ? 1 : row >= t.count - t.actions ? 6 : 2 + rng() % 4;
                p[1] = rng() % 128; p[2] = rng() % 128;
                for (int j = 3; j < width; ++j) p[j] = float(rng() % 1000) / 1000;
            }
        }
        gpu->prepare(t, packed[b]);
        inputs.push_back({&t, packed[b]});
    }
    for (int warm = 0; warm < 3; ++warm) {
        const auto predictions = gpu->evaluate(inputs);
        if (predictions.size() != inputs.size()) throw std::runtime_error{"prediction count mismatch"};
        for (const auto& prediction : predictions)
            if (!std::isfinite(prediction.value) || prediction.value < 0 || prediction.value > 1 ||
                !std::all_of(prediction.logits.begin(), prediction.logits.end(), [](float x) { return std::isfinite(x); }))
                throw std::runtime_error{"invalid prediction; check model/input encoding"};
    }

    long calls = 0;
    const auto start = Clock::now();
    double elapsed = 0;
    for (; elapsed < seconds; ++calls) {
        gpu->evaluate(inputs);
        elapsed = std::chrono::duration<double>(Clock::now() - start).count();
    }
    std::cout << "{\"evals_per_second\":" << calls * batch / elapsed
              << ",\"batch\":" << batch << ",\"state_tokens\":" << states.front().count
              << ",\"batches\":" << calls << ",\"seconds\":" << elapsed
              << ",\"batch_ms\":" << elapsed * 1000 / calls << "}" << std::endl;
    return 0;
} catch (const std::exception& e) {
    std::cerr << "infra_bench_inference: " << e.what() << '\n';
    return 1;
}
