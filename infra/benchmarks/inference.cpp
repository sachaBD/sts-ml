// GPU backend throughput at our batch size (512).
// infra_bench_inference MODEL.onnx MODEL-GPU.onnx [SECONDS]  ->  one JSON line on stdout
// The GPU model has a fixed [512, tokens, width] shape, so its cost does not depend on the states' contents.
#include "infra/inference/backend.hpp"
#include <chrono>
#include <iostream>
#include <random>

using namespace stsrl::infra;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) try {
    if (argc < 3) { std::cerr << "usage: infra_bench_inference MODEL.onnx MODEL-GPU.onnx [SECONDS]\n"; return 2; }
    const double seconds = argc > 3 ? std::stod(argv[3]) : 5;
    auto gpu = gpu_backend(argv[1], {argv[2]});

    // Schema-valid synthetic states: context token first, 4 legal actions (type 6) last, 37 tokens (our mean).
    const int width = gpu->width(), batch = gpu->max_batch(), count = 37, actions = 4;
    std::mt19937 rng{1};
    std::vector<Tokens> states(batch);
    std::vector<std::vector<std::uint16_t>> packed(batch);
    std::vector<Input> inputs;
    for (int b = 0; b < batch; ++b) {
        auto& t = states[b];
        t.count = count; t.actions = actions; t.data.resize(std::size_t(count) * width);
        for (int row = 0; row < count; ++row) {
            float* p = t.data.data() + std::size_t(row) * width;
            p[0] = row == 0 ? 1 : row >= count - actions ? 6 : 2 + rng() % 4;
            p[1] = rng() % 128; p[2] = rng() % 128;
            for (int j = 3; j < width; ++j) p[j] = float(rng() % 1000) / 1000;
        }
        gpu->prepare(t, packed[b]);
        inputs.push_back({&t, packed[b]});
    }

    long calls = 0;
    const auto start = Clock::now();
    double elapsed = 0;
    for (; elapsed < seconds; ++calls) {
        gpu->evaluate(inputs);
        elapsed = std::chrono::duration<double>(Clock::now() - start).count();
    }
    std::cout << "{\"evals_per_second\":" << calls * batch / elapsed << "}" << std::endl;
    return 0;
} catch (const std::exception& e) {
    std::cerr << "infra_bench_inference: " << e.what() << '\n';
    return 1;
}
