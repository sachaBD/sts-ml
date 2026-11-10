// GPU backend throughput at the supplied model's fixed batch size.
#include "infra/inference/backend.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>

using namespace stsrl::infra;
using Clock = std::chrono::steady_clock;

namespace {
Tokens read_tokens(const std::filesystem::path& path, int width, int actions) {
    const auto bytes = std::filesystem::file_size(path);
    const auto stride = sizeof(float) * width;
    if (!bytes || bytes % stride || bytes / stride > std::numeric_limits<int>::max())
        throw std::invalid_argument{"input must contain complete float32 token rows"};
    Tokens tokens;
    tokens.count = int(bytes / stride);
    if (actions <= 0 || actions > tokens.count) throw std::invalid_argument{"invalid input action count"};
    tokens.actions = actions;
    tokens.attended = tokens.count;
    tokens.data.resize(bytes / sizeof(float));
    std::ifstream input{path, std::ios::binary};
    input.read(reinterpret_cast<char*>(tokens.data.data()), std::streamsize(bytes));
    if (!input) throw std::runtime_error{"could not read input tokens"};
    return tokens;
}

std::vector<Tokens> synthetic_states(int batch, int width) {
    if (width < 3) throw std::invalid_argument{"synthetic inputs need at least three fields; use --input for other schemas"};
    // Legacy schema: context first, four actions last, bounded synthetic features.
    constexpr int count = 37, actions = 4;
    std::mt19937 rng{1};
    std::vector<Tokens> states(batch);
    for (auto& tokens : states) {
        tokens.count = count;
        tokens.actions = actions;
        tokens.data.resize(std::size_t(count) * width);
        for (int row = 0; row < count; ++row) {
            float* p = tokens.data.data() + std::size_t(row) * width;
            p[0] = row == 0 ? 1 : row >= count - actions ? 6 : 2 + rng() % 4;
            p[1] = rng() % 128;
            p[2] = rng() % 128;
            for (int j = 3; j < width; ++j) p[j] = float(rng() % 1000) / 1000;
        }
    }
    return states;
}
}  // namespace

int main(int argc, char** argv) try {
    constexpr auto usage = "usage: infra_bench_inference MODEL.onnx MODEL-GPU.onnx [SECONDS] [--input INPUT.f32 ACTIONS]";
    if (argc < 3) throw std::invalid_argument{usage};
    int next = 3;
    double seconds = 5;
    if (next < argc && std::string_view{argv[next]} != "--input") seconds = std::stod(argv[next++]);
    if (!std::isfinite(seconds) || seconds <= 0) throw std::invalid_argument{"positive finite duration required"};
    const char* input_path = nullptr;
    int actions = 0;
    if (next < argc) {
        if (std::string_view{argv[next]} != "--input" || argc - next != 3) throw std::invalid_argument{usage};
        input_path = argv[next + 1];
        actions = std::stoi(argv[next + 2]);
    }

    auto gpu = gpu_backend(argv[1], {argv[2]});
    const int width = gpu->width(), batch = gpu->max_batch();
    const auto states = input_path ? std::vector<Tokens>(batch, read_tokens(input_path, width, actions))
                                   : synthetic_states(batch, width);
    std::vector<std::vector<std::uint16_t>> packed(batch);
    std::vector<Input> inputs;
    for (int b = 0; b < batch; ++b) {
        if (!gpu->accepts(states[b])) throw std::invalid_argument{"input exceeds model capacity"};
        gpu->prepare(states[b], packed[b]);
        inputs.push_back({&states[b], packed[b]});
    }
    for (const auto& prediction : gpu->evaluate(inputs)) {
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
