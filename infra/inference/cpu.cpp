// ONNX Runtime on the CPU: dynamic batch and token count, one intra-op thread per lane.
#include "infra/inference/cpu.hpp"
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <numeric>
#include <stdexcept>

namespace stsrl::infra {
namespace {
using Clock = std::chrono::steady_clock;
double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }
// B*T*T attention cells per call: one unusually long row must not pad a whole batch to its length.
constexpr std::size_t attention_cells = 2048 * 64 * 64;
}

struct CpuBackend::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "infra-cpu"};
    Ort::Session session{nullptr};  // Run is safe to call concurrently, so lanes share it
    int width = 0;
    // Summed over lanes, so busy can exceed wall time while lanes overlap.
    std::mutex stats_mutex;
    long batches = 0, leaves = 0;
    double busy = 0, reported = 0;
    const Clock::time_point begin = Clock::now();

    void print(int lanes) const {  // caller holds stats_mutex
        std::fprintf(stderr, "cpu_stats: {\"seconds\":%.3f,\"lanes\":%d,\"batches\":%ld,\"leaves\":%ld,\"busy\":%.3f}\n",
                     since(begin), lanes, batches, leaves, busy);
    }

    std::vector<Prediction> run(std::span<const Tokens* const> rows) {
        int longest = 0;
        for (const auto* t : rows) longest = std::max(longest, t->count);
        std::vector<float> input(rows.size() * longest * width, 0.f);  // all-zero token = padding
        for (std::size_t b = 0; b < rows.size(); ++b)
            std::copy(rows[b]->data.begin(), rows[b]->data.end(), input.begin() + b * longest * width);
        const std::array<std::int64_t, 3> shape{std::int64_t(rows.size()), longest, width};
        const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto tensor = Ort::Value::CreateTensor<float>(memory, input.data(), input.size(), shape.data(), shape.size());
        constexpr std::array<const char*, 1> inputs{"tokens"};
        constexpr std::array<const char*, 2> outputs{"value", "logits"};
        auto result = session.Run(Ort::RunOptions{nullptr}, inputs.data(), &tensor, 1, outputs.data(), 2);
        const float* value = result[0].GetTensorData<float>();
        const float* logits = result[1].GetTensorData<float>();
        std::vector<Prediction> out;
        out.reserve(rows.size());
        for (std::size_t b = 0; b < rows.size(); ++b) {
            const float* row = logits + b * longest;
            out.push_back({value[b], {row + rows[b]->count - rows[b]->actions, row + rows[b]->count}});
        }
        return out;
    }
};

CpuBackend::CpuBackend(const std::filesystem::path& onnx, int lanes) : impl_{std::make_unique<Impl>()}, lanes_{lanes} {
    if (lanes < 1) throw std::invalid_argument{"cpu backend: positive lanes required"};
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    impl_->session = Ort::Session{impl_->env, onnx.c_str(), options};
    const auto type = impl_->session.GetInputTypeInfo(0);
    const auto shape = type.GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() != 3 || shape[2] <= 0) throw std::invalid_argument{"cpu backend: input must be [batch, tokens, width]"};
    impl_->width = int(shape[2]);
}
CpuBackend::~CpuBackend() {
    std::lock_guard lock{impl_->stats_mutex};
    if (impl_->batches) impl_->print(lanes_);
}

int CpuBackend::width() const { return impl_->width; }

std::vector<Prediction> CpuBackend::evaluate(std::span<const Tokens* const> rows) {
    // Shortest first, then chunk so no chunk exceeds the attention budget.
    std::vector<std::size_t> order(rows.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](auto a, auto b) { return rows[a]->count < rows[b]->count; });
    std::vector<Prediction> out(rows.size());
    for (std::size_t at = 0; at < order.size();) {
        std::vector<const Tokens*> chunk;
        for (std::size_t i = at; i < order.size(); ++i) {
            const std::size_t n = rows[order[i]]->count;
            if (!chunk.empty() && (chunk.size() + 1) * n * n > attention_cells) break;
            chunk.push_back(rows[order[i]]);
        }
        auto predictions = impl_->run(chunk);
        for (std::size_t i = 0; i < chunk.size(); ++i) out[order[at + i]] = std::move(predictions[i]);
        at += chunk.size();
    }
    return out;
}

std::vector<Prediction> CpuBackend::evaluate(std::span<const Input> inputs) {
    const auto begin = Clock::now();
    std::vector<const Tokens*> rows;
    rows.reserve(inputs.size());
    for (const auto& input : inputs) rows.push_back(input.tokens);
    auto out = evaluate(rows);
    std::lock_guard lock{impl_->stats_mutex};
    ++impl_->batches; impl_->leaves += long(rows.size()); impl_->busy += since(begin);
    if (since(impl_->begin) - impl_->reported >= 60) { impl_->print(lanes_); impl_->reported = since(impl_->begin); }
    return out;
}

std::unique_ptr<Backend> cpu_backend(const std::filesystem::path& onnx, int lanes) {
    return std::make_unique<CpuBackend>(onnx, lanes);
}
}  // namespace stsrl::infra
