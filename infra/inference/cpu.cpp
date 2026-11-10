// ONNX Runtime on the CPU: any batch and token count, one intra-op thread per lane.
#include "infra/inference/backend.hpp"
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

class CpuBackend final : public Backend {
public:
    CpuBackend(const std::filesystem::path& onnx, int lanes, int batch) : lanes_{lanes}, batch_{batch} {
        if (lanes < 1 || batch < 1) throw std::invalid_argument{"cpu backend: positive lanes and batch required"};
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(1);
        options.SetInterOpNumThreads(1);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_ = Ort::Session{env_, onnx.c_str(), options};
        const auto type = session_.GetInputTypeInfo(0);
        const auto shape = type.GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() != 3 || shape[2] <= 0) throw std::invalid_argument{"cpu backend: input must be [batch, tokens, width]"};
        width_ = int(shape[2]);
    }

    ~CpuBackend() override {
        std::lock_guard lock{stats_mutex_};
        print();
    }

    std::string name() const override { return "cpu"; }
    int width() const override { return width_; }
    int max_batch() const override { return batch_; }
    int lanes() const override { return lanes_; }
    bool accepts(const Tokens&) const override { return true; }
    void prepare(const Tokens&, std::vector<std::uint16_t>& packed) const override { packed.clear(); }

    std::vector<Prediction> evaluate(std::span<const Input> inputs) override {
        const auto begin = Clock::now();
        // Shortest first, then chunk so no chunk exceeds the attention budget.
        std::vector<std::size_t> order(inputs.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](auto a, auto b) { return inputs[a].tokens->count < inputs[b].tokens->count; });
        std::vector<Prediction> out(inputs.size());
        for (std::size_t at = 0; at < order.size();) {
            std::vector<const Tokens*> chunk;
            for (std::size_t i = at; i < order.size(); ++i) {
                const std::size_t n = inputs[order[i]].tokens->count;
                if (!chunk.empty() && (chunk.size() + 1) * n * n > attention_cells) break;
                chunk.push_back(inputs[order[i]].tokens);
            }
            auto predictions = run(chunk);
            for (std::size_t i = 0; i < chunk.size(); ++i) out[order[at + i]] = std::move(predictions[i]);
            at += chunk.size();
        }
        std::lock_guard lock{stats_mutex_};
        ++batches_; leaves_ += long(inputs.size()); busy_ += since(begin);
        if (since(begin_) - reported_ >= 60) { print(); reported_ = since(begin_); }
        return out;
    }

private:
    std::vector<Prediction> run(std::span<const Tokens* const> rows) {
        int longest = 0;
        for (const auto* t : rows) longest = std::max(longest, t->count);
        std::vector<float> input(rows.size() * longest * width_, 0.f);  // all-zero token = padding
        for (std::size_t b = 0; b < rows.size(); ++b)
            std::copy(rows[b]->data.begin(), rows[b]->data.end(), input.begin() + b * longest * width_);
        const std::array<std::int64_t, 3> shape{std::int64_t(rows.size()), longest, width_};
        const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto tensor = Ort::Value::CreateTensor<float>(memory, input.data(), input.size(), shape.data(), shape.size());
        constexpr std::array<const char*, 1> inputs{"tokens"};
        constexpr std::array<const char*, 2> outputs{"value", "logits"};
        auto result = session_.Run(Ort::RunOptions{nullptr}, inputs.data(), &tensor, 1, outputs.data(), 2);
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

    void print() const {  // caller holds stats_mutex_; busy is summed over lanes, so it can exceed wall time
        std::fprintf(stderr, "cpu_stats: {\"seconds\":%.3f,\"lanes\":%d,\"batches\":%ld,\"leaves\":%ld,\"busy\":%.3f}\n",
                     since(begin_), lanes_, batches_, leaves_, busy_);
    }

    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "infra-cpu"};
    Ort::Session session_{nullptr};  // Run is safe to call concurrently, so lanes share it
    const int lanes_, batch_;
    int width_ = 0;
    std::mutex stats_mutex_;
    long batches_ = 0, leaves_ = 0;
    double busy_ = 0, reported_ = 0;
    const Clock::time_point begin_ = Clock::now();
};
}  // namespace

std::unique_ptr<Backend> cpu_backend(const std::filesystem::path& onnx, int lanes, int batch) {
    return std::make_unique<CpuBackend>(onnx, lanes, batch);
}
}  // namespace stsrl::infra
