#include "agents/combat/nn/net.hpp"
#include <onnxruntime_cxx_api.h>
#include <algorithm>

namespace stsrl::nn {
struct Net::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "nn"};
    Ort::Session session{nullptr};
};

Net::Net(const std::filesystem::path& onnx) : impl_{std::make_unique<Impl>()} {
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    impl_->session = Ort::Session{impl_->env, onnx.c_str(), options};
}
Net::~Net() = default;

std::vector<Prediction> Net::operator()(std::span<const Tokens* const> batch) {
    std::int64_t most = 0;
    for (const auto* t : batch) most = std::max<std::int64_t>(most, t->count);
    std::vector<float> input(batch.size() * most * width, 0.f);  // type 0 = padding
    for (std::size_t b = 0; b < batch.size(); ++b)
        std::copy(batch[b]->data.begin(), batch[b]->data.end(), input.begin() + b * most * width);
    const std::array<std::int64_t, 3> shape{std::int64_t(batch.size()), most, width};
    const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<float>(memory, input.data(), input.size(), shape.data(), shape.size());
    constexpr std::array<const char*, 1> inputs{"tokens"};
    constexpr std::array<const char*, 2> outputs{"value", "logits"};
    auto result = impl_->session.Run(Ort::RunOptions{nullptr}, inputs.data(), &tensor, 1, outputs.data(), 2);
    const float* value = result[0].GetTensorData<float>();
    const float* logits = result[1].GetTensorData<float>();
    std::vector<Prediction> out;
    for (std::size_t b = 0; b < batch.size(); ++b) {
        const float* row = logits + b * most;
        out.push_back({value[b], {row + batch[b]->count - batch[b]->actions, row + batch[b]->count}});  // actions are last
    }
    return out;
}

namespace {
class CpuEvaluator final : public Evaluator {
public:
    explicit CpuEvaluator(const std::filesystem::path& onnx) : net_{onnx} {}
    std::future<std::vector<Prediction>> submit(std::vector<const Tokens*> leaves) override {
        std::promise<std::vector<Prediction>> done;
        done.set_value(net_(leaves));
        return done.get_future();
    }
private:
    Net net_;
};
}  // namespace

std::unique_ptr<Evaluator> cpu_evaluator(const std::filesystem::path& onnx) { return std::make_unique<CpuEvaluator>(onnx); }
}  // namespace stsrl::nn
