#include "agents/combat/pv/evaluator.hpp"
#include "agents/combat/pv/objective.hpp"
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace stsrl::pv {
struct Evaluator::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "pv"};
    Ort::Session session{nullptr};
    Impl(const std::filesystem::path& model) {
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(1); options.SetInterOpNumThreads(1);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session = Ort::Session{env, model.c_str(), options};
        Ort::AllocatorWithDefaultOptions allocator;
        const auto metadata = session.GetModelMetadata();
        const auto version = metadata.LookupCustomMetadataMapAllocated("pv_contract", allocator);
        if (!version || std::string{version.get()} != contract)
            throw std::invalid_argument{"PV: incompatible ONNX input contract"};
        if (session.GetInputCount() != names.size() || session.GetOutputCount() != 2)
            throw std::invalid_argument{"PV: unexpected ONNX inputs/outputs"};
        for (std::size_t i = 0; i < names.size(); ++i) {
            const auto name = session.GetInputNameAllocated(i, allocator);
            const auto type = session.GetInputTypeInfo(i);
            const auto info = type.GetTensorTypeAndShapeInfo();
            const auto dims = info.GetShape();
            if (std::string{name.get()} != names[i] || dims.size() != (i == 0 ? 2u : 3u) ||
                dims.back() != widths[i] || info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
                throw std::invalid_argument{"PV: ONNX input shape/type mismatch"};
        }
    }
};
Evaluator::Evaluator(const std::filesystem::path& model) : impl_{std::make_unique<Impl>(model)} {}
Evaluator::~Evaluator() = default;

std::vector<Prediction> Evaluator::evaluate(std::span<const Inputs> batch) {
    if (batch.empty()) return {};
    std::array<std::vector<float>, 6> storage;
    std::array<std::int64_t, 6> counts{};
    std::vector<Ort::Value> tensors;
    const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    for (std::size_t i = 0; i < names.size(); ++i) {
        for (const auto& state : batch) {
            if (state[i].empty() || state[i].size() % widths[i] || (i == 0 && state[i].size() != std::size_t(widths[i])))
                throw std::invalid_argument{"PV: malformed input rows"};
            counts[i] = std::max(counts[i], std::int64_t(state[i].size() / widths[i]));
        }
        const std::size_t stride = counts[i] * widths[i];
        storage[i].resize(batch.size() * stride, 0);
        for (std::size_t b = 0; b < batch.size(); ++b)
            std::copy(batch[b][i].begin(), batch[b][i].end(), storage[i].begin() + b * stride);
        std::vector<std::int64_t> shape{std::int64_t(batch.size())};
        if (i != 0) shape.push_back(counts[i]);
        shape.push_back(widths[i]);
        tensors.push_back(Ort::Value::CreateTensor<float>(memory, storage[i].data(), storage[i].size(), shape.data(), shape.size()));
    }
    constexpr std::array<const char*, 2> outputs{"value", "policy_logits"};
    auto result = impl_->session.Run(Ort::RunOptions{nullptr}, names.data(), tensors.data(), tensors.size(), outputs.data(), outputs.size());
    if (result[0].GetTensorTypeAndShapeInfo().GetElementCount() != batch.size() ||
        result[1].GetTensorTypeAndShapeInfo().GetElementCount() != batch.size() * counts[5])
        throw std::runtime_error{"PV: malformed ONNX outputs"};
    const float* values = result[0].GetTensorData<float>();
    const float* logits = result[1].GetTensorData<float>();
    std::vector<Prediction> predictions;
    for (std::size_t b = 0; b < batch.size(); ++b) {
        const auto n = batch[b][5].size() / widths[5];
        validate_value(values[b]);
        const auto* first = logits + b * counts[5];
        predictions.push_back({values[b], {first, first + n}});
    }
    return predictions;
}
}  // namespace stsrl::pv
