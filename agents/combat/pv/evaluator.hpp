#pragma once
// Thin ONNX Runtime adapter. The model/weights remain entirely in PyTorch/ONNX.
#include "agents/combat/pv/features.hpp"
#include <filesystem>
#include <memory>

namespace stsrl::pv {
struct Prediction { float value; std::vector<float> logits; };
class Evaluator {
public:
    explicit Evaluator(const std::filesystem::path& model);
    ~Evaluator();
    Evaluator(const Evaluator&) = delete;
    Evaluator& operator=(const Evaluator&) = delete;
    std::vector<Prediction> evaluate(std::span<const Inputs> batch);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace stsrl::pv
