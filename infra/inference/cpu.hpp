#pragma once
// The CPU backend as a type: the GPU backend also uses it for fp16-overflow retries.
// Everyone else should use cpu_backend() from backend.hpp.
#include "infra/inference/backend.hpp"
#include <limits>

namespace stsrl::infra {
class CpuBackend final : public Backend {
public:
    explicit CpuBackend(const std::filesystem::path& onnx, int lanes = 1);
    ~CpuBackend() override;
    int width() const override;
    int max_batch() const override { return std::numeric_limits<int>::max(); }
    int lanes() const override { return lanes_; }
    // As a Backend: counted in cpu_stats.
    std::vector<Prediction> evaluate(std::span<const Input>) override;
    // Direct use (the GPU backend's retries): not counted.
    std::vector<Prediction> evaluate(std::span<const Tokens* const>);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int lanes_;
};
}  // namespace stsrl::infra
