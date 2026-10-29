#pragma once
// The CPU backend as a type: the GPU backend also uses it for long rows and fp16-overflow retries.
// Everyone else should use cpu_backend() from backend.hpp.
#include "infra/inference/backend.hpp"
#include <limits>

namespace stsrl::infra {
class CpuBackend final : public Backend {
public:
    explicit CpuBackend(const std::filesystem::path& onnx);
    ~CpuBackend() override;
    int width() const override;
    int max_batch() const override { return std::numeric_limits<int>::max(); }
    std::vector<Prediction> evaluate(std::span<const Input>) override;
    std::vector<Prediction> evaluate(std::span<const Tokens* const>);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace stsrl::infra
