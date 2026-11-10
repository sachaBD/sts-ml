#pragma once
// The model contract (ONNX): "tokens" [batch, tokens, width] -> "value" [batch] in [0, 1], "logits" [batch, tokens].
// Rows are padded with all-zero tokens; a row's legal actions are its last `actions` tokens.
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace stsrl::infra {
struct Tokens {
    std::vector<float> data;  // count * width
    int count = 0, actions = 0;
};

struct Prediction {
    float value;
    std::vector<float> logits;  // one per legal action
};

// The Actor owns both views, unchanged, until the reply.
struct Input {
    const Tokens* tokens;
    std::span<const std::uint16_t> packed;
};

class Backend {
public:
    virtual ~Backend() = default;
    virtual std::string name() const = 0;
    virtual int width() const = 0;
    virtual int max_batch() const = 0;
    virtual bool accepts(const Tokens&) const = 0;
    // Runs concurrently on the Actors' threads, so must not touch shared state.
    virtual void prepare(const Tokens&, std::vector<std::uint16_t>& packed) const = 0;
    virtual int lanes() const = 0;  // concurrent evaluate() calls it allows
    virtual std::vector<Prediction> evaluate(std::span<const Input>) = 0;
};

// fixed: one fixed-shape fp16 model per batch size, all of one token capacity.
std::unique_ptr<Backend> gpu_backend(const std::vector<std::filesystem::path>& fixed, int lanes);
// onnx: the dynamic-shape fp32 model.
std::unique_ptr<Backend> cpu_backend(const std::filesystem::path& onnx, int lanes, int batch);
}  // namespace stsrl::infra
