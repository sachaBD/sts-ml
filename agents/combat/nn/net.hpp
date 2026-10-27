#pragma once
// The network: tokens [B, T, width] -> value [B] (win probability) and logits [B, T] (used at action tokens).
#include "agents/combat/nn/tokens.hpp"
#include <cstdint>
#include <filesystem>
#include <future>
#include <memory>

namespace stsrl::nn {
struct Prediction {
    float value;                 // win probability, 0..1
    std::vector<float> logits;   // one per legal action
};

// ONNX model on the CPU (one thread per call; calls from several threads are safe), any batch and token count.
class Net {
public:
    explicit Net(const std::filesystem::path& onnx);
    ~Net();
    std::vector<Prediction> operator()(std::span<const Tokens* const> batch);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Views remain owned/immutable on the SimWorker until its prediction returns.
struct InferenceInput {
    const Tokens* tokens;
    std::span<const std::uint16_t> packed;
};

// Batched evaluation shared by the worker threads: submit() returns at once, the predictions arrive later.
class Evaluator {
public:
    virtual ~Evaluator() = default;
    virtual std::future<std::vector<Prediction>> submit(std::vector<const Tokens*> leaves) = 0;
    virtual void prepare(const Tokens&, std::vector<std::uint16_t>& packed) { packed.clear(); }
    virtual std::future<std::vector<Prediction>> submit_prepared(std::vector<InferenceInput> inputs) {
        std::vector<const Tokens*> leaves;
        for (const auto& input : inputs) leaves.push_back(input.tokens);
        return submit(std::move(leaves));
    }
};

// Evaluates on the calling thread (the future is ready on return).
std::unique_ptr<Evaluator> cpu_evaluator(const std::filesystem::path& onnx);

// One GPU thread runs a captured CUDA graph of `fixed_onnx` (exported for exactly [batch, tokens, width] with fp16 input) on
// leaves gathered from all submitters, padded to `batch`. Leaves with more than `tokens` tokens use `onnx` on the CPU.
// With synchronous=true, submit() evaluates the already-batched request on its caller (the Coordinator);
// no evaluator thread/queue is started. All calls must then come from that one owner.
std::unique_ptr<Evaluator> gpu_evaluator(const std::filesystem::path& onnx, const std::filesystem::path& fixed_onnx,
                                         int batch, int tokens,
                                         const std::vector<std::filesystem::path>& smaller_models = {},
                                         int wait_us = 1000, bool synchronous = false);
}  // namespace stsrl::nn
