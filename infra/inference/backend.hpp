#pragma once
// What crosses between Actors and the InferenceServer: token sequences in, predictions out.
// Model contract (ONNX): input "tokens" [batch, tokens, width] -> "value" [batch] in 0..1, "logits" [batch, tokens].
// Rows are padded with all-zero tokens; the legal actions are the LAST `actions` tokens of each row.
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace stsrl::infra {
struct Tokens {
    std::vector<float> data;  // count x width
    int count = 0, actions = 0;
    int attended = 0;  // unpooled attention positions; 0 if the encoder does not supply this metadata
};

struct Prediction {
    float value;                // 0..1
    std::vector<float> logits;  // one per action
};

// A request as the backend sees it. Both views stay owned and immutable by the Actor until the reply.
struct Input {
    const Tokens* tokens;
    std::span<const std::uint16_t> packed;  // Backend::prepare output; empty = not prepared
};

class Backend {
public:
    virtual ~Backend() = default;
    virtual int width() const = 0;      // floats per token
    virtual int max_batch() const = 0;  // route 0's capacity
    // Called concurrently by Actors on their own tokens, so must not touch shared state.
    virtual void prepare(const Tokens&, std::vector<std::uint16_t>& packed) const { packed.clear(); }
    virtual int lanes() const { return 1; }  // maximum concurrent evaluate() calls
    virtual std::vector<Prediction> evaluate(std::span<const Input>) = 0;

    // Each route has its own queue and dedicated server lanes. Single-route backends need no overrides.
    virtual int routes() const { return 1; }
    virtual int route(const Tokens&) const { return 0; }
    virtual int lane_route(int) const { return 0; }
    virtual int max_batch(int route) const {
        if (route != 0) throw std::invalid_argument{"invalid inference route"};
        return max_batch();
    }
    virtual std::vector<Prediction> evaluate(std::span<const Input> inputs, int route) {
        if (route != 0) throw std::invalid_argument{"invalid inference route"};
        return evaluate(inputs);
    }
};

// ONNX Runtime on the CPU, one thread: any batch, any token count.
std::unique_ptr<Backend> cpu_backend(const std::filesystem::path& onnx);

struct GpuRoute {
    std::vector<std::filesystem::path> models;  // same token shape, distinct fixed batch sizes
    int lanes = 2;
    int attended = 0;  // 0 = no retained-attention bound
};

// TensorRT through ONNX Runtime, CUDA graphs, fp16 input. Route 0 is narrow; an optional second route
// must cover its token and attention capacities. Requests exceeding both shapes use the last route's CPU
// fallback. Non-finite GPU predictions are retried on the full fp32 CPU model, never truncated or dropped.
// Each lane owns its stream, sessions and pinned/device buffers. All graphs are captured before workers start.
// Engines are cached beside models; tactic timings under $XDG_CACHE_HOME/stsrl/tensorrt-timing.
std::unique_ptr<Backend> gpu_backend(const std::filesystem::path& onnx, const std::vector<std::filesystem::path>& fixed);
std::unique_ptr<Backend> gpu_backend(const std::filesystem::path& onnx, const GpuRoute& narrow, const GpuRoute& wide);
}  // namespace stsrl::infra
