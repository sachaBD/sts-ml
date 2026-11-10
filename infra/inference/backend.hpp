#pragma once
// What crosses between Actors and the InferenceServer: token sequences in, predictions out.
// Model contract (ONNX): input "tokens" [batch, tokens, width] -> "value" [batch] in 0..1, "logits" [batch, tokens].
// Rows are padded with all-zero tokens; the legal actions are the LAST `actions` tokens of each row.
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace stsrl::infra {
struct Tokens {
    std::vector<float> data;  // count x width
    int count = 0, actions = 0;
};

struct Prediction {
    float value;                // 0..1
    std::vector<float> logits;  // one per action
};

// A request as the backend sees it. Both views stay owned and immutable by the Actor until the reply.
struct Input {
    const Tokens* tokens;
    std::span<const std::uint16_t> packed;  // Backend::prepare output
};

class Backend {
public:
    virtual ~Backend() = default;
    virtual std::string name() const = 0;  // names the backend in telemetry
    virtual int width() const = 0;      // floats per token
    virtual int max_batch() const = 0;  // most inputs per evaluate()
    // Capacity. The scheduler sends each input to the first backend that accepts it; none accepting is an error.
    virtual bool accepts(const Tokens&) const = 0;
    // Called concurrently by Actors on their own tokens, so must not touch shared state.
    virtual void prepare(const Tokens&, std::vector<std::uint16_t>& packed) const = 0;
    virtual int lanes() const = 0;  // maximum concurrent evaluate() calls
    virtual std::vector<Prediction> evaluate(std::span<const Input>) = 0;
};

// TensorRT through ONNX Runtime, CUDA graphs, fp16 input.
//   fixed: the network exported for fixed [batch, tokens, width] fp16 input, one model per batch size, all with the
//          same token capacity; it accepts inputs of up to that many tokens. The smallest model that fits each
//          batch is used. Engines are cached next to each model, TensorRT tactic timings in
//          $XDG_CACHE_HOME/stsrl/tensorrt-timing (default ~/.cache).
//   lanes: independent streams/buffers, so lanes evaluate concurrently.
std::unique_ptr<Backend> gpu_backend(const std::vector<std::filesystem::path>& fixed, int lanes);

// ONNX Runtime on the CPU: the dynamic-shape fp32 model; accepts any input, so it is the last backend for inputs
// no GPU model fits. Up to `batch` inputs per evaluate(), `lanes` concurrent calls, one thread each.
std::unique_ptr<Backend> cpu_backend(const std::filesystem::path& onnx, int lanes, int batch);
}  // namespace stsrl::infra
