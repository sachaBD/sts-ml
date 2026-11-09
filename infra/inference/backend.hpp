#pragma once
// What crosses between Actors and the InferenceServer: token sequences in, predictions out.
// Model contract (ONNX): input "tokens" [batch, tokens, width] -> "value" [batch] in 0..1, "logits" [batch, tokens].
// Rows are padded with all-zero tokens; the legal actions are the LAST `actions` tokens of each row.
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace stsrl::infra {
struct Tokens {
    std::vector<float> data;  // count x width
    int count = 0, actions = 0;
    int attended = 0;  // retained attention positions; 0 when unknown
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
    virtual int max_batch() const = 0;  // most inputs per evaluate()
    // Capacity. The scheduler sends inputs no backend accepts to its final backend.
    virtual bool accepts(const Tokens&) const { return true; }
    // Called concurrently by Actors on their own tokens, so must not touch shared state.
    virtual void prepare(const Tokens&, std::vector<std::uint16_t>& packed) const { packed.clear(); }
    virtual int lanes() const { return 1; }  // maximum concurrent evaluate() calls
    virtual std::vector<Prediction> evaluate(std::span<const Input>) = 0;
};

// ONNX Runtime on the CPU: any batch, any token count; `lanes` concurrent evaluate() calls, one thread each.
// Reports cpu_stats on stderr.
std::unique_ptr<Backend> cpu_backend(const std::filesystem::path& onnx, int lanes = 1);

// TensorRT through ONNX Runtime, CUDA graphs, fp16 input.
//   onnx:  the dynamic-shape fp32 model; re-evaluates any non-finite prediction (never dropped).
//          Inputs it does not accept are an error: put a CPU backend after it.
//   lanes: independent streams/buffers; attended: retained-position capacity, 0 if unbounded.
//   fixed: the SAME network exported for fixed [batch, tokens, width] fp16 input, one model per batch size.
//          The smallest model that fits each batch is used. Engines are cached next to each model,
//          TensorRT tactic timings in $XDG_CACHE_HOME/stsrl/tensorrt-timing (default ~/.cache).
std::unique_ptr<Backend> gpu_backend(const std::filesystem::path& onnx, const std::vector<std::filesystem::path>& fixed,
                                     int lanes = 2, int attended = 0);
}  // namespace stsrl::infra
