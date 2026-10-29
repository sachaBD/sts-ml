// TensorRT (through ONNX Runtime) with captured CUDA graphs. Synchronous: evaluate() runs on its caller,
// the InferenceServer, so there is no extra thread or queue here.
#include "infra/inference/cpu.hpp"
#include "infra/inference/half.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>

namespace stsrl::infra {
namespace {
using Clock = std::chrono::steady_clock;
double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }
void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error{std::string{"CUDA: "} + cudaGetErrorString(e)}; }

// Tactic timings depend on the GPU, not the model: one cache for every model makes rebuilding the
// engine for each new model much cheaper. Engines themselves are cached next to their model.
std::string timing_cache() {
    const char* xdg = std::getenv("XDG_CACHE_HOME");
    const char* home = std::getenv("HOME");
    const auto root = xdg && *xdg ? std::filesystem::path{xdg} : std::filesystem::path{home ? home : "."} / ".cache";
    const auto path = root / "stsrl" / "tensorrt-timing";
    std::filesystem::create_directories(path);
    return path.string();
}

// One fixed-shape model: a TensorRT engine with one captured graph, bound permanently to the shared buffers.
struct Engine {
    Ort::Session session{nullptr};
    Ort::IoBinding binding{nullptr};
    int batch = 0, tokens = 0, width = 0;

    Engine(Ort::Env& env, const std::filesystem::path& path) {
        const auto cache = (path.parent_path() / "tensorrt").string(), timing = timing_cache();
        std::filesystem::create_directories(cache);
        const auto& api = Ort::GetApi();
        OrtTensorRTProviderOptionsV2* trt = nullptr;
        Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&trt));
        const std::array<const char*, 7> keys{"trt_fp16_enable", "trt_cuda_graph_enable", "trt_engine_cache_enable",
            "trt_engine_cache_path", "trt_timing_cache_enable", "trt_timing_cache_path", "trt_builder_optimization_level"};
        const std::array<const char*, 7> values{"1", "1", "1", cache.c_str(), "1", timing.c_str(), "3"};
        Ort::SessionOptions options;
        try {
            Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(trt, keys.data(), values.data(), keys.size()));
            options.AppendExecutionProvider_TensorRT_V2(*trt);
        } catch (...) { api.ReleaseTensorRTProviderOptions(trt); throw; }
        api.ReleaseTensorRTProviderOptions(trt);
        session = Ort::Session{env, path.c_str(), options};
        const auto type = session.GetInputTypeInfo(0);  // owns the shape info below
        const auto info = type.GetTensorTypeAndShapeInfo();
        const auto shape = info.GetShape();
        if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 || shape.size() != 3 ||
            shape[0] <= 0 || shape[1] <= 0 || shape[2] <= 0 || shape[0] > std::numeric_limits<int>::max())
            throw std::invalid_argument{"gpu backend: " + path.string() + " is not a fixed-shape fp16 [batch, tokens, width] model"};
        batch = int(shape[0]); tokens = int(shape[1]); width = int(shape[2]);
        binding = Ort::IoBinding{session};
    }

    void bind(std::uint16_t* in, float* value, float* logits) {
        const Ort::MemoryInfo gpu{"Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault};
        const std::array<std::int64_t, 3> in_shape{batch, tokens, width};
        const std::array<std::int64_t, 2> logits_shape{batch, tokens};
        const std::array<std::int64_t, 1> value_shape{batch};
        binding.BindInput("tokens", Ort::Value::CreateTensor(gpu, in, std::size_t(batch) * tokens * width * sizeof(std::uint16_t),
                                                           in_shape.data(), 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
        binding.BindOutput("value", Ort::Value::CreateTensor<float>(gpu, value, batch, value_shape.data(), 1));
        binding.BindOutput("logits", Ort::Value::CreateTensor<float>(gpu, logits, std::size_t(batch) * tokens, logits_shape.data(), 2));
    }
};

class GpuBackend final : public Backend {
public:
    GpuBackend(const std::filesystem::path& onnx, const std::vector<std::filesystem::path>& fixed) : cpu_{onnx} {
        if (fixed.empty()) throw std::invalid_argument{"gpu backend: at least one fixed-shape model required"};
        for (const auto& path : fixed) engines_.push_back(std::make_unique<Engine>(env_, path));
        std::sort(engines_.begin(), engines_.end(), [](const auto& a, const auto& b) { return a->batch < b->batch; });
        tokens_ = engines_.front()->tokens;
        width_ = engines_.front()->width;
        for (std::size_t i = 0; i < engines_.size(); ++i) {
            const auto& e = *engines_[i];
            if (e.tokens != tokens_ || e.width != width_) throw std::invalid_argument{"gpu backend: fixed models differ in token shape"};
            if (i && e.batch == engines_[i - 1]->batch) throw std::invalid_argument{"gpu backend: duplicate batch size"};
        }
        if (width_ != cpu_.width()) throw std::invalid_argument{"gpu backend: fixed and dynamic models differ in width"};
        batch_ = engines_.back()->batch;
        const std::size_t cells = std::size_t(batch_) * tokens_;
        check(cudaMallocHost(&host_in_, cells * width_ * sizeof(std::uint16_t)));
        check(cudaMallocHost(&host_value_, batch_ * sizeof(float)));
        check(cudaMallocHost(&host_logits_, cells * sizeof(float)));
        check(cudaMalloc(&in_, cells * width_ * sizeof(std::uint16_t)));
        check(cudaMalloc(&value_, batch_ * sizeof(float)));
        check(cudaMalloc(&logits_, cells * sizeof(float)));
        // Rows past a batch's end are still computed: start every row with a context token so none is all-masked.
        std::memset(host_in_, 0, cells * width_ * sizeof(std::uint16_t));
        for (int i = 0; i < batch_; ++i) host_in_[std::size_t(i) * tokens_ * width_] = std::bit_cast<std::uint16_t>(__float2half(1.f));
        check(cudaMemcpy(in_, host_in_, cells * width_ * sizeof(std::uint16_t), cudaMemcpyHostToDevice));
        for (auto& engine : engines_) engine->bind(in_, value_, logits_);
        begin_ = Clock::now();
    }

    ~GpuBackend() override {
        report();
        engines_.clear();  // sessions and captured graphs before the buffers they borrow
        cudaFreeHost(host_in_); cudaFreeHost(host_value_); cudaFreeHost(host_logits_);
        cudaFree(in_); cudaFree(value_); cudaFree(logits_);
    }

    int width() const override { return width_; }
    int max_batch() const override { return batch_; }

    void prepare(const Tokens& tokens, std::vector<std::uint16_t>& packed) const override {
        if (tokens.count > tokens_) { packed.clear(); return; }  // too long: the CPU evaluates it whole
        packed.assign(std::size_t(tokens_) * width_, 0);
        pack_half(tokens.data, packed);
    }

    std::vector<Prediction> evaluate(std::span<const Input> inputs) override {
        if (std::ssize(inputs) > batch_) throw std::invalid_argument{"gpu backend: batch too large"};
        const auto begin = Clock::now();
        std::vector<Prediction> out(inputs.size());
        std::vector<std::size_t> gpu, cpu;
        for (std::size_t i = 0; i < inputs.size(); ++i) (inputs[i].packed.empty() ? cpu : gpu).push_back(i);
        if (!cpu.empty()) fallback(inputs, cpu, out);
        if (!gpu.empty()) run(inputs, gpu, out);
        busy_ += since(begin);
        if (since(begin_) - reported_ >= 60) { report(); reported_ = since(begin_); }
        return out;
    }

private:
    void fallback(std::span<const Input> inputs, const std::vector<std::size_t>& rows, std::vector<Prediction>& out) {
        std::vector<const Tokens*> tokens;
        for (auto i : rows) tokens.push_back(inputs[i].tokens);
        auto predictions = cpu_.evaluate(tokens);
        for (std::size_t k = 0; k < rows.size(); ++k) out[rows[k]] = std::move(predictions[k]);
        cpu_leaves_ += long(rows.size());
    }

    void run(std::span<const Input> inputs, const std::vector<std::size_t>& rows, std::vector<Prediction>& out) {
        const std::size_t stride = std::size_t(tokens_) * width_;
        const int n = int(rows.size());
        for (int k = 0; k < n; ++k) {
            const auto packed = inputs[rows[k]].packed;
            if (packed.size() != stride) throw std::invalid_argument{"gpu backend: packed input has the wrong size"};
            std::memcpy(host_in_ + k * stride, packed.data(), packed.size_bytes());
        }
        auto& engine = **std::find_if(engines_.begin(), engines_.end(), [n](const auto& e) { return e->batch >= n; });
        auto t = Clock::now();
        check(cudaMemcpy(in_, host_in_, n * stride * sizeof(std::uint16_t), cudaMemcpyHostToDevice));
        h2d_ += since(t); t = Clock::now();
        engine.session.Run(Ort::RunOptions{}, engine.binding);
        inference_ += since(t); t = Clock::now();
        check(cudaMemcpy(host_value_, value_, n * sizeof(float), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(host_logits_, logits_, std::size_t(n) * tokens_ * sizeof(float), cudaMemcpyDeviceToHost));
        d2h_ += since(t);
        for (int k = 0; k < n; ++k) {
            const auto& tokens = *inputs[rows[k]].tokens;
            const float* logits = host_logits_ + std::size_t(k) * tokens_;
            auto& p = out[rows[k]] = {host_value_[k], {logits + tokens.count - tokens.actions, logits + tokens.count}};
            const bool finite = std::isfinite(p.value) && std::all_of(p.logits.begin(), p.logits.end(), [](float x) { return std::isfinite(x); });
            if (!finite) {  // fp16 overflow on large features: redo this row in fp32 once
                const std::array<const Tokens*, 1> retry{&tokens};
                p = cpu_.evaluate(retry).front();
                if (++precision_retries_ == 1) std::fprintf(stderr, "gpu backend: non-finite fp16 prediction; retried in fp32 on the CPU\n");
            }
        }
        ++batches_; leaves_ += n; capacity_ += engine.batch; ++shapes_[engine.batch];
    }

    void report() const {
        std::fprintf(stderr, "gpu_stats: {\"seconds\":%.3f,\"batches\":%ld,\"leaves\":%ld,\"capacity\":%ld,\"cpu_leaves\":%ld,"
                     "\"cpu_precision_retries\":%ld,\"busy\":%.3f,\"h2d_seconds\":%.3f,\"inference_seconds\":%.3f,\"d2h_seconds\":%.3f,\"shapes\":{",
                     since(begin_), batches_, leaves_, capacity_, cpu_leaves_, precision_retries_, busy_, h2d_, inference_, d2h_);
        const char* separator = "";
        for (const auto& [batch, count] : shapes_) { std::fprintf(stderr, "%s\"%d\":%ld", separator, batch, count); separator = ","; }
        std::fprintf(stderr, "}}\n");
    }

    CpuBackend cpu_;
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "infra-gpu"};
    std::vector<std::unique_ptr<Engine>> engines_;
    int batch_ = 0, tokens_ = 0, width_ = 0;
    std::uint16_t *host_in_ = nullptr, *in_ = nullptr;  // fp16 bits
    float *host_value_ = nullptr, *host_logits_ = nullptr, *value_ = nullptr, *logits_ = nullptr;
    long batches_ = 0, leaves_ = 0, capacity_ = 0, cpu_leaves_ = 0, precision_retries_ = 0;
    std::map<int, long> shapes_;
    double busy_ = 0, h2d_ = 0, inference_ = 0, d2h_ = 0, reported_ = 0;
    Clock::time_point begin_;
};
}  // namespace

std::unique_ptr<Backend> gpu_backend(const std::filesystem::path& onnx, const std::vector<std::filesystem::path>& fixed) {
    return std::make_unique<GpuBackend>(onnx, fixed);
}
}  // namespace stsrl::infra
