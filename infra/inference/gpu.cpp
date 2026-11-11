// TensorRT through ONNX Runtime; independent streams overlap inference with host batch preparation.
#include "infra/inference/backend.hpp"
#include <cuda_fp16.h>
#include <immintrin.h>
#include <cuda_runtime.h>
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>

namespace stsrl::infra {
namespace {
// Vectorized float32 -> fp16 for the first multiple of 8 values; returns how many it converted.
__attribute__((target("avx,f16c")))
std::size_t to_fp16_f16c(std::span<const float> source, std::uint16_t* destination) {
    std::size_t i = 0;
    for (; i + 8 <= source.size(); i += 8)
        _mm_storeu_si128(reinterpret_cast<__m128i*>(destination + i),
            _mm256_cvtps_ph(_mm256_loadu_ps(source.data() + i), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    return i;
}

// float32 -> fp16 bits, round-to-nearest-even (as CUDA's __float2half). F16C when the CPU has it.
void to_fp16(std::span<const float> source, std::uint16_t* destination) {
    static const bool f16c = __builtin_cpu_supports("avx") && __builtin_cpu_supports("f16c");
    for (std::size_t i = f16c ? to_fp16_f16c(source, destination) : 0; i < source.size(); ++i)
        destination[i] = std::bit_cast<std::uint16_t>(__float2half(source[i]));
}

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

// One fixed-shape model: a TensorRT engine with one captured graph, bound permanently to its lane's buffers.
struct Engine {
    Ort::Session session{nullptr};
    Ort::IoBinding binding{nullptr};
    int batch = 0, tokens = 0, width = 0;

    Engine(Ort::Env& env, const std::filesystem::path& path, cudaStream_t stream) {
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
            Ort::ThrowOnError(api.UpdateTensorRTProviderOptionsWithValue(trt, "user_compute_stream", stream));
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

// Captured graphs retain buffer addresses; each lane needs its own engines and buffers.
struct Lane {
    cudaStream_t stream = nullptr;
    std::vector<std::unique_ptr<Engine>> engines;  // ascending batch size
    std::uint16_t *host_in = nullptr, *in = nullptr;  // fp16 bits
    float *host_value = nullptr, *host_logits = nullptr, *value = nullptr, *logits = nullptr;

    Lane(Ort::Env& env, const std::vector<std::filesystem::path>& fixed) {
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        try {
            for (const auto& path : fixed) engines.push_back(std::make_unique<Engine>(env, path, stream));
            std::sort(engines.begin(), engines.end(), [](const auto& a, const auto& b) { return a->batch < b->batch; });
        } catch (...) {
            engines.clear();
            cudaStreamDestroy(stream);
            throw;
        }
    }

    void allocate(int batch, int tokens, int width) {
        const std::size_t cells = std::size_t(batch) * tokens;
        check(cudaMallocHost(&host_in, cells * width * sizeof(std::uint16_t)));
        check(cudaMallocHost(&host_value, batch * sizeof(float)));
        check(cudaMallocHost(&host_logits, cells * sizeof(float)));
        check(cudaMalloc(&in, cells * width * sizeof(std::uint16_t)));
        check(cudaMalloc(&value, batch * sizeof(float)));
        check(cudaMalloc(&logits, cells * sizeof(float)));
        // Rows past a batch's end are still computed: start every row with a context token so none is all-masked.
        std::memset(host_in, 0, cells * width * sizeof(std::uint16_t));
        for (int i = 0; i < batch; ++i) host_in[std::size_t(i) * tokens * width] = std::bit_cast<std::uint16_t>(__float2half(1.f));
        check(cudaMemcpy(in, host_in, cells * width * sizeof(std::uint16_t), cudaMemcpyHostToDevice));
        for (auto& engine : engines) {
            engine->bind(in, value, logits);
            // Capture graphs before starting threads: concurrent first calls cause CUDA capture errors.
            for (int i = 0; i < 3; ++i) engine->session.Run(Ort::RunOptions{}, engine->binding);
            check(cudaStreamSynchronize(stream));
        }
    }

    ~Lane() {
        if (stream) cudaStreamSynchronize(stream);
        engines.clear();  // sessions and captured graphs before the buffers they borrow
        cudaFreeHost(host_in); cudaFreeHost(host_value); cudaFreeHost(host_logits);
        cudaFree(in); cudaFree(value); cudaFree(logits);
        if (stream) cudaStreamDestroy(stream);
    }
};

class GpuBackend final : public Backend {
    const int lane_count_;

public:
    GpuBackend(const std::vector<std::filesystem::path>& fixed, int lanes) : lane_count_{lanes} {
        if (lanes < 1) throw std::invalid_argument{"gpu backend: positive lanes required"};
        if (fixed.empty()) throw std::invalid_argument{"gpu backend: at least one fixed-shape model required"};
        for (int i = 0; i < lane_count_; ++i) lanes_.push_back(std::make_unique<Lane>(env_, fixed));
        const auto& engines = lanes_.front()->engines;
        tokens_ = engines.front()->tokens;
        width_ = engines.front()->width;
        for (std::size_t i = 0; i < engines.size(); ++i) {
            const auto& e = *engines[i];
            if (e.tokens != tokens_ || e.width != width_) throw std::invalid_argument{"gpu backend: fixed models differ in token shape"};
            if (i && e.batch == engines[i - 1]->batch) throw std::invalid_argument{"gpu backend: duplicate batch size"};
        }
        batch_ = engines.back()->batch;
        for (auto& lane : lanes_) { lane->allocate(batch_, tokens_, width_); free_.push_back(lane.get()); }
    }

    std::string name() const override { return "gpu-" + std::to_string(tokens_); }
    int width() const override { return width_; }
    int max_batch() const override { return batch_; }
    int lanes() const override { return lane_count_; }
    bool accepts(const Tokens& tokens) const override { return tokens.count <= tokens_; }

    void prepare(const Tokens& tokens, std::vector<std::uint16_t>& packed) const override {
        packed.assign(std::size_t(tokens_) * width_, 0);
        to_fp16(tokens.data, packed.data());
    }

    std::vector<Prediction> evaluate(std::span<const Input> inputs) override {
        if (std::ssize(inputs) > batch_) throw std::invalid_argument{"gpu backend: batch too large"};
        std::vector<Prediction> out(inputs.size());
        if (!inputs.empty()) {
            Lane& lane = acquire();
            try { run(lane, inputs, out); } catch (...) { release(lane); throw; }
            release(lane);
        }
        return out;
    }

private:
    Lane& acquire() {
        std::lock_guard lock{lanes_mutex_};
        if (free_.empty()) throw std::logic_error{"gpu backend: more concurrent evaluate() calls than lanes"};
        Lane* lane = free_.back();
        free_.pop_back();
        return *lane;
    }

    void release(Lane& lane) {
        std::lock_guard lock{lanes_mutex_};
        free_.push_back(&lane);
    }

    void run(Lane& lane, std::span<const Input> inputs, std::vector<Prediction>& out) {
        const std::size_t stride = std::size_t(tokens_) * width_;
        const int n = int(inputs.size());
        for (int k = 0; k < n; ++k) {
            const auto packed = inputs[k].packed;
            if (packed.size() != stride) throw std::invalid_argument{"gpu backend: packed input has the wrong size"};
            std::memcpy(lane.host_in + k * stride, packed.data(), packed.size_bytes());
        }
        auto& engine = **std::find_if(lane.engines.begin(), lane.engines.end(), [n](const auto& e) { return e->batch >= n; });
        check(cudaMemcpyAsync(lane.in, lane.host_in, n * stride * sizeof(std::uint16_t), cudaMemcpyHostToDevice, lane.stream));
        check(cudaStreamSynchronize(lane.stream));
        engine.session.Run(Ort::RunOptions{}, engine.binding);
        check(cudaStreamSynchronize(lane.stream));
        check(cudaMemcpyAsync(lane.host_value, lane.value, n * sizeof(float), cudaMemcpyDeviceToHost, lane.stream));
        check(cudaMemcpyAsync(lane.host_logits, lane.logits, std::size_t(n) * tokens_ * sizeof(float), cudaMemcpyDeviceToHost, lane.stream));
        check(cudaStreamSynchronize(lane.stream));
        for (int k = 0; k < n; ++k) {
            const auto& tokens = *inputs[k].tokens;
            const float* logits = lane.host_logits + std::size_t(k) * tokens_;
            out[k] = {lane.host_value[k], {logits + tokens.count - tokens.actions, logits + tokens.count}};
        }
    }

    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "infra-gpu"};
    std::vector<std::unique_ptr<Lane>> lanes_;
    std::mutex lanes_mutex_;
    std::vector<Lane*> free_;
    int batch_ = 0, tokens_ = 0, width_ = 0;
};
}  // namespace

std::unique_ptr<Backend> gpu_backend(const std::vector<std::filesystem::path>& fixed, int lanes) {
    return std::make_unique<GpuBackend>(fixed, lanes);
}
}  // namespace stsrl::infra
