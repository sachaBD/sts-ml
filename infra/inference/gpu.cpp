// TensorRT through ONNX Runtime; independent streams overlap inference with host batch preparation.
#include "infra/inference/cpu.hpp"
#include <cuda_fp16.h>
#include <immintrin.h>
#include <cuda_runtime.h>
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>

namespace stsrl::infra {
namespace {
using Clock = std::chrono::steady_clock;
double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }
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
    struct Group {
        std::vector<Lane*> free;
        int tokens = 0, batch = 0, attended = 0, lanes = 0;
        bool fits(const Tokens& t) const {
            return t.count <= tokens && (!attended || (t.attended > 0 && t.attended <= attended));
        }
    };

public:
    GpuBackend(const std::filesystem::path& onnx, const std::vector<GpuRoute>& routes) : cpu_{onnx} {
        if (routes.empty()) throw std::invalid_argument{"gpu backend: at least one route required"};
        for (const auto& config : routes) {
            if (config.models.empty() || config.lanes < 1 || config.attended < 0)
                throw std::invalid_argument{"gpu backend: models, positive lanes and nonnegative attended required"};
            Group group;
            group.attended = config.attended;
            group.lanes = config.lanes;
            for (int i = 0; i < config.lanes; ++i) {
                auto lane = std::make_unique<Lane>(env_, config.models);
                const auto& engines = lane->engines;
                group.tokens = engines.front()->tokens;
                group.batch = engines.back()->batch;
                for (std::size_t k = 0; k < engines.size(); ++k) {
                    const auto& e = *engines[k];
                    if (e.tokens != group.tokens || e.width != cpu_.width())
                        throw std::invalid_argument{"gpu backend: fixed models differ in token shape or width"};
                    if (k && e.batch == engines[k - 1]->batch)
                        throw std::invalid_argument{"gpu backend: duplicate batch size"};
                }
                if (group.attended > group.tokens)
                    throw std::invalid_argument{"gpu backend: attended exceeds token capacity"};
                lane->allocate(group.batch, group.tokens, cpu_.width());
                group.free.push_back(lane.get());
                lanes_.push_back(std::move(lane));
            }
            if (!groups_.empty()) {
                const auto& previous = groups_.back();
                if (group.tokens < previous.tokens ||
                    (group.attended && (!previous.attended || group.attended < previous.attended)))
                    throw std::invalid_argument{"gpu backend: wider route must cover the preceding route"};
            }
            groups_.push_back(std::move(group));
        }
        route_stats_.resize(groups_.size());
        begin_ = Clock::now();
    }

    ~GpuBackend() override {
        report();
        lanes_.clear();
    }

    int width() const override { return cpu_.width(); }
    int max_batch() const override { return max_batch(0); }
    int max_batch(int route) const override { return groups_.at(route).batch; }
    int lanes() const override { return int(lanes_.size()); }
    int routes() const override { return int(groups_.size()); }
    int route(const Tokens& tokens) const override {
        for (int r = 0; r < routes(); ++r) if (groups_[r].fits(tokens)) return r;
        return routes() - 1;  // CPU fallback shares only the widest route, never the narrow lanes.
    }
    int lane_route(int lane) const override {
        for (int r = 0; r < routes(); ++r) {
            if (lane < groups_[r].lanes) return r;
            lane -= groups_[r].lanes;
        }
        throw std::out_of_range{"invalid inference lane"};
    }

    void prepare(const Tokens& tokens, std::vector<std::uint16_t>& packed) const override {
        const auto& group = groups_[route(tokens)];
        if (!group.fits(tokens)) { packed.clear(); return; }  // full CPU model, no truncation
        packed.assign(std::size_t(group.tokens) * width(), 0);
        to_fp16(tokens.data, packed.data());
    }

    std::vector<Prediction> evaluate(std::span<const Input> inputs) override {
        return evaluate(inputs, inputs.empty() ? 0 : route(*inputs.front().tokens));
    }

    std::vector<Prediction> evaluate(std::span<const Input> inputs, int route_id) override {
        auto& group = groups_.at(route_id);
        if (std::ssize(inputs) > group.batch) throw std::invalid_argument{"gpu backend: batch too large"};
        for (const auto& input : inputs)
            if (route(*input.tokens) != route_id) throw std::invalid_argument{"gpu backend: mixed inference routes"};
        const auto begin = Clock::now();
        Stats stats;
        std::vector<Prediction> out(inputs.size());
        std::vector<std::size_t> gpu, cpu;
        for (std::size_t i = 0; i < inputs.size(); ++i) (inputs[i].packed.empty() ? cpu : gpu).push_back(i);
        if (!cpu.empty()) fallback(inputs, cpu, out, stats);
        if (!gpu.empty()) {
            Lane& lane = acquire(group);
            try { run(lane, group, inputs, gpu, out, stats); } catch (...) { release(group, lane); throw; }
            release(group, lane);
        }
        stats.busy = since(begin);
        std::lock_guard lock{stats_mutex_};
        total_ += stats;
        route_stats_[route_id] += stats;
        if (since(begin_) - reported_ >= 60) { print(); reported_ = since(begin_); }
        return out;
    }

private:
    // Summed over lanes, so the seconds can exceed wall time while lanes overlap.
    struct Stats {
        long batches = 0, leaves = 0, capacity = 0, cpu_leaves = 0, precision_retries = 0;
        std::map<int, long> shapes;
        double busy = 0, h2d = 0, inference = 0, d2h = 0;
        Stats& operator+=(const Stats& o) {
            batches += o.batches; leaves += o.leaves; capacity += o.capacity; cpu_leaves += o.cpu_leaves;
            precision_retries += o.precision_retries; busy += o.busy; h2d += o.h2d; inference += o.inference; d2h += o.d2h;
            for (const auto& [batch, count] : o.shapes) shapes[batch] += count;
            return *this;
        }
    };

    Lane& acquire(Group& group) {
        std::lock_guard lock{lanes_mutex_};
        if (group.free.empty()) throw std::logic_error{"gpu backend: more concurrent evaluate() calls than route lanes"};
        Lane* lane = group.free.back();
        group.free.pop_back();
        return *lane;
    }

    void release(Group& group, Lane& lane) {
        std::lock_guard lock{lanes_mutex_};
        group.free.push_back(&lane);
    }

    void fallback(std::span<const Input> inputs, const std::vector<std::size_t>& rows, std::vector<Prediction>& out, Stats& stats) {
        std::vector<const Tokens*> tokens;
        for (auto i : rows) tokens.push_back(inputs[i].tokens);
        auto predictions = cpu_.evaluate(tokens);
        for (std::size_t k = 0; k < rows.size(); ++k) out[rows[k]] = std::move(predictions[k]);
        stats.cpu_leaves += long(rows.size());
    }

    void run(Lane& lane, const Group& group, std::span<const Input> inputs, const std::vector<std::size_t>& rows, std::vector<Prediction>& out, Stats& stats) {
        const int tokens_ = group.tokens;
        const std::size_t stride = std::size_t(tokens_) * width();
        const int n = int(rows.size());
        for (int k = 0; k < n; ++k) {
            const auto packed = inputs[rows[k]].packed;
            if (packed.size() != stride) throw std::invalid_argument{"gpu backend: packed input has the wrong size"};
            std::memcpy(lane.host_in + k * stride, packed.data(), packed.size_bytes());
        }
        auto& engine = **std::find_if(lane.engines.begin(), lane.engines.end(), [n](const auto& e) { return e->batch >= n; });
        auto t = Clock::now();
        check(cudaMemcpyAsync(lane.in, lane.host_in, n * stride * sizeof(std::uint16_t), cudaMemcpyHostToDevice, lane.stream));
        check(cudaStreamSynchronize(lane.stream));
        stats.h2d += since(t); t = Clock::now();
        engine.session.Run(Ort::RunOptions{}, engine.binding);
        check(cudaStreamSynchronize(lane.stream));
        stats.inference += since(t); t = Clock::now();
        check(cudaMemcpyAsync(lane.host_value, lane.value, n * sizeof(float), cudaMemcpyDeviceToHost, lane.stream));
        check(cudaMemcpyAsync(lane.host_logits, lane.logits, std::size_t(n) * tokens_ * sizeof(float), cudaMemcpyDeviceToHost, lane.stream));
        check(cudaStreamSynchronize(lane.stream));
        stats.d2h += since(t);
        for (int k = 0; k < n; ++k) {
            const auto& tokens = *inputs[rows[k]].tokens;
            const float* logits = lane.host_logits + std::size_t(k) * tokens_;
            auto& p = out[rows[k]] = {lane.host_value[k], {logits + tokens.count - tokens.actions, logits + tokens.count}};
            const bool finite = std::isfinite(p.value) && std::all_of(p.logits.begin(), p.logits.end(), [](float x) { return std::isfinite(x); });
            if (!finite) {  // capacity or numerical failure: redo this row on the full fp32 model
                const std::array<const Tokens*, 1> retry{&tokens};
                p = cpu_.evaluate(retry).front();
                if (precision_warned_.exchange(true) == false)
                    std::fprintf(stderr, "gpu backend: non-finite GPU prediction; retried in fp32 on the CPU\n");
                ++stats.precision_retries;
            }
        }
        ++stats.batches; stats.leaves += n; stats.capacity += engine.batch; ++stats.shapes[engine.batch];
    }

    void report() {
        std::lock_guard lock{stats_mutex_};
        print();
    }

    void print() const {  // caller holds stats_mutex_
        print_stats(total_, "gpu_stats", -1, lanes());
        for (int r = 0; r < routes(); ++r) print_stats(route_stats_[r], "gpu_route_stats", r, groups_[r].lanes);
    }

    void print_stats(const Stats& s, const char* label, int route, int lane_count) const {
        std::fprintf(stderr, "%s: {\"route\":%d,\"seconds\":%.3f,\"lanes\":%d,\"batches\":%ld,\"leaves\":%ld,\"capacity\":%ld,\"cpu_leaves\":%ld,"
                     "\"cpu_precision_retries\":%ld,\"busy\":%.3f,\"h2d_seconds\":%.3f,\"inference_seconds\":%.3f,\"d2h_seconds\":%.3f,\"shapes\":{",
                     label, route, since(begin_), lane_count, s.batches, s.leaves, s.capacity, s.cpu_leaves, s.precision_retries, s.busy, s.h2d, s.inference, s.d2h);
        const char* separator = "";
        for (const auto& [batch, count] : s.shapes) { std::fprintf(stderr, "%s\"%d\":%ld", separator, batch, count); separator = ","; }
        std::fprintf(stderr, "}}\n");
    }

    CpuBackend cpu_;  // ONNX Runtime sessions run concurrently, so lanes share it
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "infra-gpu"};
    std::vector<std::unique_ptr<Lane>> lanes_;
    std::mutex lanes_mutex_, stats_mutex_;
    std::vector<Group> groups_;
    Stats total_;
    std::vector<Stats> route_stats_;
    std::atomic<bool> precision_warned_{false};
    double reported_ = 0;
    Clock::time_point begin_;
};
}  // namespace

std::unique_ptr<Backend> gpu_backend(const std::filesystem::path& onnx, const std::vector<std::filesystem::path>& fixed) {
    return gpu_backend(onnx, GpuRoute{fixed}, {});
}

std::unique_ptr<Backend> gpu_backend(const std::filesystem::path& onnx, const GpuRoute& narrow, const GpuRoute& wide) {
    std::vector<GpuRoute> routes{narrow};
    if (!wide.models.empty()) routes.push_back(wide);
    return std::make_unique<GpuBackend>(onnx, routes);
}
}  // namespace stsrl::infra
