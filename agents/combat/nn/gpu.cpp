// Shared TensorRT evaluator. CPU workers pack requests; the GPU thread chooses a fixed-shape engine.
#include "agents/combat/nn/net.hpp"
#include "agents/combat/nn/packing.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <limits>
#include <mutex>
#include <thread>

namespace stsrl::nn {
namespace {
using Clock = std::chrono::steady_clock;
void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error{std::string{"CUDA: "} + cudaGetErrorString(e)}; }

// Each engine has one captured graph and permanent bindings. All engines share the same buffers;
// only the single evaluator thread runs them, so their input/output lifetimes cannot overlap.
struct Engine {
    int batch;
    Ort::Session session{nullptr};
    Ort::IoBinding binding{nullptr};

    Engine(Ort::Env& env, const std::filesystem::path& path, int tokens) {
        const auto cache = (path.parent_path() / "tensorrt").string();
        const auto timing = std::filesystem::absolute("build/nn/tensorrt-timing").string();
        std::filesystem::create_directories(cache);
        std::filesystem::create_directories(timing);
        const auto& api = Ort::GetApi();
        OrtTensorRTProviderOptionsV2* trt = nullptr;
        Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&trt));
        const std::array<const char*, 7> keys{
            "trt_fp16_enable", "trt_cuda_graph_enable", "trt_engine_cache_enable", "trt_engine_cache_path",
            "trt_timing_cache_enable", "trt_timing_cache_path", "trt_builder_optimization_level"};
        const std::array<const char*, 7> values{"1", "1", "1", cache.c_str(), "1", timing.c_str(), "3"};
        Ort::SessionOptions options;
        try {
            Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(trt, keys.data(), values.data(), keys.size()));
            options.AppendExecutionProvider_TensorRT_V2(*trt);
        } catch (...) { api.ReleaseTensorRTProviderOptions(trt); throw; }
        api.ReleaseTensorRTProviderOptions(trt);
        session = Ort::Session{env, path.c_str(), options};
        const auto input_type = session.GetInputTypeInfo(0);
        const auto input = input_type.GetTensorTypeAndShapeInfo();
        const auto shape = input.GetShape();
        if (input.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 || shape.size() != 3 ||
            shape[0] <= 0 || shape[0] > std::numeric_limits<int>::max() || shape[1] != tokens || shape[2] != width)
            throw std::invalid_argument{"gpu: export a fixed-shape fp16-input model matching tokens/features"};
        batch = int(shape[0]);
        binding = Ort::IoBinding{session};
    }

    void bind(__half* in, float* value, float* logits, int tokens) {
        const Ort::MemoryInfo gpu{"Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault};
        const std::array<std::int64_t, 3> input_shape{batch, tokens, width};
        const std::array<std::int64_t, 2> logits_shape{batch, tokens};
        const std::array<std::int64_t, 1> value_shape{batch};
        binding.BindInput("tokens", Ort::Value::CreateTensor(gpu, in, std::size_t(batch) * tokens * width * sizeof(__half),
                                                           input_shape.data(), 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
        binding.BindOutput("value", Ort::Value::CreateTensor<float>(gpu, value, batch, value_shape.data(), 1));
        binding.BindOutput("logits", Ort::Value::CreateTensor<float>(gpu, logits, batch * tokens, logits_shape.data(), 2));
    }
};

class GpuEvaluator final : public Evaluator {
public:
    GpuEvaluator(const std::filesystem::path& onnx, const std::filesystem::path& fixed_onnx, int batch, int tokens,
                 const std::vector<std::filesystem::path>& smaller_models, int wait_us, bool synchronous)
        : cpu_{onnx}, batch_{batch}, tokens_{tokens}, wait_us_{wait_us}, synchronous_{synchronous} {
        if (batch <= 0 || tokens <= 0 || wait_us < 0)
            throw std::invalid_argument{"gpu: positive batch/tokens and nonnegative wait required"};
        auto largest = std::make_unique<Engine>(env_, fixed_onnx, tokens);
        if (largest->batch != batch) throw std::invalid_argument{"gpu: fixed model does not match gpu-batch"};
        engines_.push_back(std::move(largest));
        for (const auto& path : smaller_models) {
            auto engine = std::make_unique<Engine>(env_, path, tokens);
            if (engine->batch >= batch) throw std::invalid_argument{"gpu: bank models must have smaller batches"};
            for (const auto& other : engines_) if (other->batch == engine->batch)
                throw std::invalid_argument{"gpu: duplicate bank batch size"};
            engines_.push_back(std::move(engine));
        }
        std::sort(engines_.begin(), engines_.end(), [](const auto& a, const auto& b) { return a->batch < b->batch; });
        check(cudaMallocHost(&host_in_, in_size() * sizeof(__half)));
        check(cudaMallocHost(&host_value_, batch * sizeof(float)));
        check(cudaMallocHost(&host_logits_, batch * tokens * sizeof(float)));
        check(cudaMalloc(&in_, in_size() * sizeof(__half)));
        check(cudaMalloc(&value_, batch * sizeof(float)));
        check(cudaMalloc(&logits_, batch * tokens * sizeof(float)));
        // Every row starts with a valid context token, including padding; no all-masked attention.
        std::memset(host_in_, 0, in_size() * sizeof(__half));
        for (int i = 0; i < batch; ++i) host_in_[std::size_t(i) * tokens * width] = __float2half(1.f);
        check(cudaMemcpy(in_, host_in_, in_size() * sizeof(__half), cudaMemcpyHostToDevice));
        for (auto& engine : engines_) engine->bind(in_, value_, logits_, tokens);
        begin_ = Clock::now();
        if (!synchronous_) thread_ = std::jthread{[this](std::stop_token stop) { loop(stop); }};
    }

    ~GpuEvaluator() override {
        if (thread_.joinable()) { thread_.request_stop(); wake_.notify_all(); thread_.join(); }
        std::fprintf(stderr, "gpu: %ld batches, mean %.0f leaves, %.1f s busy\n", runs_, runs_ ? double(leaves_) / runs_ : 0., busy_);
        report();
        engines_.clear(); // sessions/captured graphs must die before their borrowed buffers
        cudaFreeHost(host_in_); cudaFreeHost(host_value_); cudaFreeHost(host_logits_);
        cudaFree(in_); cudaFree(value_); cudaFree(logits_);
    }

    void prepare(const Tokens& tokens, std::vector<std::uint16_t>& packed) override {
        if (tokens.count > tokens_) { packed.clear(); return; } // full CPU fallback, never truncate
        packed.resize(std::size_t(tokens_) * width);
        pack_half(tokens.data, packed);
        std::fill(packed.begin() + tokens.data.size(), packed.end(), std::uint16_t{0});
    }

    std::future<std::vector<Prediction>> submit(std::vector<const Tokens*> leaves) override {
        std::vector<InferenceInput> inputs;
        for (auto tokens : leaves) inputs.push_back({tokens, {}});
        return submit_prepared(std::move(inputs));
    }

    std::future<std::vector<Prediction>> submit_prepared(std::vector<InferenceInput> inputs) override {
        if (std::ssize(inputs) > batch_) throw std::invalid_argument{"gpu: more leaves than the batch"};
        auto r = std::make_shared<Request>();
        for (const auto& input : inputs) r->leaves.push_back(input.tokens);
        const bool prepared = std::any_of(inputs.begin(), inputs.end(), [](const auto& input) { return !input.packed.empty(); });
        if (prepared) r->prepared = std::move(inputs);
        r->out.resize(r->leaves.size());
        std::vector<const Tokens*> large;
        std::vector<std::size_t> large_at;
        for (std::size_t i = 0; i < r->leaves.size(); ++i)
            if (r->leaves[i]->count > tokens_) { large.push_back(r->leaves[i]); large_at.push_back(i); }
            else r->gpu.push_back(i);
        if (!large.empty()) {
            cpu_leaves_ += large.size();
            auto p = cpu_(large);
            for (std::size_t k = 0; k < large.size(); ++k) r->out[large_at[k]] = std::move(p[k]);
        }
        auto future = r->done.get_future();
        if (r->gpu.empty()) { r->done.set_value(std::move(r->out)); return future; }
        if (!prepared) {
            std::lock_guard lock{mutex_};
            if (failure_) { r->done.set_exception(failure_); return future; }
            if (!buffers_.empty()) { r->packed = std::move(buffers_.back()); buffers_.pop_back(); }
        }
        const auto pack_begin = Clock::now();
        const auto stride = std::size_t(tokens_) * width;
        if (!prepared) r->packed.resize(r->gpu.size() * stride);
        long token_sum = 0, over32 = 0, over40 = 0;
        int token_max = 0;
        for (std::size_t k = 0; k < r->gpu.size(); ++k) {
            const auto& t = *r->leaves[r->gpu[k]];
            token_sum += t.count; token_max = std::max(token_max, t.count);
            over32 += t.count > 32; over40 += t.count > 40;
            if (prepared) {
                if (r->prepared[r->gpu[k]].packed.size() != stride)
                    throw std::invalid_argument{"gpu: prepared input has wrong size"};
            } else {
                auto* row = r->packed.data() + k * stride;
                pack_half(t.data, {reinterpret_cast<std::uint16_t*>(row), t.data.size()});
                std::fill(row + t.data.size(), row + stride, __float2half(0.f));
            }
        }
        token_sum_ += token_sum; over32_ += over32; over40_ += over40;
        for (int previous = token_max_.load(); previous < token_max && !token_max_.compare_exchange_weak(previous, token_max);) {}
        if (!prepared) pack_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - pack_begin).count();
        r->enqueued = Clock::now();
        if (synchronous_) {
            // The caller is the coordinator: no second dispatch thread, queue or batching delay.
            run({r});
            return future;
        }
        {
            std::lock_guard lock{mutex_};
            if (failure_) { r->done.set_exception(failure_); return future; }
            pending_ += int(r->gpu.size()); queue_.push_back(std::move(r));
        }
        wake_.notify_one();
        return future;
    }

private:
    struct Request {
        std::vector<const Tokens*> leaves;
        std::vector<std::size_t> gpu;
        std::vector<Prediction> out;
        std::vector<__half> packed;
        std::vector<InferenceInput> prepared;
        std::promise<std::vector<Prediction>> done;
        Clock::time_point enqueued;
    };
    std::size_t in_size() const { return std::size_t(batch_) * tokens_ * width; }

    void loop(std::stop_token stop) {
        using namespace std::chrono_literals;
        for (;;) {
            std::vector<std::shared_ptr<Request>> taken;
            {
                std::unique_lock lock{mutex_};
                while (!stop.stop_requested() && pending_ == 0) wake_.wait(lock);
                for (int seen = -1; wait_us_ > 0 && !stop.stop_requested() && pending_ < batch_ && pending_ != seen;) {
                    seen = pending_; wake_.wait_for(lock, std::chrono::microseconds{wait_us_});
                }
                if (queue_.empty()) { if (stop.stop_requested()) return; continue; }
                int n = 0;
                while (!queue_.empty() && n + int(queue_.front()->gpu.size()) <= batch_) {
                    n += int(queue_.front()->gpu.size());
                    taken.push_back(std::move(queue_.front())); queue_.pop_front();
                }
                pending_ -= n;
            }
            try { run(taken); }
            catch (...) {
                std::lock_guard lock{mutex_};
                failure_ = std::current_exception();
                for (const auto& r : taken) r->done.set_exception(failure_);
                for (const auto& r : queue_) r->done.set_exception(failure_);
                queue_.clear(); pending_ = 0;
                return;
            }
        }
    }

    void report() {
        const double seconds = std::chrono::duration<double>(Clock::now() - begin_).count();
        std::fprintf(stderr, "gpu_stats: {\"seconds\":%.6f,\"batches\":%ld,\"leaves\":%ld,\"capacity\":%ld,\"busy\":%.6f,\"cpu_leaves\":%ld,\"shapes\":{",
                     seconds, runs_, leaves_, capacity_, busy_, cpu_leaves_.load());
        bool first = true;
        for (const auto& [batch, count] : shape_runs_) {
            std::fprintf(stderr, "%s\"%d\":%ld", first ? "" : ",", batch, count);
            first = false;
        }
        std::fprintf(stderr, "},\"host_pack_seconds\":%.6f,\"concat_seconds\":%.6f,\"h2d_seconds\":%.6f,\"inference_seconds\":%.6f,\"d2h_seconds\":%.6f,\"scatter_seconds\":%.6f,\"queue_leaf_seconds\":%.6f,\"queue_max_seconds\":%.6f,\"token_sum\":%ld,\"token_max\":%d,\"over32\":%ld,\"over40\":%ld}\n",
                     double(pack_ns_.load()) / 1e9, concat_, h2d_, inference_, d2h_, scatter_, queue_leaf_seconds_, queue_max_,
                     token_sum_.load(), token_max_.load(), over32_.load(), over40_.load());
    }

    void run(const std::vector<std::shared_ptr<Request>>& taken) {
        const auto begin = Clock::now();
        int n = 0;
        for (const auto& r : taken) {
            const double wait = std::chrono::duration<double>(begin - r->enqueued).count();
            queue_leaf_seconds_ += wait * r->gpu.size(); queue_max_ = std::max(queue_max_, wait);
            if (r->prepared.empty()) {
                std::memcpy(host_in_ + std::size_t(n) * tokens_ * width, r->packed.data(), r->packed.size() * sizeof(__half));
                n += int(r->gpu.size());
            } else {
                for (auto index : r->gpu) {
                    const auto row = r->prepared[index].packed;
                    std::memcpy(host_in_ + std::size_t(n++) * tokens_ * width, row.data(), row.size_bytes());
                }
            }
        }
        auto& engine = **std::find_if(engines_.begin(), engines_.end(), [n](const auto& e) { return e->batch >= n; });
        const auto packed = Clock::now();
        check(cudaMemcpy(in_, host_in_, std::size_t(n) * tokens_ * width * sizeof(__half), cudaMemcpyHostToDevice));
        const auto uploaded = Clock::now();
        engine.session.Run(Ort::RunOptions{}, engine.binding);
        const auto inferred = Clock::now();
        check(cudaMemcpy(host_value_, value_, std::size_t(n) * sizeof(float), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(host_logits_, logits_, std::size_t(n) * tokens_ * sizeof(float), cudaMemcpyDeviceToHost));
        const auto downloaded = Clock::now();
        int row_index = 0;
        for (const auto& r : taken) {
            for (const auto i : r->gpu) {
                const auto& t = *r->leaves[i];
                const float* row = host_logits_ + std::size_t(row_index) * tokens_;
                r->out[i] = {host_value_[row_index++], {row + t.count - t.actions, row + t.count}};
            }
        }
        {
            std::lock_guard lock{mutex_};
            for (const auto& r : taken) if (!r->packed.empty()) buffers_.push_back(std::move(r->packed));
        }
        for (const auto& r : taken) r->done.set_value(std::move(r->out));
        const auto done = Clock::now();
        concat_ += std::chrono::duration<double>(packed - begin).count();
        h2d_ += std::chrono::duration<double>(uploaded - packed).count();
        inference_ += std::chrono::duration<double>(inferred - uploaded).count();
        d2h_ += std::chrono::duration<double>(downloaded - inferred).count();
        scatter_ += std::chrono::duration<double>(done - downloaded).count();
        ++runs_; leaves_ += n; capacity_ += engine.batch; ++shape_runs_[engine.batch];
        busy_ += std::chrono::duration<double>(done - begin).count();
        const double seconds = std::chrono::duration<double>(Clock::now() - begin_).count();
        if (seconds - last_report_ >= 1.) { report(); last_report_ = seconds; }
    }

    Net cpu_;
    const int batch_, tokens_, wait_us_;
    const bool synchronous_;
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "nn-gpu"};
    std::vector<std::unique_ptr<Engine>> engines_;
    __half *host_in_ = nullptr, *in_ = nullptr;
    float *host_value_ = nullptr, *host_logits_ = nullptr, *value_ = nullptr, *logits_ = nullptr;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::deque<std::shared_ptr<Request>> queue_;
    std::vector<std::vector<__half>> buffers_;
    std::exception_ptr failure_;
    int pending_ = 0;
    long runs_ = 0, leaves_ = 0, capacity_ = 0;
    std::atomic<long> cpu_leaves_{0};
    std::atomic<long long> pack_ns_{0};
    std::atomic<long> token_sum_{0}, over32_{0}, over40_{0};
    std::atomic<int> token_max_{0};
    std::map<int, long> shape_runs_;
    double busy_ = 0, last_report_ = 0;
    double concat_ = 0, h2d_ = 0, inference_ = 0, d2h_ = 0, scatter_ = 0;
    double queue_leaf_seconds_ = 0, queue_max_ = 0;
    Clock::time_point begin_;
    std::jthread thread_;
};
}  // namespace

std::unique_ptr<Evaluator> gpu_evaluator(const std::filesystem::path& onnx, const std::filesystem::path& fixed_onnx,
                                         int batch, int tokens, const std::vector<std::filesystem::path>& smaller_models,
                                         int wait_us, bool synchronous) {
    return std::make_unique<GpuEvaluator>(onnx, fixed_onnx, batch, tokens, smaller_models, wait_us, synchronous);
}
}  // namespace stsrl::nn
