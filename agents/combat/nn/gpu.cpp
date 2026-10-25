// One TensorRT GPU thread; CPU workers prepare padded fp16 requests before queueing them.
#include "agents/combat/nn/net.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <onnxruntime_cxx_api.h>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>

namespace stsrl::nn {
namespace {
void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error{std::string{"CUDA: "} + cudaGetErrorString(e)}; }

class GpuEvaluator final : public Evaluator {
public:
    GpuEvaluator(const std::filesystem::path& onnx, const std::filesystem::path& fixed_onnx, int batch, int tokens)
        : cpu_{onnx}, batch_{batch}, tokens_{tokens} {
        if (batch <= 0 || tokens <= 0) throw std::invalid_argument{"gpu: positive batch and token counts required"};
        // Engine cache is per model. Timing cache is shared across models: retain measured kernel timings instead
        // of retiming the same topology every iteration. Keep optimization level 3; level 5 was slower to build
        // without resolving tactic variability. A timing cache reduces variability, not a determinism guarantee.
        const auto engine_cache = (fixed_onnx.parent_path() / "tensorrt").string();
        const auto timing_cache = std::filesystem::absolute("build/nn/tensorrt-timing").string();
        std::filesystem::create_directories(engine_cache);
        std::filesystem::create_directories(timing_cache);
        const auto& api = Ort::GetApi();
        OrtTensorRTProviderOptionsV2* trt = nullptr;
        Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&trt));
        const std::array<const char*, 7> keys{
            "trt_fp16_enable", "trt_cuda_graph_enable", "trt_engine_cache_enable", "trt_engine_cache_path",
            "trt_timing_cache_enable", "trt_timing_cache_path", "trt_builder_optimization_level"};
        const std::array<const char*, 7> values{"1", "1", "1", engine_cache.c_str(), "1", timing_cache.c_str(), "3"};
        Ort::SessionOptions options;
        try {
            Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(trt, keys.data(), values.data(), keys.size()));
            options.AppendExecutionProvider_TensorRT_V2(*trt);
        } catch (...) { api.ReleaseTensorRTProviderOptions(trt); throw; }
        api.ReleaseTensorRTProviderOptions(trt);
        session_ = Ort::Session{env_, fixed_onnx.c_str(), options};
        binding_ = Ort::IoBinding{session_};
        const auto input_type = session_.GetInputTypeInfo(0);  // owns the shape/type view below
        const auto input = input_type.GetTensorTypeAndShapeInfo();
        if (input.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
            input.GetShape() != std::vector<std::int64_t>{batch, tokens, width})
            throw std::invalid_argument{"gpu: export a fixed-shape fp16-input model matching batch/tokens"};

        check(cudaMallocHost(&host_in_, in_size() * sizeof(__half)));
        check(cudaMallocHost(&host_value_, batch * sizeof(float)));
        check(cudaMallocHost(&host_logits_, batch * tokens * sizeof(float)));
        check(cudaMalloc(&in_, in_size() * sizeof(__half)));
        check(cudaMalloc(&value_, batch * sizeof(float)));
        check(cudaMalloc(&logits_, batch * tokens * sizeof(float)));
        // Unused rows have a valid context token, avoiding all-masked attention. Their outputs are discarded.
        std::memset(host_in_, 0, in_size() * sizeof(__half));
        for (int i = 0; i < batch; ++i) host_in_[std::size_t(i) * tokens * width] = __float2half(1.f);
        check(cudaMemcpy(in_, host_in_, in_size() * sizeof(__half), cudaMemcpyHostToDevice));
        const Ort::MemoryInfo gpu{"Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault};
        const std::array<std::int64_t, 3> in_shape{batch, tokens, width};
        const std::array<std::int64_t, 2> logits_shape{batch, tokens};
        const std::array<std::int64_t, 1> value_shape{batch};
        binding_.BindInput("tokens", Ort::Value::CreateTensor(gpu, in_, in_size() * sizeof(__half), in_shape.data(), 3,
                                                           ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
        binding_.BindOutput("value", Ort::Value::CreateTensor<float>(gpu, value_, batch, value_shape.data(), 1));
        binding_.BindOutput("logits", Ort::Value::CreateTensor<float>(gpu, logits_, batch * tokens, logits_shape.data(), 2));
        thread_ = std::jthread{[this](std::stop_token stop) { loop(stop); }};
    }

    ~GpuEvaluator() override {
        thread_.request_stop(); wake_.notify_all(); thread_.join();
        std::fprintf(stderr, "gpu: %ld batches, mean %.0f leaves, %.1f s busy\n", runs_, runs_ ? double(leaves_) / runs_ : 0., busy_);
        cudaFreeHost(host_in_); cudaFreeHost(host_value_); cudaFreeHost(host_logits_);
        cudaFree(in_); cudaFree(value_); cudaFree(logits_);
    }

    std::future<std::vector<Prediction>> submit(std::vector<const Tokens*> leaves) override {
        if (std::ssize(leaves) > batch_) throw std::invalid_argument{"gpu: more leaves than the batch"};
        auto r = std::make_shared<Request>();
        r->leaves = std::move(leaves);
        r->out.resize(r->leaves.size());
        std::vector<const Tokens*> large;
        std::vector<std::size_t> large_at;
        for (std::size_t i = 0; i < r->leaves.size(); ++i)
            if (r->leaves[i]->count > tokens_) { large.push_back(r->leaves[i]); large_at.push_back(i); }
            else r->gpu.push_back(i);
        if (!large.empty()) { // oversized states retain the full CPU input; never truncate tokens
            auto p = cpu_(large);
            for (std::size_t k = 0; k < large.size(); ++k) r->out[large_at[k]] = std::move(p[k]);
        }
        auto future = r->done.get_future();
        if (r->gpu.empty()) { r->done.set_value(std::move(r->out)); return future; }

        // This runs on the submitting CPU worker, not on the GPU thread. Reuse a buffer to avoid per-call allocation.
        {
            std::lock_guard lock{mutex_};
            if (failure_) { r->done.set_exception(failure_); return future; }
            if (!buffers_.empty()) { r->packed = std::move(buffers_.back()); buffers_.pop_back(); }
        }
        const auto stride = std::size_t(tokens_) * width;
        r->packed.resize(r->gpu.size() * stride);
        for (std::size_t k = 0; k < r->gpu.size(); ++k) {
            const auto& t = *r->leaves[r->gpu[k]];
            auto* row = r->packed.data() + k * stride;
            for (std::size_t j = 0; j < t.data.size(); ++j) row[j] = __float2half(t.data[j]);
            std::fill(row + t.data.size(), row + stride, __float2half(0.f));
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
        std::vector<__half> packed;  // already converted and padded by the CPU worker
        std::vector<Prediction> out;
        std::promise<std::vector<Prediction>> done;
    };

    std::size_t in_size() const { return std::size_t(batch_) * tokens_ * width; }

    // Run a full batch immediately, or a partial batch once arrivals have stalled for 1 ms.
    void loop(std::stop_token stop) {
        using namespace std::chrono_literals;
        for (;;) {
            std::vector<std::shared_ptr<Request>> taken;
            {
                std::unique_lock lock{mutex_};
                for (int seen = -1; !stop.stop_requested() && pending_ < batch_ && (pending_ == 0 || pending_ != seen);) {
                    seen = pending_; wake_.wait_for(lock, 1ms);
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

    void run(const std::vector<std::shared_ptr<Request>>& taken) {
        const auto begin = std::chrono::steady_clock::now();
        // Concatenate prepacked chunks into pinned memory. No token conversion or padding on the GPU thread.
        int n = 0;
        for (const auto& r : taken) {
            std::memcpy(host_in_ + std::size_t(n) * tokens_ * width, r->packed.data(), r->packed.size() * sizeof(__half));
            n += int(r->gpu.size());
        }
        check(cudaMemcpy(in_, host_in_, std::size_t(n) * tokens_ * width * sizeof(__half), cudaMemcpyHostToDevice));
        session_.Run(Ort::RunOptions{}, binding_);
        check(cudaMemcpy(host_value_, value_, std::size_t(n) * sizeof(float), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(host_logits_, logits_, std::size_t(n) * tokens_ * sizeof(float), cudaMemcpyDeviceToHost));
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
            for (const auto& r : taken) buffers_.push_back(std::move(r->packed));
        }
        for (const auto& r : taken) r->done.set_value(std::move(r->out));
        ++runs_; leaves_ += n; busy_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    }

    Net cpu_;
    const int batch_, tokens_;
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "nn-gpu"};
    Ort::Session session_{nullptr};
    Ort::IoBinding binding_{nullptr};
    __half *host_in_ = nullptr, *in_ = nullptr;
    float *host_value_ = nullptr, *host_logits_ = nullptr, *value_ = nullptr, *logits_ = nullptr;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::deque<std::shared_ptr<Request>> queue_;
    std::vector<std::vector<__half>> buffers_;
    std::exception_ptr failure_;
    int pending_ = 0;
    long runs_ = 0, leaves_ = 0;
    double busy_ = 0;
    std::jthread thread_;
};
}  // namespace

std::unique_ptr<Evaluator> gpu_evaluator(const std::filesystem::path& onnx, const std::filesystem::path& fixed_onnx,
                                         int batch, int tokens) {
    return std::make_unique<GpuEvaluator>(onnx, fixed_onnx, batch, tokens);
}
}  // namespace stsrl::nn
