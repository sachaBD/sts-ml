#include "infra/workers/inference_service.hpp"
#include "infra/workers/thread_time.hpp"
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace stsrl::infra {
namespace {
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;
struct Ticket { SlotId slot; std::uint64_t serial; };
struct Reply { Ticket ticket; Prediction prediction; };

class CompletionInbox {
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<Reply> replies_;
public:
    void post(std::vector<Reply> replies) {
        {
            std::lock_guard lock{mutex_};
            for (auto& reply : replies) replies_.push_back(std::move(reply));
        }
        ready_.notify_one();
    }
    std::deque<Reply> drain() {
        std::lock_guard lock{mutex_};
        return std::exchange(replies_, {});
    }
    void wait(std::chrono::milliseconds timeout) {
        std::unique_lock lock{mutex_};
        ready_.wait_for(lock, timeout, [&] { return !replies_.empty(); });
    }
};

struct Request {
    CompletionInbox* destination;
    Ticket ticket;
    Input input;
    const std::string* label;
    Clock::time_point enqueued;
};

class RequestInbox {
    enum class State { open, draining, stopped };
    std::atomic<State> state_{State::open};
    mutable std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<Request> pending_;
    std::size_t peak_depth_ = 0;
    long requests_ = 0;
    Clock::duration waited_{};
public:
    Backend& backend;
    const int id, batch;
    RequestInbox(Backend& b, int id, int limit) : backend{b}, id{id}, batch{std::min(limit, b.max_batch())} {}
    void submit(Request request) {
        {
            std::lock_guard lock{mutex_};
            if (state_ != State::open) throw std::runtime_error{"inference service is closed"};
            request.enqueued = Clock::now();
            pending_.push_back(request);
            peak_depth_ = std::max(peak_depth_, pending_.size());
        }
        ready_.notify_one();
    }
    std::vector<Request> gather(std::chrono::microseconds wait) {
        std::vector<Request> out;
        out.reserve(batch);
        std::unique_lock lock{mutex_};
        const auto ready = [&] { return state_ != State::open || !pending_.empty(); };
        ready_.wait_for(lock, 10ms, ready);
        const auto deadline = Clock::now() + wait;
        while (state_ != State::stopped && int(out.size()) < batch) {
            if (pending_.empty()) {
                if (state_ != State::open || !ready_.wait_until(lock, deadline, ready)) break;
                continue;
            }
            out.push_back(pending_.front());
            pending_.pop_front();
        }
        const auto now = Clock::now();
        for (const auto& request : out) waited_ += now - request.enqueued;
        requests_ += long(out.size());
        return out;
    }
private:
    void transition(State next) {
        {
            std::lock_guard lock{mutex_};
            if (state_ != State::stopped) state_ = next;
        }
        ready_.notify_all();
    }
public:
    void close() { transition(State::draining); }
    void stop() { transition(State::stopped); }
    bool stopped() const { return state_ == State::stopped; }
    bool closed() const { return state_ != State::open; }
    QueueCounters counters() const {
        std::lock_guard lock{mutex_};
        return {requests_, waited_, pending_.size(), peak_depth_};
    }
};

struct Lane {
    RequestInbox& queue;
    long batches = 0;
    std::atomic<long> requests{0}, capacity{0};
    std::atomic<Clock::rep> busy{0};
    std::atomic<long long> heartbeat{now_ms()};
    std::atomic<bool> done{false};
    double cpu_seconds = std::nan("");
    explicit Lane(RequestInbox& q) : queue{q} {}
};

void validate(const Request& request, const Prediction& prediction) {
    const auto& tokens = *request.input.tokens;
    bool finite = std::isfinite(prediction.value) && prediction.value >= 0 && prediction.value <= 1;
    for (float x : prediction.logits) finite = finite && std::isfinite(x);
    if (finite && prediction.logits.size() == std::size_t(tokens.actions)) return;
    std::ostringstream message;
    message << "invalid prediction: episode=" << *request.label << " value=" << prediction.value
            << " logits=" << prediction.logits.size() << " actions=" << tokens.actions << " tokens=" << tokens.count;
    throw std::runtime_error{message.str()};
}

void dispatch(Lane& lane, const std::vector<Request>& batch) {
    const auto begun = Clock::now();
    std::vector<Input> inputs;
    inputs.reserve(batch.size());
    for (const auto& request : batch) inputs.push_back(request.input);
    auto predictions = lane.queue.backend.evaluate(inputs);
    if (lane.queue.stopped()) return;
    if (predictions.size() != batch.size()) throw std::runtime_error{"prediction count mismatch"};
    // No payload may be released until every borrowed input in the batch has been validated.
    for (std::size_t i = 0; i < batch.size(); ++i) validate(batch[i], predictions[i]);
    std::map<CompletionInbox*, std::vector<Reply>> replies;
    for (std::size_t i = 0; i < batch.size(); ++i)
        replies[batch[i].destination].push_back({batch[i].ticket, std::move(predictions[i])});
    for (auto& [destination, group] : replies) destination->post(std::move(group));
    ++lane.batches;
    lane.requests += long(batch.size());
    lane.capacity += lane.queue.batch;
    lane.busy += (Clock::now() - begun).count();
}
}  // namespace

struct InferenceService::Impl {
    const std::chrono::microseconds batch_wait;
    std::vector<std::unique_ptr<RequestInbox>> queues;
    std::vector<std::unique_ptr<Lane>> lanes;
    Impl(const std::vector<Backend*>& backends, int batch, std::chrono::microseconds wait) : batch_wait{wait} {
        if (batch < 1 || wait.count() < 0 || backends.empty() || !backends.front())
            throw std::invalid_argument{"invalid actor configuration"};
        for (auto* backend : backends) {
            if (!backend || backend->lanes() < 1 || backend->max_batch() < 1 || backend->width() != backends.front()->width())
                throw std::invalid_argument{"invalid actor configuration"};
            queues.push_back(std::make_unique<RequestInbox>(*backend, int(queues.size()), batch));
            for (int i = 0; i < backend->lanes(); ++i) lanes.push_back(std::make_unique<Lane>(*queues.back()));
        }
    }
};

struct InferenceService::Connection::Impl {
    InferenceService::Impl& service;
    CompletionInbox inbox;
    struct Pending {
        std::uint64_t id;
        Tokens tokens;
        std::vector<std::uint16_t> packed;
        std::string label;
        Clock::time_point submitted = Clock::now();
    };
    std::map<SlotId, std::map<std::uint64_t, Pending>> pending;
    std::uint64_t serial = 0;
    explicit Impl(InferenceService::Impl& s) : service{s} {}
};

InferenceService::InferenceService(const std::vector<Backend*>& backends, int batch, std::chrono::microseconds wait)
    : impl_{std::make_unique<Impl>(backends, batch, wait)} {}
InferenceService::~InferenceService() = default;
InferenceService::Connection::Connection(InferenceService& s) : impl_{std::make_unique<Impl>(*s.impl_)} {}
InferenceService::Connection::~Connection() = default;
InferenceService::Connection InferenceService::connect() { return Connection{*this}; }

void InferenceService::Connection::submit(SlotId slot, EvaluationRequest request, std::string label) {
    const auto& tokens = request.tokens;
    const int width = impl_->service.queues.front()->backend.width();
    if (tokens.count <= 0 || tokens.actions <= 0 || tokens.actions > tokens.count ||
        tokens.data.size() != std::size_t(tokens.count) * width || tokens.attended < 0 || tokens.attended > tokens.count)
        throw std::runtime_error{"invalid tokens from episode " + label};
    const auto serial = ++impl_->serial;
    auto& requests = impl_->pending[slot];
    const auto it = requests.emplace(serial, Impl::Pending{request.id, std::move(request.tokens), {}, std::move(label)}).first;
    auto& pending = it->second;
    try {
        std::size_t i = 0;
        const auto& queues = impl_->service.queues;
        while (i + 1 < queues.size() && !queues[i]->backend.accepts(pending.tokens)) ++i;
        auto& queue = *queues[i];
        queue.backend.prepare(pending.tokens, pending.packed);
        queue.submit({&impl_->inbox, {slot, serial}, {&pending.tokens, pending.packed}, &pending.label, {}});
    } catch (...) { requests.erase(it); throw; }
}

std::vector<Completion> InferenceService::Connection::receive() {
    auto replies = impl_->inbox.drain();
    std::vector<Completion> out;
    out.reserve(replies.size());
    for (auto& reply : replies) {
        auto& requests = impl_->pending.at(reply.ticket.slot);
        const auto it = requests.find(reply.ticket.serial);
        if (it == requests.end()) throw std::runtime_error{"stale or duplicate reply"};
        out.push_back({reply.ticket.slot, it->second.id, std::move(reply.prediction)});
        requests.erase(it);
    }
    return out;
}
bool InferenceService::Connection::pending(SlotId slot) const {
    const auto it = impl_->pending.find(slot);
    return it != impl_->pending.end() && !it->second.empty();
}
std::optional<SlotId> InferenceService::Connection::timed_out(std::chrono::milliseconds timeout) const {
    const auto now = Clock::now();
    for (const auto& [slot, requests] : impl_->pending)
        for (const auto& [serial, request] : requests)
            if (now - request.submitted >= timeout) return slot;
    return std::nullopt;
}
void InferenceService::Connection::wait(std::chrono::milliseconds timeout) { impl_->inbox.wait(timeout); }

int InferenceService::lanes() const { return int(impl_->lanes.size()); }
void InferenceService::run(int index) {
    auto& lane = *impl_->lanes.at(index);
    const double cpu = thread_cpu_seconds();
    const auto finished = [&] { lane.cpu_seconds = thread_cpu_seconds() - cpu; lane.done = true; };
    try {
        while (!lane.queue.stopped()) {
            lane.heartbeat = now_ms();
            const auto batch = lane.queue.gather(impl_->batch_wait);
            if (batch.empty() && lane.queue.closed()) break;
            if (!batch.empty() && !lane.queue.stopped()) dispatch(lane, batch);
        }
    } catch (...) { finished(); throw; }
    finished();
}
void InferenceService::close() { for (auto& queue : impl_->queues) queue->close(); }
void InferenceService::stop() { for (auto& queue : impl_->queues) queue->stop(); }
InferenceService::LaneStatus InferenceService::status(int index) const {
    const auto& lane = *impl_->lanes.at(index);
    return {lane.heartbeat, lane.done};
}
InferenceCounters InferenceService::counters() const {
    InferenceCounters out;
    for (const auto& queue : impl_->queues) out.queues.push_back(queue->counters());
    for (const auto& lane : impl_->lanes)
        out.lanes.push_back({lane->queue.id, lane->requests, lane->capacity, Clock::duration{lane->busy}});
    return out;
}
InferenceService::Totals InferenceService::totals() const {
    Totals out{};
    for (const auto& lane : impl_->lanes) {
        out.batches += lane->batches;
        out.requests += lane->requests;
        out.cpu_seconds += lane->cpu_seconds;
    }
    return out;
}
}  // namespace stsrl::infra
