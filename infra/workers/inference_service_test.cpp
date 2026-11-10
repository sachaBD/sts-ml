#include "infra/workers/inference_service.hpp"
#include <gtest/gtest.h>

using namespace stsrl::infra;
using namespace std::chrono_literals;

namespace {
Tokens tokens(float value) { return {std::vector<float>(8, value), 2, 1}; }
class Echo final : public Backend {
public:
    int calls = 0;
    const float* expected_storage = nullptr;
    int width() const override { return 4; }
    int max_batch() const override { return 8; }
    void prepare(const Tokens& t, std::vector<std::uint16_t>& packed) const override {
        packed = {std::uint16_t(t.data[0] * 100)};
    }
    std::vector<Prediction> evaluate(std::span<const Input> inputs) override {
        ++calls;
        std::vector<Prediction> out;
        for (const auto& input : inputs) {
            if (expected_storage) { EXPECT_EQ(input.tokens->data.data(), expected_storage); }
            EXPECT_EQ(input.packed[0], std::uint16_t(input.tokens->data[0] * 100));
            out.push_back({input.tokens->data[0], {0.f}});
        }
        return out;
    }
};
}  // namespace

TEST(InferenceConnection, OwnsMovedPayloadAfterProducerDiesAndReleasesItOnCompletion) {
    Echo backend;
    InferenceService service{{&backend}, 8, 0us};
    auto connection = service.connect();
    const SlotId slot{7};
    {
        auto payload = tokens(0.25f);
        backend.expected_storage = payload.data.data();
        connection.submit(slot, {42, std::move(payload)}, "owned");
    }
    EXPECT_TRUE(connection.pending(slot));
    EXPECT_TRUE(connection.receive().empty());
    service.close();
    service.run(0);
    const auto completions = connection.receive();
    ASSERT_EQ(completions.size(), 1);
    EXPECT_EQ(completions[0].slot, slot);
    EXPECT_EQ(completions[0].id, 42);
    EXPECT_EQ(completions[0].prediction.value, 0.25f);
    EXPECT_FALSE(connection.pending(slot));
    EXPECT_TRUE(connection.receive().empty());
}

TEST(InferenceConnection, RoutesIdenticalLocalIdsToTheirOwnConnectionAndSlot) {
    Echo backend;
    InferenceService service{{&backend}, 8, 0us};
    auto a = service.connect();
    auto b = service.connect();
    a.submit(SlotId{3}, {17, tokens(0.25f)}, "a");
    a.submit(SlotId{5}, {17, tokens(0.5f)}, "a-other-slot");
    b.submit(SlotId{3}, {17, tokens(0.75f)}, "b");
    service.close();
    service.run(0);
    const auto first = a.receive(), second = b.receive();
    ASSERT_EQ(first.size(), 2);
    ASSERT_EQ(second.size(), 1);
    EXPECT_EQ(first[0].slot, SlotId{3});
    EXPECT_EQ(first[0].prediction.value, 0.25f);
    EXPECT_EQ(first[1].slot, SlotId{5});
    EXPECT_EQ(first[1].prediction.value, 0.5f);
    EXPECT_EQ(second[0].slot, SlotId{3});
    EXPECT_EQ(second[0].prediction.value, 0.75f);
}

TEST(InferenceConnection, ClosedServiceRejectsSubmissionWithoutLeavingOutstandingWork) {
    Echo backend;
    InferenceService service{{&backend}, 8, 0us};
    auto connection = service.connect();
    service.close();
    EXPECT_THROW(connection.submit(SlotId{0}, {0, tokens(0.5f)}, "closed"), std::runtime_error);
    EXPECT_FALSE(connection.pending(SlotId{0}));
    service.run(0);
    EXPECT_EQ(backend.calls, 0);
}

TEST(InferenceConnection, StopDoesNotDispatchQueuedWorkOrPretendItCompleted) {
    Echo backend;
    InferenceService service{{&backend}, 8, 0us};
    auto connection = service.connect();
    connection.submit(SlotId{0}, {0, tokens(0.5f)}, "stopped");
    service.stop();
    service.run(0);
    EXPECT_EQ(backend.calls, 0);
    EXPECT_TRUE(connection.pending(SlotId{0}));
    EXPECT_TRUE(connection.receive().empty());
}
