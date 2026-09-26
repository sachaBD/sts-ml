#pragma once

// Native CPU forward pass of the Deep Sets value net (python/sts_combat_rl/models/deep_sets.py)
// for v3 encodings. Weights come from python/sts_combat_rl/training/export_value_weights.py.
// Single-threaded (not thread-safe: it caches card encodings); plain loops.

#include "combat/encoding.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace stsrl {

class ValueNet {
public:
    explicit ValueNet(const std::string& path);

    // One value per state, written to `values` (resized to states.size()).
    void evaluate(std::span<const EncodedCombatState> states, std::vector<float>& values) const;
    float evaluate(const EncodedCombatState& state) const;

    struct Embedding {
        int rows = 0, dim = 0;
        std::vector<float> weight;  // rows x dim
    };
    struct Linear {
        int in = 0, out = 0;
        std::vector<float> weight_t;  // in x out (transposed from PyTorch's out x in)
        std::vector<float> bias;      // out
    };

private:
    // Monster MLP / interaction MLP inputs, compared bit for bit (a cached output is exactly what the MLP
    // would compute again).
    struct CardKey {
        std::array<std::uint32_t, 16> bits{};
        bool operator==(const CardKey&) const = default;
    };
    struct MonsterKey {
        std::array<std::uint32_t, 11> bits{};
        bool operator==(const MonsterKey&) const = default;
    };
    struct InteractionKey {
        std::array<std::uint32_t, 18 + 11 + 6> bits{};
        bool operator==(const InteractionKey&) const = default;
    };
    struct BitsHash {
        template <class Key> std::size_t operator()(const Key& k) const;
    };

    const float* card_hidden(const CardToken& card) const;
    const float* monster_hidden(const MonsterToken& monster) const;

    int width_ = 0;
    // Card MLP outputs by token. Leaves of one search share most of their cards, so this
    // skips most card MLP work; cleared when it grows large.
    mutable std::unordered_map<CardKey, std::vector<float>, BitsHash> card_cache_;
    mutable std::unordered_map<MonsterKey, std::vector<float>, BitsHash> monster_cache_;
    mutable std::unordered_map<InteractionKey, std::vector<float>, BitsHash> interaction_cache_;
    // Scratch buffers of evaluate (no allocation per state).
    mutable std::vector<float> x_, hidden_, features_, out_;
    mutable std::vector<const float*> card_out_, monster_out_;
    Embedding input_state_, card_selection_task_, card_id_, zone_, card_type_, target_type_, monster_id_, move_;
    Linear card1_, card2_, monster1_, monster2_, interaction1_, interaction2_, head1_, head2_;
};

}  // namespace stsrl
