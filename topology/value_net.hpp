#pragma once

// Native CPU forward pass of the value net topologies (python/sts_combat_rl/topology/).
// Weights come from python/sts_combat_rl/training/export_value_weights.py.
// Single-threaded (not thread-safe: it caches token encodings); plain loops.
// Three kinds (config architecture.kind, required):
//   "deep_sets_v1" (encoding v3; DeepSetsV1, one hidden head layer, tanh),
//   "deep_sets_v2" (encoding v3; DeepSetsV2: configurable widths and embeddings, log1p pool counts,
//     LayerNorm'd residual head, sigmoid or tanh; its auxiliary won_out / hp_out heads are ignored),
//   "deep_sets_v3" (encoding v4; DeepSetsV3: v2's trunk plus monster status / previous move, player v4
//     numeric, potion and relic pools; output = the terminal score formula over its won / hp / keep heads).
// The token encoders are context-free MLPs, so card / monster / interaction / potion / relic outputs are cached.

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

    // Token cache lookups since construction (hits, misses) for card, monster and interaction encoders.
    struct CacheStats {
        std::uint64_t card_hits = 0, card_misses = 0, monster_hits = 0, monster_misses = 0,
                      interaction_hits = 0, interaction_misses = 0, potion_hits = 0, potion_misses = 0,
                      relic_hits = 0, relic_misses = 0;
    };
    const CacheStats& cache_stats() const { return stats_; }

    struct Embedding {
        int rows = 0, dim = 0;
        std::vector<float> weight;  // rows x dim
    };
    struct Linear {
        int in = 0, out = 0;
        std::vector<float> weight_t;  // in x out (transposed from PyTorch's out x in)
        std::vector<float> bias;      // out
    };
    struct LayerNorm {
        int size = 0;
        std::vector<float> weight, bias;
    };
    struct HeadBlock {
        LayerNorm norm;
        Linear fc1, fc2;
    };

private:
    // Monster MLP / interaction MLP inputs, compared bit for bit (a cached output is exactly what the MLP
    // would compute again).
    struct CardKey {
        std::array<std::uint32_t, 16> bits{};
        bool operator==(const CardKey&) const = default;
    };
    // Every monster token field (v4 ones too, so a key is exact for all kinds).
    static constexpr std::size_t monster_key_words = 3 + 9 + monster_status_features;
    struct MonsterKey {
        std::array<std::uint32_t, monster_key_words> bits{};
        bool operator==(const MonsterKey&) const = default;
    };
    struct InteractionKey {
        std::array<std::uint32_t, 16 + monster_key_words + 6> bits{};
        bool operator==(const InteractionKey&) const = default;
    };
    struct PotionKey {
        std::array<std::uint32_t, 1 + potion_features> bits{};
        bool operator==(const PotionKey&) const = default;
    };
    struct RelicKey {
        std::array<std::uint32_t, 1 + relic_features> bits{};
        bool operator==(const RelicKey&) const = default;
    };
    struct BitsHash {
        template <class Key> std::size_t operator()(const Key& k) const;
    };

    const float* card_hidden(const CardToken& card) const;
    const float* monster_hidden(const MonsterToken& monster) const;
    const float* potion_hidden(const PotionToken& potion) const;
    const float* relic_hidden(const RelicToken& relic) const;
    // Fills features_ (size `size`): global numeric, (v3: player numeric,) the two embeddings, the width-sized
    // pools (six; v3: eight) and, if counts, log1p of the pool token counts.
    void pool_features(const EncodedCombatState& s, std::size_t size, bool counts) const;
    // v2 / v3 head up to h_ (after head_out_norm).
    void head(const EncodedCombatState& s) const;
    float evaluate_v2(const EncodedCombatState& s) const;
    float evaluate_v3(const EncodedCombatState& s) const;

    int width_ = 0;
    // Card MLP outputs by token. Leaves of one search share most of their cards, so this
    // skips most card MLP work; cleared when it grows large.
    mutable std::unordered_map<CardKey, std::vector<float>, BitsHash> card_cache_;
    mutable std::unordered_map<MonsterKey, std::vector<float>, BitsHash> monster_cache_;
    mutable std::unordered_map<InteractionKey, std::vector<float>, BitsHash> interaction_cache_;
    mutable std::unordered_map<PotionKey, std::vector<float>, BitsHash> potion_cache_;
    mutable std::unordered_map<RelicKey, std::vector<float>, BitsHash> relic_cache_;
    // Scratch buffers of evaluate (no allocation per state).
    mutable std::vector<float> x_, hidden_, features_, out_;
    mutable std::vector<const float*> card_out_, monster_out_;
    Embedding input_state_, card_selection_task_, card_id_, zone_, card_type_, target_type_, monster_id_, move_;
    Linear card1_, card2_, monster1_, monster2_, interaction1_, interaction2_, head1_, head2_;
    // v2 / v3 head (v2_ for both; v3_ for deep_sets_v3 only)
    bool v2_ = false, v3_ = false, count_features_ = false, input_norm_ = false, sigmoid_ = false;
    LayerNorm head_in_norm_, head_out_norm_;
    Linear head_in_, value_out_;
    // v3 only
    Embedding potion_id_, relic_id_;
    Linear potion1_, potion2_, relic1_, relic2_, won_out_, hp_out_, keep_out_;
    float score_hp_offset_ = 0, score_potion_hp_ = 0, score_max_hp_offset_ = 0;
    std::vector<HeadBlock> head_blocks_;
    mutable std::vector<float> h_, normed_, block_hidden_;
    mutable CacheStats stats_;
};

}  // namespace stsrl
