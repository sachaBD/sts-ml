// Verbatim copy of models/value_net.{hpp,cpp} before the search-perf changes (commit ff4e094), renamed
// OriginalValueNet: search_perf_test checks the current ValueNet against it bit for bit.
#pragma once


// Native CPU forward pass of the Deep Sets value net (python/sts_combat_rl/models/deep_sets.py)
// for v3 encodings. Weights come from python/sts_combat_rl/training/export_value_weights.py.
// Single-threaded (not thread-safe: it caches card encodings); plain loops.

#include "combat/encoding.hpp"

#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace stsrl::original {

class OriginalValueNet {
public:
    explicit OriginalValueNet(const std::string& path);

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
    struct CardHash {
        std::size_t operator()(const CardToken& c) const;
    };

    const float* card_hidden(const CardToken& card) const;

    int width_ = 0;
    // Card MLP outputs by token. Leaves of one search share most of their cards, so this
    // skips most card MLP work; cleared when it grows large.
    mutable std::unordered_map<CardToken, std::vector<float>, CardHash> card_cache_;
    Embedding input_state_, card_selection_task_, card_id_, zone_, card_type_, target_type_, monster_id_, move_;
    Linear card1_, card2_, monster1_, monster2_, interaction1_, interaction2_, head1_, head2_;
};

}  // namespace stsrl



#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>

namespace stsrl::original {
namespace {

// File layout (little endian): "STSVNET1", u32 config length, config JSON, u32 tensor count,
// then per tensor: u32 name length, name, u32 ndim, u32 dims[ndim], f32 data (row major).
struct Tensor {
    std::vector<std::uint32_t> shape;
    std::vector<float> data;
};

std::uint32_t read_u32(std::ifstream& in) {
    std::uint32_t x{};
    in.read(reinterpret_cast<char*>(&x), sizeof(x));
    if (!in) throw std::runtime_error{"value net file truncated"};
    return x;
}

std::map<std::string, Tensor> read_tensors(const std::string& path, nlohmann::json& config) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error{"cannot open value net file " + path};
    char magic[8]{};
    in.read(magic, sizeof(magic));
    if (!in || std::memcmp(magic, "STSVNET1", 8) != 0) throw std::runtime_error{"not a value net file: " + path};
    std::string text(read_u32(in), '\0');
    in.read(text.data(), static_cast<std::streamsize>(text.size()));
    config = nlohmann::json::parse(text);
    std::map<std::string, Tensor> tensors;
    for (std::uint32_t n = read_u32(in); n > 0; --n) {
        std::string name(read_u32(in), '\0');
        in.read(name.data(), static_cast<std::streamsize>(name.size()));
        Tensor t;
        std::size_t size = 1;
        for (std::uint32_t d = read_u32(in); d > 0; --d) size *= t.shape.emplace_back(read_u32(in));
        t.data.resize(size);
        in.read(reinterpret_cast<char*>(t.data.data()), static_cast<std::streamsize>(size * sizeof(float)));
        if (!in) throw std::runtime_error{"value net file truncated"};
        tensors.emplace(std::move(name), std::move(t));
    }
    return tensors;
}

const Tensor& get(const std::map<std::string, Tensor>& tensors, const std::string& name, std::size_t ndim) {
    const auto it = tensors.find(name);
    if (it == tensors.end()) throw std::runtime_error{"value net file lacks " + name};
    if (it->second.shape.size() != ndim) throw std::runtime_error{"bad shape for " + name};
    return it->second;
}

OriginalValueNet::Embedding embedding(const std::map<std::string, Tensor>& tensors, const std::string& name) {
    const auto& w = get(tensors, name + ".weight", 2);
    return {static_cast<int>(w.shape[0]), static_cast<int>(w.shape[1]), w.data};
}

OriginalValueNet::Linear linear(const std::map<std::string, Tensor>& tensors, const std::string& name, int in, int out) {
    const auto& w = get(tensors, name + ".weight", 2);
    const auto& b = get(tensors, name + ".bias", 1);
    if (w.shape[0] != static_cast<std::uint32_t>(out) || w.shape[1] != static_cast<std::uint32_t>(in)
        || b.shape[0] != static_cast<std::uint32_t>(out))
        throw std::runtime_error{"bad shape for " + name};
    OriginalValueNet::Linear l{in, out, std::vector<float>(static_cast<std::size_t>(in) * out), b.data};
    for (int o = 0; o < out; ++o)
        for (int i = 0; i < in; ++i) l.weight_t[static_cast<std::size_t>(i) * out + o] = w.data[static_cast<std::size_t>(o) * in + i];
    return l;
}

// Appends embedding row `index` to x.
float* put(float* x, const OriginalValueNet::Embedding& e, int index) {
    if (index < 0 || index >= e.rows) throw std::out_of_range{"embedding index out of range"};
    return std::copy_n(e.weight.data() + static_cast<std::size_t>(index) * e.dim, e.dim, x);
}

template <std::size_t N>
float* put(float* x, const std::array<float, N>& values) {
    return std::copy(values.begin(), values.end(), x);
}

// y = W x + b, optionally followed by ReLU.
void apply(const OriginalValueNet::Linear& l, const float* x, float* y, bool relu) {
    std::copy(l.bias.begin(), l.bias.end(), y);
    for (int i = 0; i < l.in; ++i) {
        const float xi = x[i];
        const float* w = l.weight_t.data() + static_cast<std::size_t>(i) * l.out;
        for (int o = 0; o < l.out; ++o) y[o] += xi * w[o];
    }
    if (relu)
        for (int o = 0; o < l.out; ++o) y[o] = std::max(y[o], 0.0f);
}

// Two-layer ReLU MLP of the token encoders.
void mlp(const OriginalValueNet::Linear& a, const OriginalValueNet::Linear& b, const float* x, float* hidden, float* y) {
    apply(a, x, hidden, true);
    apply(b, hidden, y, true);
}

}  // namespace

OriginalValueNet::OriginalValueNet(const std::string& path) {
    nlohmann::json config;
    const auto t = read_tensors(path, config);
    if (config.at("encoding_version").get<int>() != 3) throw std::runtime_error{"value net is not encoding v3"};
    width_ = config.at("architecture").at("width").get<int>();
    const int w = width_;
    input_state_ = embedding(t, "input_state");
    card_selection_task_ = embedding(t, "card_selection_task");
    card_id_ = embedding(t, "card_id");
    zone_ = embedding(t, "zone");
    card_type_ = embedding(t, "card_type");
    target_type_ = embedding(t, "target_type");
    monster_id_ = embedding(t, "monster_id");
    move_ = embedding(t, "move");
    card1_ = linear(t, "card_mlp.0", card_id_.dim + zone_.dim + card_type_.dim + target_type_.dim + 14, w);
    card2_ = linear(t, "card_mlp.2", w, w);
    monster1_ = linear(t, "monster_mlp.0", monster_id_.dim + move_.dim + 9, w);
    monster2_ = linear(t, "monster_mlp.2", w, w);
    interaction1_ = linear(t, "interaction_mlp.0", 2 * w + 6, w);
    interaction2_ = linear(t, "interaction_mlp.2", w, w);
    head1_ = linear(t, "head.0", 50 + input_state_.dim + card_selection_task_.dim + 6 * w, w);
    head2_ = linear(t, "head.2", w, 1);
}

std::size_t OriginalValueNet::CardHash::operator()(const CardToken& c) const {
    std::size_t h = std::hash<int>{}(c.card_id);
    const auto mix = [&h](std::size_t x) { h ^= x + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2); };
    mix(static_cast<std::size_t>(c.zone) | static_cast<std::size_t>(c.card_type) << 8
        | static_cast<std::size_t>(c.target_type) << 16);
    for (const float x : c.numeric) mix(std::hash<float>{}(x));
    return h;
}

const float* OriginalValueNet::card_hidden(const CardToken& card) const {
    if (const auto it = card_cache_.find(card); it != card_cache_.end()) return it->second.data();
    std::vector<float> x(card1_.in), hidden(width_), out(width_);
    float* p = put(x.data(), card_id_, card.card_id);
    p = put(p, zone_, static_cast<int>(card.zone));
    p = put(p, card_type_, static_cast<int>(card.card_type));
    put(put(p, target_type_, static_cast<int>(card.target_type)), card.numeric);
    mlp(card1_, card2_, x.data(), hidden.data(), out.data());
    return card_cache_.emplace(card, std::move(out)).first->second.data();
}

float OriginalValueNet::evaluate(const EncodedCombatState& s) const {
    if (card_cache_.size() >= 1 << 16) card_cache_.clear();  // not mid-state: `cards` points into it
    const std::size_t w = static_cast<std::size_t>(width_);
    std::vector<float> x(std::max<std::size_t>(head1_.in, 2 * w + 6));
    std::vector<float> hidden(w);
    std::vector<float> features(head1_.in, 0.0f);
    // features: global numeric, two embeddings, then six width-sized sums
    float* sums = put(put(put(features.data(), s.global.numeric), input_state_, s.global.input_state),
                      card_selection_task_, s.global.card_selection_task);
    float* card_pools = sums;           // zones hand, draw, discard, exhaust (offered cards are not pooled)
    float* monster_sum = sums + 4 * w;
    float* interaction_sum = sums + 5 * w;

    std::vector<const float*> cards(s.cards.size());
    for (std::size_t c = 0; c < s.cards.size(); ++c) {
        const auto& card = s.cards[c];
        const float* out = cards[c] = card_hidden(card);
        const auto zone = static_cast<std::size_t>(card.zone);
        if (zone < 4)
            for (std::size_t k = 0; k < w; ++k) card_pools[zone * w + k] += out[k];
    }
    std::vector<float> monsters(s.monsters.size() * w);
    for (std::size_t m = 0; m < s.monsters.size(); ++m) {
        const auto& monster = s.monsters[m];
        put(put(put(x.data(), monster_id_, monster.monster_id), move_, monster.move_id), monster.numeric);
        float* out = monsters.data() + m * w;
        mlp(monster1_, monster2_, x.data(), hidden.data(), out);
        for (std::size_t k = 0; k < w; ++k) monster_sum[k] += out[k];
    }
    std::vector<float> out(w);
    for (const auto& i : s.card_monster_interactions) {
        if (i.card_index >= s.cards.size() || i.monster_index >= s.monsters.size())
            throw std::out_of_range{"interaction index out of range"};
        float* p = std::copy_n(cards[i.card_index], w, x.data());
        p = std::copy_n(monsters.data() + i.monster_index * w, w, p);
        put(p, i.numeric);
        mlp(interaction1_, interaction2_, x.data(), hidden.data(), out.data());
        for (std::size_t k = 0; k < w; ++k) interaction_sum[k] += out[k];
    }
    apply(head1_, features.data(), hidden.data(), true);
    float value{};
    apply(head2_, hidden.data(), &value, false);
    return std::tanh(value);
}

void OriginalValueNet::evaluate(std::span<const EncodedCombatState> states, std::vector<float>& values) const {
    values.resize(states.size());
    for (std::size_t i = 0; i < states.size(); ++i) values[i] = evaluate(states[i]);
}

}  // namespace stsrl
