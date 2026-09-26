#include "models/value_net.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>

namespace stsrl {
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

ValueNet::Embedding embedding(const std::map<std::string, Tensor>& tensors, const std::string& name) {
    const auto& w = get(tensors, name + ".weight", 2);
    return {static_cast<int>(w.shape[0]), static_cast<int>(w.shape[1]), w.data};
}

ValueNet::Linear linear(const std::map<std::string, Tensor>& tensors, const std::string& name, int in, int out) {
    const auto& w = get(tensors, name + ".weight", 2);
    const auto& b = get(tensors, name + ".bias", 1);
    if (w.shape[0] != static_cast<std::uint32_t>(out) || w.shape[1] != static_cast<std::uint32_t>(in)
        || b.shape[0] != static_cast<std::uint32_t>(out))
        throw std::runtime_error{"bad shape for " + name};
    ValueNet::Linear l{in, out, std::vector<float>(static_cast<std::size_t>(in) * out), b.data};
    for (int o = 0; o < out; ++o)
        for (int i = 0; i < in; ++i) l.weight_t[static_cast<std::size_t>(i) * out + o] = w.data[static_cast<std::size_t>(o) * in + i];
    return l;
}

// Appends embedding row `index` to x.
float* put(float* x, const ValueNet::Embedding& e, int index) {
    if (index < 0 || index >= e.rows) throw std::out_of_range{"embedding index out of range"};
    return std::copy_n(e.weight.data() + static_cast<std::size_t>(index) * e.dim, e.dim, x);
}

template <std::size_t N>
float* put(float* x, const std::array<float, N>& values) {
    return std::copy(values.begin(), values.end(), x);
}

// y = W x + b, optionally followed by ReLU. Every output is bias + sum over inputs in input order, one
// multiply and one add each (no FMA), whichever version runs: the results are bit-identical.
// OUT-wide layers accumulate in registers; the AVX2 clones do the same float operations 8 lanes at a time.
template <int OUT>
__attribute__((target_clones("avx2", "default")))
void apply_fixed(const float* __restrict weight_t, const float* __restrict bias, int in, const float* __restrict x,
                 float* __restrict y) {
    float acc[OUT];
    std::memcpy(acc, bias, sizeof acc);
    for (int i = 0; i < in; ++i) {
        const float xi = x[i];
        const float* __restrict w = weight_t + static_cast<std::size_t>(i) * OUT;
        for (int o = 0; o < OUT; ++o) acc[o] += xi * w[o];
    }
    std::memcpy(y, acc, sizeof acc);
}

__attribute__((target_clones("avx2", "default")))
void apply_any(const ValueNet::Linear& l, const float* x, float* y) {
    std::copy(l.bias.begin(), l.bias.end(), y);
    for (int i = 0; i < l.in; ++i) {
        const float xi = x[i];
        const float* w = l.weight_t.data() + static_cast<std::size_t>(i) * l.out;
        for (int o = 0; o < l.out; ++o) y[o] += xi * w[o];
    }
}

void apply(const ValueNet::Linear& l, const float* x, float* y, bool relu) {
    if (l.out == 64) apply_fixed<64>(l.weight_t.data(), l.bias.data(), l.in, x, y);
    else apply_any(l, x, y);
    if (relu)
        for (int o = 0; o < l.out; ++o) y[o] = std::max(y[o], 0.0f);
}

// Two-layer ReLU MLP of the token encoders.
void mlp(const ValueNet::Linear& a, const ValueNet::Linear& b, const float* x, float* hidden, float* y) {
    apply(a, x, hidden, true);
    apply(b, hidden, y, true);
}

}  // namespace

ValueNet::ValueNet(const std::string& path) {
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

std::size_t ValueNet::CardHash::operator()(const CardToken& c) const {
    std::size_t h = std::hash<int>{}(c.card_id);
    const auto mix = [&h](std::size_t x) { h ^= x + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2); };
    mix(static_cast<std::size_t>(c.zone) | static_cast<std::size_t>(c.card_type) << 8
        | static_cast<std::size_t>(c.target_type) << 16);
    for (const float x : c.numeric) mix(std::hash<float>{}(x));
    return h;
}

template <class Key> std::size_t ValueNet::BitsHash::operator()(const Key& k) const {
    std::uint64_t h = 0x9E3779B97F4A7C15ULL;
    for (std::size_t i = 0; i < k.bits.size(); i += 2) {
        std::uint64_t v = k.bits[i] | (i + 1 < k.bits.size() ? std::uint64_t{k.bits[i + 1]} << 32 : 0);
        h = (h ^ v) * 0xbf58476d1ce4e5b9ULL;
        h ^= h >> 31;
    }
    return h;
}

namespace {
template <std::size_t N>
std::uint32_t* put_bits(std::uint32_t* out, const std::array<float, N>& values) {
    for (const float v : values) *out++ = std::bit_cast<std::uint32_t>(v);
    return out;
}
std::uint32_t* put_card(std::uint32_t* out, const CardToken& c) {
    *out++ = static_cast<std::uint32_t>(c.card_id);
    *out++ = static_cast<std::uint32_t>(c.zone) | static_cast<std::uint32_t>(c.card_type) << 8
           | static_cast<std::uint32_t>(c.target_type) << 16;
    return put_bits(out, c.numeric);
}
std::uint32_t* put_monster(std::uint32_t* out, const MonsterToken& m) {
    *out++ = static_cast<std::uint32_t>(m.monster_id);
    *out++ = static_cast<std::uint32_t>(m.move_id);
    return put_bits(out, m.numeric);
}
}  // namespace

const float* ValueNet::monster_hidden(const MonsterToken& monster) const {
    MonsterKey key;
    put_monster(key.bits.data(), monster);
    if (const auto it = monster_cache_.find(key); it != monster_cache_.end()) return it->second.data();
    const std::size_t w = static_cast<std::size_t>(width_);
    put(put(put(x_.data(), monster_id_, monster.monster_id), move_, monster.move_id), monster.numeric);
    std::vector<float> out(w);
    mlp(monster1_, monster2_, x_.data(), hidden_.data(), out.data());
    return monster_cache_.emplace(key, std::move(out)).first->second.data();
}

const float* ValueNet::card_hidden(const CardToken& card) const {
    if (const auto it = card_cache_.find(card); it != card_cache_.end()) return it->second.data();
    std::vector<float> x(card1_.in), hidden(width_), out(width_);
    float* p = put(x.data(), card_id_, card.card_id);
    p = put(p, zone_, static_cast<int>(card.zone));
    p = put(p, card_type_, static_cast<int>(card.card_type));
    put(put(p, target_type_, static_cast<int>(card.target_type)), card.numeric);
    mlp(card1_, card2_, x.data(), hidden.data(), out.data());
    return card_cache_.emplace(card, std::move(out)).first->second.data();
}

float ValueNet::evaluate(const EncodedCombatState& s) const {
    // Not mid-state: card_out_ / monster_out_ point into the caches.
    if (card_cache_.size() >= 1 << 16) card_cache_.clear();
    if (monster_cache_.size() >= 1 << 16) monster_cache_.clear();
    if (interaction_cache_.size() >= 1 << 16) interaction_cache_.clear();
    const std::size_t w = static_cast<std::size_t>(width_);
    x_.resize(std::max<std::size_t>({static_cast<std::size_t>(head1_.in), 2 * w + 6,
                                     static_cast<std::size_t>(monster1_.in)}));
    hidden_.resize(w);
    features_.assign(head1_.in, 0.0f);
    // features: global numeric, two embeddings, then six width-sized sums
    float* sums = put(put(put(features_.data(), s.global.numeric), input_state_, s.global.input_state),
                      card_selection_task_, s.global.card_selection_task);
    float* card_pools = sums;           // zones hand, draw, discard, exhaust (offered cards are not pooled)
    float* monster_sum = sums + 4 * w;
    float* interaction_sum = sums + 5 * w;

    card_out_.resize(s.cards.size());
    for (std::size_t c = 0; c < s.cards.size(); ++c) {
        const auto& card = s.cards[c];
        const float* out = card_out_[c] = card_hidden(card);
        const auto zone = static_cast<std::size_t>(card.zone);
        if (zone < 4)
            for (std::size_t k = 0; k < w; ++k) card_pools[zone * w + k] += out[k];
    }
    monster_out_.resize(s.monsters.size());
    for (std::size_t m = 0; m < s.monsters.size(); ++m) {
        const float* out = monster_out_[m] = monster_hidden(s.monsters[m]);
        for (std::size_t k = 0; k < w; ++k) monster_sum[k] += out[k];
    }
    for (const auto& i : s.card_monster_interactions) {
        if (i.card_index >= s.cards.size() || i.monster_index >= s.monsters.size())
            throw std::out_of_range{"interaction index out of range"};
        InteractionKey key;
        put_bits(put_monster(put_card(key.bits.data(), s.cards[i.card_index]), s.monsters[i.monster_index]), i.numeric);
        auto it = interaction_cache_.find(key);
        if (it == interaction_cache_.end()) {
            float* p = std::copy_n(card_out_[i.card_index], w, x_.data());
            p = std::copy_n(monster_out_[i.monster_index], w, p);
            put(p, i.numeric);
            std::vector<float> out(w);
            mlp(interaction1_, interaction2_, x_.data(), hidden_.data(), out.data());
            it = interaction_cache_.emplace(key, std::move(out)).first;
        }
        const float* out = it->second.data();
        for (std::size_t k = 0; k < w; ++k) interaction_sum[k] += out[k];
    }
    apply(head1_, features_.data(), hidden_.data(), true);
    float value{};
    apply(head2_, hidden_.data(), &value, false);
    return std::tanh(value);
}

void ValueNet::evaluate(std::span<const EncodedCombatState> states, std::vector<float>& values) const {
    values.resize(states.size());
    for (std::size_t i = 0; i < states.size(); ++i) values[i] = evaluate(states[i]);
}

}  // namespace stsrl
