#include "agents/combat/value/value_net.hpp"

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

// A Linear without bias (PyTorch bias=False): zero bias.
ValueNet::Linear linear_no_bias(const std::map<std::string, Tensor>& tensors, const std::string& name, int in, int out) {
    const auto& w = get(tensors, name + ".weight", 2);
    if (w.shape[0] != static_cast<std::uint32_t>(out) || w.shape[1] != static_cast<std::uint32_t>(in))
        throw std::runtime_error{"bad shape for " + name};
    if (tensors.contains(name + ".bias")) throw std::runtime_error{name + " should have no bias"};
    ValueNet::Linear l{in, out, std::vector<float>(static_cast<std::size_t>(in) * out), std::vector<float>(out, 0.f)};
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

// y = W x + b, optionally followed by ReLU. Every output is bias + sum over inputs in input order, as
// in the original loop, so the results are bit-identical to it (this file is built with -march=native,
// see CMakeLists.txt). OUT-wide layers accumulate in registers.
template <int OUT>
__attribute__((noinline)) void apply_fixed(const float* __restrict weight_t, const float* __restrict bias, int in, const float* __restrict x,
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

// apply_fixed over OUT-wide column blocks of a wider layer (out a multiple of OUT): same per-output
// arithmetic, with the accumulators of one block in registers.
template <int OUT>
__attribute__((noinline)) void apply_blocked(const float* __restrict weight_t, const float* __restrict bias, int in, int out,
                                             const float* __restrict x, float* __restrict y) {
    for (int b = 0; b < out; b += OUT) {
        float acc[OUT];
        std::memcpy(acc, bias + b, sizeof acc);
        for (int i = 0; i < in; ++i) {
            const float xi = x[i];
            const float* __restrict w = weight_t + static_cast<std::size_t>(i) * out + b;
            for (int o = 0; o < OUT; ++o) acc[o] += xi * w[o];
        }
        std::memcpy(y + b, acc, sizeof acc);
    }
}

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
    else if (l.out % 64 == 0) apply_blocked<64>(l.weight_t.data(), l.bias.data(), l.in, l.out, x, y);
    else apply_any(l, x, y);
    if (relu)
        for (int o = 0; o < l.out; ++o) y[o] = std::max(y[o], 0.0f);
}

// Two-layer ReLU MLP of the token encoders.
void mlp(const ValueNet::Linear& a, const ValueNet::Linear& b, const float* x, float* hidden, float* y) {
    apply(a, x, hidden, true);
    apply(b, hidden, y, true);
}

ValueNet::LayerNorm layer_norm(const std::map<std::string, Tensor>& tensors, const std::string& name, int size) {
    const auto& w = get(tensors, name + ".weight", 1);
    const auto& b = get(tensors, name + ".bias", 1);
    if (w.shape[0] != static_cast<std::uint32_t>(size) || b.shape[0] != static_cast<std::uint32_t>(size))
        throw std::runtime_error{"bad shape for " + name};
    return {size, w.data, b.data};
}

// PyTorch LayerNorm: biased variance, eps 1e-5, elementwise affine. y may alias x.
void layer_norm(const ValueNet::LayerNorm& n, const float* x, float* y) {
    double mean = 0, var = 0;
    for (int i = 0; i < n.size; ++i) mean += x[i];
    mean /= n.size;
    for (int i = 0; i < n.size; ++i) var += (x[i] - mean) * (x[i] - mean);
    const double inv = 1.0 / std::sqrt(var / n.size + 1e-5);
    for (int i = 0; i < n.size; ++i)
        y[i] = static_cast<float>((x[i] - mean) * inv) * n.weight[i] + n.bias[i];
}

}  // namespace

ValueNet::ValueNet(const std::string& path) {
    nlohmann::json config;
    const auto t = read_tensors(path, config);
    const auto& architecture = config.at("architecture");
    const auto kind = architecture.at("kind").get<std::string>();
    if (kind != "deep_sets_v1" && kind != "deep_sets_v2" && kind != "deep_sets_v3")
        throw std::runtime_error{"unknown value net kind " + kind};
    v3_ = kind == "deep_sets_v3";
    if (config.at("encoding_version").get<int>() != (v3_ ? 4 : 3))
        throw std::runtime_error{"value net " + kind + " has the wrong encoding version"};
    width_ = architecture.at("width").get<int>();
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
    monster1_ = linear(t, "monster_mlp.0", monster_id_.dim + (v3_ ? 2 : 1) * move_.dim + 9
                                                 + (v3_ ? static_cast<int>(monster_status_features) : 0), w);
    monster2_ = linear(t, "monster_mlp.2", w, w);
    interaction1_ = linear(t, "interaction_mlp.0", 2 * w + 6, w);
    interaction2_ = linear(t, "interaction_mlp.2", w, w);
    if (kind == "deep_sets_v1") {
        head1_ = linear(t, "head.0", 50 + input_state_.dim + card_selection_task_.dim + 6 * w, w);
        head2_ = linear(t, "head.2", w, 1);
        return;
    }
    v2_ = true;
    count_features_ = architecture.at("pool_count_features").get<bool>();
    input_norm_ = architecture.at("head_input_norm").get<bool>();
    if (v3_) {
        potion_id_ = embedding(t, "potion_id");
        relic_id_ = embedding(t, "relic_id");
        potion1_ = linear(t, "potion_mlp.0", potion_id_.dim + static_cast<int>(potion_features), w);
        potion2_ = linear(t, "potion_mlp.2", w, w);
        relic1_ = linear(t, "relic_mlp.0", relic_id_.dim + static_cast<int>(relic_features), w);
        relic2_ = linear(t, "relic_mlp.2", w, w);
        score_hp_offset_ = architecture.at("score_hp_offset").get<float>();
        score_potion_hp_ = architecture.at("score_potion_hp").get<float>();
        score_max_hp_offset_ = architecture.at("score_max_hp_offset").get<float>();
    } else {
        const auto output = architecture.at("output").get<std::string>();
        if (output != "sigmoid" && output != "tanh") throw std::runtime_error{"unknown value net output " + output};
        sigmoid_ = output == "sigmoid";
    }
    const int hw = architecture.at("head_width").get<int>();
    const int blocks = architecture.at("head_blocks").get<int>();
    const int pools = v3_ ? 8 : 6;
    const int features = 50 + (v3_ ? static_cast<int>(player_v4_features) : 0) + input_state_.dim
                       + card_selection_task_.dim + pools * w + (count_features_ ? pools : 0);
    if (input_norm_) head_in_norm_ = layer_norm(t, "head_in_norm", features);
    head_in_ = linear(t, "head_in", features, hw);
    for (int b = 0; b < blocks; ++b) {
        const auto name = "head_blocks." + std::to_string(b);
        head_blocks_.push_back({layer_norm(t, name + ".norm", hw), linear(t, name + ".fc1", hw, hw),
                                linear(t, name + ".fc2", hw, hw)});
    }
    if (blocks > 0) head_out_norm_ = layer_norm(t, "head_out_norm", hw);
    if (v3_) {
        won_out_ = linear(t, "won_out", hw, 1);
        hp_out_ = linear(t, "hp_out", hw, 1);
        keep_out_ = linear(t, "keep_out", hw, 1);
        policy_width_ = architecture.at("policy_width").get<int>();
        if (policy_width_ > 0) {
            action_kind_ = embedding(t, "action_kind");
            policy_state_ = linear(t, "policy_state", hw, policy_width_);
            policy_action_ = linear_no_bias(t, "policy_action",
                                            action_kind_.dim + card_selection_task_.dim + 4 * w + 6, policy_width_);
            policy_out_ = linear(t, "policy_out", policy_width_, 1);
        }
    } else {
        value_out_ = linear(t, "value_out", hw, 1);
    }
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
    *out++ = static_cast<std::uint32_t>(m.previous_move_id);
    return put_bits(put_bits(out, m.numeric), m.status);
}
}  // namespace

const float* ValueNet::monster_hidden(const MonsterToken& monster) const {
    MonsterKey key;
    put_monster(key.bits.data(), monster);
    if (const auto it = monster_cache_.find(key); it != monster_cache_.end()) {
        ++stats_.monster_hits;
        return it->second.data();
    }
    ++stats_.monster_misses;
    const std::size_t w = static_cast<std::size_t>(width_);
    float* p = put(put(x_.data(), monster_id_, monster.monster_id), move_, monster.move_id);
    if (v3_) p = put(p, move_, monster.previous_move_id);
    p = put(p, monster.numeric);
    if (v3_) put(p, monster.status);
    std::vector<float> out(w);
    mlp(monster1_, monster2_, x_.data(), hidden_.data(), out.data());
    return monster_cache_.emplace(key, std::move(out)).first->second.data();
}

const float* ValueNet::card_hidden(const CardToken& card) const {
    CardKey key;
    put_card(key.bits.data(), card);
    if (const auto it = card_cache_.find(key); it != card_cache_.end()) {
        ++stats_.card_hits;
        return it->second.data();
    }
    ++stats_.card_misses;
    std::vector<float> x(card1_.in), hidden(width_), out(width_);
    float* p = put(x.data(), card_id_, card.card_id);
    p = put(p, zone_, static_cast<int>(card.zone));
    p = put(p, card_type_, static_cast<int>(card.card_type));
    put(put(p, target_type_, static_cast<int>(card.target_type)), card.numeric);
    mlp(card1_, card2_, x.data(), hidden.data(), out.data());
    return card_cache_.emplace(key, std::move(out)).first->second.data();
}

const float* ValueNet::potion_hidden(const PotionToken& potion) const {
    PotionKey key;
    key.bits[0] = static_cast<std::uint32_t>(potion.potion_id);
    put_bits(key.bits.data() + 1, potion.numeric);
    if (const auto it = potion_cache_.find(key); it != potion_cache_.end()) {
        ++stats_.potion_hits;
        return it->second.data();
    }
    ++stats_.potion_misses;
    std::vector<float> x(potion1_.in), out(width_);
    put(put(x.data(), potion_id_, potion.potion_id), potion.numeric);
    mlp(potion1_, potion2_, x.data(), hidden_.data(), out.data());
    return potion_cache_.emplace(key, std::move(out)).first->second.data();
}

const float* ValueNet::relic_hidden(const RelicToken& relic) const {
    RelicKey key;
    key.bits[0] = static_cast<std::uint32_t>(relic.relic_id);
    put_bits(key.bits.data() + 1, relic.numeric);
    if (const auto it = relic_cache_.find(key); it != relic_cache_.end()) {
        ++stats_.relic_hits;
        return it->second.data();
    }
    ++stats_.relic_misses;
    std::vector<float> x(relic1_.in), out(width_);
    put(put(x.data(), relic_id_, relic.relic_id), relic.numeric);
    mlp(relic1_, relic2_, x.data(), hidden_.data(), out.data());
    return relic_cache_.emplace(key, std::move(out)).first->second.data();
}

const float* ValueNet::interaction_hidden(const CardToken& card, const float* card_out, const MonsterToken& monster,
                                          const float* monster_out, const std::array<float, 6>& numeric) const {
    InteractionKey key;
    put_bits(put_monster(put_card(key.bits.data(), card), monster), numeric);
    auto it = interaction_cache_.find(key);
    ++(it == interaction_cache_.end() ? stats_.interaction_misses : stats_.interaction_hits);
    if (it == interaction_cache_.end()) {
        const std::size_t w = static_cast<std::size_t>(width_);
        std::vector<float> x(interaction1_.in), hidden(w), out(w);
        float* p = std::copy_n(card_out, w, x.data());
        p = std::copy_n(monster_out, w, p);
        put(p, numeric);
        mlp(interaction1_, interaction2_, x.data(), hidden.data(), out.data());
        it = interaction_cache_.emplace(key, std::move(out)).first;
    }
    return it->second.data();
}

const float* ValueNet::action_hidden(const ActionToken& a) const {
    ActionKey key;
    auto* k = key.bits.data();
    *k++ = static_cast<std::uint32_t>(a.kind);
    *k++ = static_cast<std::uint32_t>(a.card_selection_task);
    *k++ = std::uint32_t(a.skips_selection) | std::uint32_t(a.source_card.has_value()) << 1
         | std::uint32_t(a.target_monster.has_value()) << 2 | std::uint32_t(a.potion.has_value()) << 3
         | std::uint32_t(a.interaction.has_value()) << 4 | std::uint32_t(a.discards_potion) << 5;
    *k++ = 0;
    k = a.source_card ? put_card(k, *a.source_card) : k + 16;
    k = a.target_monster ? put_monster(k, *a.target_monster) : k + monster_key_words;
    if (a.potion) {
        *k++ = static_cast<std::uint32_t>(a.potion->potion_id);
        k = put_bits(k, a.potion->numeric);
    } else {
        k += 1 + potion_features;
    }
    if (a.interaction) put_bits(k, *a.interaction);
    if (const auto it = action_cache_.find(key); it != action_cache_.end()) {
        ++stats_.action_hits;
        return it->second.data();
    }
    ++stats_.action_misses;
    // a = cat(emb(kind), emb(task), card, monster, potion, interaction, flags); absent parts zero
    const std::size_t w = static_cast<std::size_t>(width_);
    action_x_.assign(static_cast<std::size_t>(policy_action_.in), 0.0f);
    float* x = put(put(action_x_.data(), action_kind_, static_cast<int>(a.kind)), card_selection_task_,
                   a.card_selection_task);
    const float* card = a.source_card ? card_hidden(*a.source_card) : nullptr;
    const float* monster = a.target_monster ? monster_hidden(*a.target_monster) : nullptr;
    if (card) std::copy_n(card, w, x);
    if (monster) std::copy_n(monster, w, x + w);
    if (a.potion) std::copy_n(potion_hidden(*a.potion), w, x + 2 * w);
    if (a.interaction) {
        // (Python multiplies the absent card / monster parts by 0, so an interaction needs both.)
        if (!card || !monster) throw std::runtime_error{"action interaction without card and target"};
        std::copy_n(interaction_hidden(*a.source_card, card, *a.target_monster, monster, *a.interaction), w, x + 3 * w);
    }
    float* flags = x + 4 * w;
    flags[0] = float(a.skips_selection);
    flags[1] = float(a.discards_potion);
    flags[2] = float(a.source_card.has_value());
    flags[3] = float(a.target_monster.has_value());
    flags[4] = float(a.potion.has_value());
    flags[5] = float(a.interaction.has_value());
    std::vector<float> out(static_cast<std::size_t>(policy_width_));
    apply(policy_action_, action_x_.data(), out.data(), false);
    return action_cache_.emplace(key, std::move(out)).first->second.data();
}

void ValueNet::pool_features(const EncodedCombatState& s, std::size_t size, bool counts) const {
    // Not mid-state: card_out_ / monster_out_ point into the caches.
    if (card_cache_.size() >= 1 << 16) card_cache_.clear();
    if (monster_cache_.size() >= 1 << 16) monster_cache_.clear();
    if (interaction_cache_.size() >= 1 << 16) interaction_cache_.clear();
    if (potion_cache_.size() >= 1 << 12) potion_cache_.clear();
    if (relic_cache_.size() >= 1 << 12) relic_cache_.clear();
    if (action_cache_.size() >= 1 << 16) action_cache_.clear();
    const std::size_t w = static_cast<std::size_t>(width_);
    const std::size_t pools = v3_ ? 8 : 6;
    x_.resize(std::max<std::size_t>({size, 2 * w + 6, static_cast<std::size_t>(monster1_.in)}));
    hidden_.resize(w);
    features_.assign(size, 0.0f);
    // features: global numeric, (v3: player numeric,) two embeddings, then the width-sized sums
    float* sums = put(features_.data(), s.global.numeric);
    if (v3_) sums = put(sums, s.global.player);
    sums = put(put(sums, input_state_, s.global.input_state), card_selection_task_, s.global.card_selection_task);
    float* card_pools = sums;           // zones hand, draw, discard, exhaust (offered cards are not pooled)
    float* monster_sum = sums + 4 * w;
    float* interaction_sum = sums + 5 * w;

    std::array<int, 8> count{};
    card_out_.resize(s.cards.size());
    for (std::size_t c = 0; c < s.cards.size(); ++c) {
        const auto& card = s.cards[c];
        const float* out = card_out_[c] = card_hidden(card);
        const auto zone = static_cast<std::size_t>(card.zone);
        if (zone < 4) {
            ++count[zone];
            for (std::size_t k = 0; k < w; ++k) card_pools[zone * w + k] += out[k];
        }
    }
    monster_out_.resize(s.monsters.size());
    for (std::size_t m = 0; m < s.monsters.size(); ++m) {
        const float* out = monster_out_[m] = monster_hidden(s.monsters[m]);
        for (std::size_t k = 0; k < w; ++k) monster_sum[k] += out[k];
    }
    for (const auto& i : s.card_monster_interactions) {
        if (i.card_index >= s.cards.size() || i.monster_index >= s.monsters.size())
            throw std::out_of_range{"interaction index out of range"};
        const float* out = interaction_hidden(s.cards[i.card_index], card_out_[i.card_index],
                                              s.monsters[i.monster_index], monster_out_[i.monster_index], i.numeric);
        for (std::size_t k = 0; k < w; ++k) interaction_sum[k] += out[k];
    }
    if (v3_) {
        float* potion_sum = sums + 6 * w;
        float* relic_sum = sums + 7 * w;
        for (const auto& potion : s.potions) {
            const float* out = potion_hidden(potion);
            for (std::size_t k = 0; k < w; ++k) potion_sum[k] += out[k];
        }
        for (const auto& relic : s.relics) {
            const float* out = relic_hidden(relic);
            for (std::size_t k = 0; k < w; ++k) relic_sum[k] += out[k];
        }
    }
    if (counts) {
        count[4] = static_cast<int>(s.monsters.size());
        count[5] = static_cast<int>(s.card_monster_interactions.size());
        count[6] = static_cast<int>(s.potions.size());
        count[7] = static_cast<int>(s.relics.size());
        for (std::size_t p = 0; p < pools; ++p) sums[pools * w + p] = std::log1p(static_cast<float>(count[p]));
    }
}

float ValueNet::evaluate(const EncodedCombatState& s) const {
    if (v3_) return evaluate_v3(s);
    if (v2_) return evaluate_v2(s);
    pool_features(s, static_cast<std::size_t>(head1_.in), false);
    apply(head1_, features_.data(), hidden_.data(), true);
    float value{};
    apply(head2_, hidden_.data(), &value, false);
    return std::tanh(value);
}

void ValueNet::head(const EncodedCombatState& s) const {
    pool_features(s, static_cast<std::size_t>(head_in_.in), count_features_);
    const std::size_t hw = static_cast<std::size_t>(head_in_.out);
    h_.resize(hw);
    normed_.resize(std::max<std::size_t>(hw, features_.size()));
    block_hidden_.resize(hw);
    const float* f = features_.data();
    if (input_norm_) {
        layer_norm(head_in_norm_, f, normed_.data());
        f = normed_.data();
    }
    apply(head_in_, f, h_.data(), true);
    for (const auto& b : head_blocks_) {
        layer_norm(b.norm, h_.data(), normed_.data());
        apply(b.fc1, normed_.data(), block_hidden_.data(), true);
        apply(b.fc2, block_hidden_.data(), normed_.data(), false);
        for (std::size_t k = 0; k < hw; ++k) h_[k] += normed_[k];
    }
    if (!head_blocks_.empty()) layer_norm(head_out_norm_, h_.data(), h_.data());
}

float ValueNet::evaluate_v2(const EncodedCombatState& s) const {
    head(s);
    float value{};
    apply(value_out_, h_.data(), &value, false);
    return sigmoid_ ? 1.0f / (1.0f + std::exp(-value)) : std::tanh(value);
}

float ValueNet::evaluate_v3(const EncodedCombatState& s) const {
    head(s);
    const auto sigmoid = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
    float won{}, hp{}, keep{};
    apply(won_out_, h_.data(), &won, false);
    apply(hp_out_, h_.data(), &hp, false);
    apply(keep_out_, h_.data(), &keep, false);
    const float max_hp = s.global.max_hp, potions = static_cast<float>(s.potions.size());
    return sigmoid(won) * (score_hp_offset_ + sigmoid(hp) * max_hp + score_potion_hp_ * sigmoid(keep) * potions)
         / (score_max_hp_offset_ + max_hp);
}

float ValueNet::evaluate(const EncodedCombatState& s, std::span<const ActionToken> actions,
                         std::vector<float>& logits) const {
    if (!has_policy()) throw std::logic_error{"value net has no policy head"};
    const float value = evaluate_v3(s);  // leaves h_
    const std::size_t pw = static_cast<std::size_t>(policy_width_);
    policy_h_.resize(pw);
    policy_hidden_.resize(pw);
    apply(policy_state_, h_.data(), policy_h_.data(), false);
    logits.resize(actions.size());
    for (std::size_t i = 0; i < actions.size(); ++i) {
        const float* a = action_hidden(actions[i]);
        for (std::size_t k = 0; k < pw; ++k) policy_hidden_[k] = std::max(policy_h_[k] + a[k], 0.0f);
        apply(policy_out_, policy_hidden_.data(), &logits[i], false);
    }
    return value;
}

void ValueNet::evaluate(std::span<const EncodedCombatState> states, std::vector<float>& values) const {
    values.resize(states.size());
    for (std::size_t i = 0; i < states.size(); ++i) values[i] = evaluate(states[i]);
}

}  // namespace stsrl
