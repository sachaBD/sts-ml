#include "agents/combat/player/play_config.hpp"
#include <set>
#include <stdexcept>
#include <string>
#include <toml++/toml.hpp>

namespace stsrl::player {
namespace {
// One table's settings: every key is required, and a key never read is a mistake.
class Table {
public:
    Table(const toml::table& parent, std::string name) : name_{std::move(name)} {
        const auto* table = parent[name_].as_table();
        if (!table) throw std::invalid_argument{"missing [" + name_ + "]"};
        table_ = table;
    }

    template <class T> T get(const std::string& key) {
        read_.insert(key);
        const auto node = (*table_)[key];
        if (!node) throw std::invalid_argument{"missing [" + name_ + "]." + key};
        std::optional<T> value;
        if constexpr (std::is_integral_v<T>) {
            if (const auto exact = node.value_exact<std::int64_t>(); exact && T(*exact) == *exact) value = T(*exact);
        } else {
            value = node.value<T>();  // an integer is also a float
        }
        if (!value) throw std::invalid_argument{"[" + name_ + "]." + key + " has the wrong type"};
        return *value;
    }

    bool has(const std::string& key) const { return bool((*table_)[key]); }

    void done() const {
        for (const auto& [key, _] : *table_)
            if (!read_.contains(std::string{key.str()}))
                throw std::invalid_argument{"unknown [" + name_ + "]." + std::string{key.str()}};
    }

private:
    std::string name_;
    const toml::table* table_;
    std::set<std::string> read_;
};

PlayConfig parse(const toml::table& root) {
    PlayConfig c;
    Table search{root, "search"};
    c.search.sims = search.get<int>("sims");
    c.search.particles = search.get<int>("particles");
    c.search.gumbel_scale = search.get<double>("gumbel_scale");
    c.search.rollout_mix = search.get<double>("rollout_mix");
    if (search.has("sieve_gain")) c.search.sieve_gain = search.get<double>("sieve_gain");
    if (search.has("algorithm")) c.search.algorithm = search.get<std::string>("algorithm");
    if (search.has("init_contenders")) c.search.horizon.init_contenders = search.get<int>("init_contenders");
    if (search.has("init_worlds")) c.search.horizon.init_worlds = search.get<int>("init_worlds");
    if (search.has("wave_worlds")) c.search.horizon.wave_worlds = search.get<int>("wave_worlds");
    if (search.has("max_plunge_choices")) c.search.horizon.max_plunge_choices = search.get<int>("max_plunge_choices");
    if (search.has("horizon_turns")) c.search.horizon.horizon_turns = search.get<int>("horizon_turns");
    if (search.has("c_puct")) c.search.horizon.c_puct = search.get<double>("c_puct");
    if (search.has("readout_k0")) c.search.horizon.readout_k0 = search.get<double>("readout_k0");
    if (search.has("prior_se")) c.search.horizon.prior_se = search.get<double>("prior_se");
    if (search.has("lambda")) c.search.horizon.lambda = search.get<double>("lambda");
    c.budget = {.full = c.search.sims, .cheap = c.search.sims, .full_share = 1};
    if (search.has("cheap_sims") || search.has("full_share"))
        c.budget = {.full = c.search.sims, .cheap = search.get<int>("cheap_sims"),
                    .full_share = search.get<double>("full_share")};
    search.done();
    Table workers{root, "workers"};
    c.workers.actors = workers.get<int>("actors");
    c.workers.slots = workers.get<int>("slots");  // fights in flight per actor
    c.workers.batch_wait = std::chrono::microseconds{workers.get<int>("gpu_wait_us")};
    c.workers.timeout = std::chrono::milliseconds{workers.get<int>("timeout_ms")};
    c.workers.shutdown_timeout = std::chrono::milliseconds{workers.get<int>("shutdown_timeout_ms")};
    workers.done();
    for (const auto& [key, _] : root)
        if (key != "search" && key != "workers") throw std::invalid_argument{"unknown [" + std::string{key.str()} + "]"};
    c.search.validate();
    c.budget.validate();
    c.workers.validate();
    return c;
}
}  // namespace

PlayConfig parse_play_config(std::string_view text) { return parse(toml::parse(text)); }

PlayConfig read_play_config(const std::filesystem::path& path) {
    try {
        return parse(toml::parse_file(path.string()));
    } catch (const std::exception& e) {
        throw std::invalid_argument{path.string() + ": " + e.what()};
    }
}
}  // namespace stsrl::player
