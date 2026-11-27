#include "agents/combat/player/play_config.hpp"
#include "agents/combat/search/config.hpp"
#include "agents/combat/search/sift_sieve/sift_sieve.hpp"
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
    const auto algorithm = search.has("algorithm") ? search.get<std::string>("algorithm") : "gumbel";
    if (algorithm == "sift-sieve") {
        const auto choice = search.has("root_choice") ? search.get<std::string>("root_choice") : "sample";
        if (choice != "sample" && choice != "argmax")
            throw std::invalid_argument{"[search].root_choice must be sample or argmax"};
        c.full_search = c.cheap_search = search::sift_sieve(
            choice == "sample" ? search::RootChoice::sample : search::RootChoice::argmax);
        c.budget = {.full = 0, .cheap = 0, .full_share = 1};
    } else if (algorithm == "gumbel") {
        search::SearchConfig gumbel{};
        gumbel.sims = search.get<int>("sims");
        gumbel.particles = search.get<int>("particles");
        gumbel.gumbel_scale = search.get<double>("gumbel_scale");
        gumbel.rollout_mix = search.get<double>("rollout_mix");
        if (search.has("sieve_gain")) gumbel.sieve_gain = search.get<double>("sieve_gain");
        c.budget = {.full = gumbel.sims, .cheap = gumbel.sims, .full_share = 1};
        if (search.has("cheap_sims") || search.has("full_share"))
            c.budget = {.full = gumbel.sims, .cheap = search.get<int>("cheap_sims"),
                        .full_share = search.get<double>("full_share")};
        c.full_search = gumbel.factory();
        gumbel.gumbel_scale = 0;  // cheap searches only move the fight on
        c.cheap_search = gumbel.factory();
    } else {
        throw std::invalid_argument{"unknown [search].algorithm: " + algorithm};
    }
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
