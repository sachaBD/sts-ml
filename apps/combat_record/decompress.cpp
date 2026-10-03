// combat_v4 -> combat_v4_full (runs/schema=combat_v4_full/schema.py), Parquet in, Parquet out.
//
//   combat_v4_decompress --out DIR --fights FIGHTS.parquet [--search SEARCH.parquet] [--part K --parts N]
//
// Replays the fights of FIGHTS (combat_v4 rows; select them first, e.g. with DuckDB) and writes
// DIR/decisions-K.parquet and DIR/fights-K.parquet. SEARCH (optional) is joined on fight_id + step for the playing
// agent. To use several cores run N processes with --part 0..N-1 --parts N (each takes every N-th fight).
// Fights that do not replay as recorded are skipped and reported on stderr; a contract violation aborts.
// stdout: {"fights", "decisions", "mismatches"}.
#include "environments/combat/record_v4_full.hpp"

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>

#include <iostream>
#include <string>
#include <unordered_map>

namespace full = stsrl::combat_v4_full;

namespace {

template <typename T>
T value(arrow::Result<T> r) {
    if (!r.ok()) throw std::runtime_error{r.status().ToString()};
    return std::move(*r);
}
void ok(const arrow::Status& s) {
    if (!s.ok()) throw std::runtime_error{s.ToString()};
}

std::shared_ptr<arrow::Table> read(const std::string& path) {
    auto reader = value(parquet::arrow::OpenFile(value(arrow::io::ReadableFile::Open(path)), arrow::default_memory_pool()));
    std::shared_ptr<arrow::Table> table;
    ok(reader->ReadTable(&table));
    return table;
}

void write(const std::shared_ptr<arrow::Table>& table, const std::string& path) {
    auto props = parquet::WriterProperties::Builder().compression(parquet::Compression::ZSTD)->build();
    auto arrow_props = parquet::ArrowWriterProperties::Builder().store_schema()->build();
    auto out = value(arrow::io::FileOutputStream::Open(path));
    ok(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), out, 64 * 1024, props, arrow_props));
    ok(out->Close());
}

// ---- column access ----

std::shared_ptr<arrow::Array> column(const arrow::Table& t, const char* name) {
    auto c = t.GetColumnByName(name);
    if (!c) throw std::runtime_error{std::string{"missing column "} + name};
    return c->num_chunks() == 1 ? c->chunk(0) : value(arrow::Concatenate(c->chunks()));
}
std::shared_ptr<arrow::Array> field(const arrow::Array& a, const char* name) {
    auto f = static_cast<const arrow::StructArray&>(a).GetFieldByName(name);
    if (!f) throw std::runtime_error{std::string{"missing field "} + name};
    return f;
}
std::int64_t integer(const arrow::Array& a, std::int64_t i) {
    switch (a.type_id()) {
        case arrow::Type::INT8: return static_cast<const arrow::Int8Array&>(a).Value(i);
        case arrow::Type::INT16: return static_cast<const arrow::Int16Array&>(a).Value(i);
        case arrow::Type::INT32: return static_cast<const arrow::Int32Array&>(a).Value(i);
        case arrow::Type::UINT8: return static_cast<const arrow::UInt8Array&>(a).Value(i);
        case arrow::Type::UINT32: return static_cast<const arrow::UInt32Array&>(a).Value(i);
        case arrow::Type::UINT64: return static_cast<std::int64_t>(static_cast<const arrow::UInt64Array&>(a).Value(i));
        case arrow::Type::BOOL: return static_cast<const arrow::BooleanArray&>(a).Value(i);
        default: throw std::runtime_error{"unexpected column type " + a.type()->ToString()};
    }
}
int int_of(const arrow::Array& a, std::int64_t i) { return static_cast<int>(integer(a, i)); }
std::string text(const arrow::Array& a, std::int64_t i) { return std::string{static_cast<const arrow::StringArray&>(a).GetView(i)}; }

struct Elements {  // row i of a list column: elements [begin, end) of `values`
    std::shared_ptr<arrow::Array> values;
    std::int64_t begin, end;
    Elements(const arrow::Array& list, std::int64_t i) {
        const auto& l = static_cast<const arrow::ListArray&>(list);
        values = l.values();
        begin = l.value_offset(i);
        end = begin + l.value_length(i);
    }
};

template <typename T>
std::vector<T> ints(const arrow::Array& list, std::int64_t i) {
    const Elements e{list, i};
    std::vector<T> v;
    for (auto k = e.begin; k < e.end; ++k) v.push_back(static_cast<T>(integer(*e.values, k)));
    return v;
}

stsrl::combat_v4::RngState rng(const arrow::Array& s, std::int64_t i) {
    return {int_of(*field(s, "counter"), i), static_cast<std::uint64_t>(integer(*field(s, "seed0"), i)),
            static_cast<std::uint64_t>(integer(*field(s, "seed1"), i))};
}

stsrl::combat_v4::Start start(const arrow::Array& s, std::int64_t i) {
    const auto I = [&](const char* name) { return int_of(*field(s, name), i); };
    stsrl::combat_v4::Start r{};
    r.seed = static_cast<std::uint64_t>(integer(*field(s, "seed"), i));
    r.ascension = I("ascension"); r.act = I("act"); r.floor = I("floor"); r.encounter = I("encounter");
    r.cur_room = I("cur_room"); r.last_room = I("last_room"); r.burning_elite_buff = I("burning_elite_buff");
    r.hp = I("hp"); r.max_hp = I("max_hp"); r.gold = I("gold"); r.potion_capacity = I("potion_capacity");
    r.misc_rng = rng(*field(s, "misc_rng"), i);
    r.potion_rng = rng(*field(s, "potion_rng"), i);
    r.potions = ints<int>(*field(s, "potions"), i);
    const auto bottled = ints<int>(*field(s, "bottled"), i);
    if (bottled.size() != 3) throw std::runtime_error{"bottled: expected 3 entries"};
    std::copy(bottled.begin(), bottled.end(), r.bottled);
    const Elements relics{*field(s, "relics"), i}, deck{*field(s, "deck"), i};
    for (auto k = relics.begin; k < relics.end; ++k)
        r.relics.push_back({int_of(*field(*relics.values, "id"), k), int_of(*field(*relics.values, "data"), k)});
    for (auto k = deck.begin; k < deck.end; ++k)
        r.deck.push_back({int_of(*field(*deck.values, "id"), k), integer(*field(*deck.values, "upgraded"), k) != 0,
                          int_of(*field(*deck.values, "misc"), k)});
    return r;
}

// fights.agent has a launcher prefix ('run_rl mcts leaf=...') that search.agent lacks ('mcts leaf=...').
bool same_agent(const std::string& fight_agent, const std::string& search_agent) {
    return fight_agent == search_agent ||
           (fight_agent.size() > search_agent.size() && fight_agent.ends_with(" " + search_agent));
}

}  // namespace

int main(int argc, char** argv) {
    std::string out, fights_path, search_path;
    int part = 0, parts = 1;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--out") out = v;
        else if (a == "--fights") fights_path = v;
        else if (a == "--search") search_path = v;
        else if (a == "--part") part = std::stoi(v);
        else if (a == "--parts") parts = std::stoi(v);
        else { std::cerr << "unknown argument " << a << '\n'; return 2; }
    }
    if (out.empty() || fights_path.empty()) {
        std::cerr << "usage: combat_v4_decompress --out DIR --fights F.parquet [--search S.parquet] [--part K --parts N]\n";
        return 2;
    }
    try {
        const auto table = read(fights_path);
        const auto st = column(*table, "start"), id = column(*table, "fight_id"), agent = column(*table, "agent"),
                   won = column(*table, "won"), final_hp = column(*table, "final_hp"),
                   actions = column(*table, "actions"), explored = column(*table, "explored");

        std::vector<full::Fight> fights;
        std::vector<std::vector<full::SearchStep>> search;  // per fight, per step; children empty: no row
        std::unordered_map<std::string, std::size_t> index;
        for (std::int64_t i = part; i < table->num_rows(); i += parts) {
            full::Fight f;
            f.fight_id = text(*id, i);
            f.agent = text(*agent, i);
            f.start = start(*st, i);
            f.actions = ints<std::uint32_t>(*actions, i);
            f.explored = ints<char>(*explored, i);
            f.won = integer(*won, i) != 0;
            f.final_hp = int_of(*final_hp, i);
            index[f.fight_id] = fights.size();
            search.emplace_back(f.actions.size());
            fights.push_back(std::move(f));
        }

        if (!search_path.empty()) {
            const auto s = read(search_path);
            const auto sid = column(*s, "fight_id"), step = column(*s, "step"), sagent = column(*s, "agent"),
                       root = column(*s, "root_value"), sims = column(*s, "simulations"), kids = column(*s, "children");
            for (std::int64_t i = 0; i < s->num_rows(); ++i) {
                const auto it = index.find(text(*sid, i));
                if (it == index.end() || !same_agent(fights[it->second].agent, text(*sagent, i))) continue;
                auto& dst = search[it->second].at(static_cast<std::size_t>(integer(*step, i)));
                dst.root_value = static_cast<const arrow::FloatArray&>(*root).Value(i);
                dst.simulations = static_cast<std::uint32_t>(integer(*sims, i));
                const Elements e{*kids, i};
                for (auto k = e.begin; k < e.end; ++k)
                    dst.children.push_back({static_cast<std::uint32_t>(integer(*field(*e.values, "action"), k)),
                                            static_cast<std::uint32_t>(integer(*field(*e.values, "visits"), k)),
                                            static_cast<const arrow::FloatArray&>(*field(*e.values, "value")).Value(k)});
            }
            for (std::size_t k = 0; k < fights.size(); ++k) {
                bool any = false;
                for (const auto& step : search[k]) any = any || step.simulations > 0;
                if (!any) continue;
                for (std::size_t j = 0; j < search[k].size(); ++j)
                    fights[k].search.push_back(search[k][j].simulations ? &search[k][j] : nullptr);
            }
        }

        full::Expander expander;
        int mismatches = 0;
        for (const auto& f : fights) {
            std::string why;
            if (!expander.add(f, why)) {
                std::cerr << "mismatch " << f.fight_id << ": " << why << '\n';
                ++mismatches;
            }
        }
        const auto suffix = "-" + std::to_string(part) + ".parquet";
        write(expander.take_decisions(), out + "/decisions" + suffix);
        write(expander.take_fights(), out + "/fights" + suffix);
        std::cout << "{\"fights\": " << expander.fights() << ", \"decisions\": " << expander.decisions()
                  << ", \"mismatches\": " << mismatches << "}\n";
    } catch (const std::exception& e) {
        std::cerr << "combat_v4_decompress: " << e.what() << '\n';
        return 1;
    }
}
