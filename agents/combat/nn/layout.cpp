#include "agents/combat/nn/layout.hpp"
#include <stdexcept>
#include <toml++/toml.hpp>
#include <vector>

namespace stsrl::nn {
namespace {
constexpr int header = 3;  // kind, id, id2: the columns every row begins with (tokens.hpp Row)

template<class T> T required(const toml::node_view<const toml::node>& node, const std::string& what) {
    const auto value = node.value<T>();
    if (!value) throw std::invalid_argument{"layout: missing or mistyped " + what};
    return *value;
}

std::map<std::string, int> integers(const toml::table& table, const std::string& name) {
    std::map<std::string, int> out;
    if (const auto* section = table[name].as_table())
        for (const auto& [key, value] : *section) out[std::string{key.str()}] = required<int>(toml::node_view<const toml::node>{&value}, name + "." + std::string{key.str()});
    return out;
}

template<class Map> const auto& at(const Map& map, const std::string& key, const std::string& what, const std::string& layout) {
    const auto found = map.find(key);
    if (found == map.end()) throw std::invalid_argument{"layout " + layout + " has no " + what + " " + key};
    return found->second;
}
}  // namespace

const Layout::Kind& Layout::kind(const std::string& key) const { return at(kinds, key, "kind", name); }
int Layout::link(const std::string& key) const { return at(links, key, "link", name); }
int Layout::card_zone(const std::string& key) const { return at(card_zones, key, "card zone", name); }
int Layout::action_kind(const std::string& key) const { return at(action_kinds, key, "action kind", name); }

Layout load_layout(const std::filesystem::path& path) {
    const toml::table table = toml::parse_file(path.string());
    Layout layout{required<std::string>(table["name"], "name"), required<int>(table["version"], "version"),
                  required<int>(table["width"], "width"), required<int>(table["id_capacity"], "id_capacity"),
                  required<int>(table["link_offset"], "link_offset"), {}, integers(table, "links"),
                  integers(table, "card_zones"), integers(table, "action_kinds")};
    std::vector<std::string> columns;
    if (const auto* names = table["header"].as_array())
        for (const auto& name : *names) columns.push_back(required<std::string>(toml::node_view<const toml::node>{&name}, "header"));
    if (columns != std::vector<std::string>{"kind", "id", "id2"})
        throw std::invalid_argument{path.string() + ": header must be kind, id, id2"};
    if (const auto* kinds = table["kinds"].as_table())
        for (const auto& [key, value] : *kinds) {
            const toml::node_view<const toml::node> kind{&value};
            const std::string name{key.str()};
            layout.kinds[name] = {required<int>(kind["id"], "kinds." + name + ".id"),
                                  required<int>(kind["features"], "kinds." + name + ".features")};
            if (header + layout.kinds[name].features > layout.width)
                throw std::invalid_argument{path.string() + ": kind " + name + " has more features than fit"};
        }
    for (const auto& [name, column] : layout.links)
        if (column < header || column >= layout.width) throw std::invalid_argument{path.string() + ": link " + name + " outside the row"};
    if (layout.kinds.empty() || layout.id_capacity < 1) throw std::invalid_argument{path.string() + ": kinds and a positive id capacity required"};
    return layout;
}
}  // namespace stsrl::nn
