#pragma once
#include <filesystem>
#include <map>
#include <string>

namespace stsrl::nn {
// An encoding's token layout, read from its layout.toml: the one description its encoder and model share.
struct Layout {
    struct Kind {
        int id, features;
    };
    std::string name;
    int version, width, id_capacity, link_offset;
    std::map<std::string, Kind> kinds;
    std::map<std::string, int> links, card_zones, action_kinds;

    // Each throws naming the entry when the layout has none.
    const Kind& kind(const std::string&) const;
    int link(const std::string&) const;
    int card_zone(const std::string&) const;
    int action_kind(const std::string&) const;
};

Layout load_layout(const std::filesystem::path&);
}  // namespace stsrl::nn
