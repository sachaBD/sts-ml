#include "agents/combat/nn/tokens.hpp"
#include "agents/combat/nn/entity/encoding.hpp"
#include "agents/combat/nn/inventory/encoding.hpp"
#include <stdexcept>
#include <string>

namespace stsrl::nn {
std::unique_ptr<const Encoder> make_encoder(Layout layout) {
    if (layout.name == "entity") return entity_encoder(std::move(layout));
    if (layout.name == "inventory_v1") return inventory_encoder(std::move(layout));
    throw std::invalid_argument{"no encoder writes layout " + layout.name};
}

Row::Row(std::vector<float>& out, const Layout& layout, const Layout::Kind& kind, int id, int id2)
    : out_{out}, layout_{layout}, index_{int(out.size()) / layout.width}, next_{3}, end_{3 + kind.features} {
    if (id < 0 || id >= layout.id_capacity || id2 < 0 || id2 >= layout.id_capacity)
        throw std::logic_error{layout.name + ": id outside the layout's id capacity"};
    out.resize(out.size() + layout.width, 0);
    const auto start = std::size_t(index_) * layout.width;
    out[start] = float(kind.id);
    out[start + 1] = float(id);
    out[start + 2] = float(id2);
}

void Row::put(float value) {
    if (next_ >= end_) throw std::logic_error{layout_.name + ": more features than the row's kind declares"};
    out_[std::size_t(index_) * layout_.width + next_++] = value;
}

void Row::link(int column, int row) {
    out_[std::size_t(index_) * layout_.width + column] = float(row + layout_.link_offset);
}
}  // namespace stsrl::nn
