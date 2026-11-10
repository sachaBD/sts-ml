#pragma once
// A combat decision as one sequence of tokens (the network's only input); the legal actions are the last tokens,
// in the order given. Each encoding's layout is its layout.toml: entity/, inventory/.
#include "agents/combat/nn/layout.hpp"
#include "infra/inference/backend.hpp"
#include <memory>
#include <span>
#include <vector>
#include "combat/BattleContext.h"
#include "sim/search/Action.h"

namespace stsrl::nn {
class Encoder {
public:
    explicit Encoder(Layout layout) : layout_{std::move(layout)} {}
    virtual ~Encoder() = default;
    const Layout& layout() const { return layout_; }
    virtual infra::Tokens encode(const sts::BattleContext&, std::span<const sts::search::Action> actions) const = 0;

protected:
    const Layout layout_;
};

// The encoder that writes `layout`, chosen by its name.
std::unique_ptr<const Encoder> make_encoder(Layout layout);

// One row being appended to `out`: its header (kind, id, id2), then at most the kind's feature count of values.
class Row {
public:
    Row(std::vector<float>& out, const Layout& layout, const Layout::Kind& kind, int id = 0, int id2 = 0);
    int index() const { return index_; }
    void put(float value);  // throws past the kind's feature count
    template<class R> void put_all(const R& values) { for (auto x : values) put(float(x)); }
    void link(int column, int row);  // a reference to `row`, as the layout stores it

private:
    std::vector<float>& out_;
    const Layout& layout_;
    int index_, next_, end_;
};
}  // namespace stsrl::nn
