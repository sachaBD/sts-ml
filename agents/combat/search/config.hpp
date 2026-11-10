#pragma once
// The one place that knows which search runs and how it is configured. Callers see only a SearchFactory.
#include "agents/combat/search/search.hpp"
#include <string>
#include <string_view>

namespace stsrl::search {
class SearchConfig {
public:
    static constexpr std::string_view usage =
        "[--sims S] [--particles P] [--gumbel-scale G] [--value-scale V] [--rollout-mix M]";

    // False when `flag` is not a search option; throws on a bad value.
    bool parse(std::string_view flag, const std::string& value);
    // Gumbel search; with rollout_mix > 0 each leaf value is blended with a guided rollout.
    SearchFactory factory() const;

private:
    int simulations_ = 256, particles_ = 8;
    double gumbel_scale_ = 0, value_scale_ = 0.1, rollout_mix_ = 0.5;
};
}  // namespace stsrl::search
