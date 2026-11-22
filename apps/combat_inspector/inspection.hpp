#pragma once
#include "agents/combat/nn/tokens.hpp"
#include <functional>
#include <nlohmann/json.hpp>

namespace stsrl::inspector {
using Predict = std::function<std::vector<infra::Prediction>(std::span<const infra::Tokens>)>;
// Same replay, search and JSON for native inspection and the browser. Hashes cross JS as decimal strings.
nlohmann::json inspect(const nlohmann::json&, nn::Encoding, const Predict&);
}  // namespace stsrl::inspector
