#pragma once
#include "agents/combat/nn/tokens.hpp"
#include "infra/inference/backend.hpp"
#include <filesystem>
#include <memory>
#include <vector>

namespace stsrl::nn {
// An exported model directory: its network.toml, as agents/combat/nn/pytorch_to_onnx.py writes it.
struct Network {
    Encoding encoding;
    std::vector<std::unique_ptr<infra::Backend>> backends;  // in the order requests try them
};

enum class Backends { all, cpu };
Network load_network(const std::filesystem::path& models, Backends = Backends::all);
}  // namespace stsrl::nn
