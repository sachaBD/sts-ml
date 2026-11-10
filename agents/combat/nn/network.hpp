#pragma once
#include "agents/combat/nn/tokens.hpp"
#include "infra/inference/backend.hpp"
#include <filesystem>
#include <memory>
#include <vector>

namespace stsrl::nn {
// An exported model directory, as agents/combat/nn/pytorch_to_onnx.py writes it.
struct Network {
    Encoding encoding;
    std::vector<std::unique_ptr<infra::Backend>> backends;  // in the order requests try them
};

Network load_network(const std::filesystem::path& models);
}  // namespace stsrl::nn
