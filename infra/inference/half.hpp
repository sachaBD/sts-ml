#pragma once
#include <cstdint>
#include <span>

namespace stsrl::infra {
// float32 -> IEEE fp16 bits, round-to-nearest-even (same as CUDA __float2half). F16C when available.
void pack_half(std::span<const float> source, std::span<std::uint16_t> destination);
}
