#pragma once
#include <cstdint>
#include <span>

namespace stsrl::nn {
// IEEE round-to-nearest-even, matching the original CUDA host __float2half path.
// Uses F16C when supported, with a scalar fallback. No CUDA calls or shared buffers.
void pack_half(std::span<const float> source, std::span<std::uint16_t> destination);
}
