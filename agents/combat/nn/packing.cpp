#include "agents/combat/nn/packing.hpp"
#include <cuda_fp16.h>
#include <bit>
#include <stdexcept>
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#endif

namespace stsrl::nn {
namespace {
void scalar(std::span<const float> source, std::span<std::uint16_t> destination) {
    for (std::size_t i = 0; i < source.size(); ++i)
        destination[i] = std::bit_cast<std::uint16_t>(__float2half(source[i]));
}
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
__attribute__((target("avx,f16c")))
void vectorized(std::span<const float> source, std::span<std::uint16_t> destination) {
    std::size_t i = 0;
    for (; i + 8 <= source.size(); i += 8)
        _mm_storeu_si128(reinterpret_cast<__m128i*>(destination.data() + i),
            _mm256_cvtps_ph(_mm256_loadu_ps(source.data() + i), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    scalar(source.subspan(i), destination.subspan(i));
}
#endif
}
void pack_half(std::span<const float> source, std::span<std::uint16_t> destination) {
    if (destination.size() < source.size()) throw std::invalid_argument{"FP16 output buffer too small"};
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    static const bool supported = __builtin_cpu_supports("avx") && __builtin_cpu_supports("f16c");
    if (supported) { vectorized(source, destination); return; }
#endif
    scalar(source, destination);
}
}
