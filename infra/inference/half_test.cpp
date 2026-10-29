#include "infra/inference/half.hpp"
#include <cuda_fp16.h>
#include <bit>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

int main() try {
    std::vector<float> input;
    for (unsigned bits = 0; bits < 65536; ++bits) {
        const float value = __half2float(std::bit_cast<__half>(std::uint16_t(bits)));
        if (std::isfinite(value)) {
            input.push_back(value);
            input.push_back(std::nextafter(value, INFINITY));
            input.push_back(std::nextafter(value, -INFINITY));
        }
    }
    std::mt19937 rng{7};
    for (int i = 0; i < 100003; ++i) {
        const float value = std::bit_cast<float>(std::uint32_t(rng()));
        if (std::isfinite(value)) input.push_back(value);
    }
    std::vector<std::uint16_t> output(input.size() + 8, 0x1234);
    stsrl::infra::pack_half(input, output);
    for (std::size_t i = 0; i < input.size(); ++i)
        if (output[i] != std::bit_cast<std::uint16_t>(__float2half(input[i])))
            throw std::runtime_error{"FP16 conversion differs at " + std::to_string(i)};
    for (std::size_t i = input.size(); i < output.size(); ++i)
        if (output[i] != 0x1234) throw std::runtime_error{"packing overrun"};
    std::cout << input.size() << " packing values match CUDA reference exactly\n";
    return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
