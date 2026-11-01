// Keep the native explanation path; Asyncify suspends it while ORT Web evaluates a query.
#include "agents/combat/nn/explanation.hpp"
#include "agents/combat/nn/tokens.hpp"
#include <emscripten.h>
#include <cmath>
#include <stdexcept>
#include <string>

EM_ASYNC_JS(int, predict, (const float* data, int count, int width, int actions, float* output), {
    try {
        // Copy before await: growing the WASM heap invalidates its previous typed-array views.
        const tokens = new Float32Array(HEAPF32.subarray(data / 4, data / 4 + count * width));
        const result = await globalThis.gumbelSession.run({tokens: new ort.Tensor('float32', tokens, [1, count, width])});
        const value = result.value.data[0];
        const logits = result.logits.data.slice(count - actions, count);
        if (!Number.isFinite(value) || logits.length !== actions || !logits.every(Number.isFinite))
            throw new Error('invalid network prediction');
        HEAPF32[output / 4] = value;
        HEAPF32.set(logits, output / 4 + 1);
        return 1;
    } catch (error) {
        console.error('ONNX inference:', error);
        return 0;
    }
});

extern "C" EMSCRIPTEN_KEEPALIVE const char* explain_json(const char* input) {
    static std::string response;
    try {
        auto request = nlohmann::json::parse(input);
        // Battle RNG state is parsed only in C++, never through JavaScript's lossy numbers.
        request["start"] = nlohmann::json::parse(request.at("start_json").get<std::string>());
        response = stsrl::nn::explain(request, [](const stsrl::infra::Tokens& tokens) {
            std::vector<float> output(tokens.actions + 1);
            if (!predict(tokens.data.data(), tokens.count, stsrl::nn::width, tokens.actions, output.data()))
                throw std::runtime_error{"browser network evaluation failed (see console)"};
            return stsrl::infra::Prediction{output.front(), {output.begin() + 1, output.end()}};
        }).dump();
    } catch (const std::exception& error) {
        response = nlohmann::json{{"error", error.what()}}.dump();
    }
    return response.c_str();
}
