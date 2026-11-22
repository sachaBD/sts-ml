#include "apps/combat_inspector/inspection.hpp"
#include <emscripten.h>
#include <stdexcept>

EM_ASYNC_JS(int, predict, (const float* data, int count, int width, int actions, float* output), {
    try {
        // Growing the WASM heap across await invalidates its old typed-array views.
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
        // Parse the saved start only in C++: JavaScript numbers cannot represent its uint64 RNG states.
        request["start"] = nlohmann::json::parse(request.at("start_json").get<std::string>());
        response = stsrl::inspector::inspect(request,
            stsrl::nn::encoding_from_name(request.at("encoding").get<std::string>()), [](auto requests) {
                std::vector<stsrl::infra::Prediction> predictions;
                for (const auto& tokens : requests) {
                    std::vector<float> output(tokens.actions + 1);
                    if (!predict(tokens.data.data(), tokens.count, stsrl::nn::width, tokens.actions, output.data()))
                        throw std::runtime_error{"browser inference failed (see console)"};
                    predictions.push_back({output.front(), {output.begin() + 1, output.end()}});
                }
                return predictions;
            }).dump();
    } catch (const std::exception& error) {
        response = nlohmann::json{{"error", error.what()}}.dump();
    }
    return response.c_str();
}
