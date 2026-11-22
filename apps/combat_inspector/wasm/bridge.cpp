// Browser/WASM session driver over the same inspection::Session as native combat_lab.
#include "apps/combat_inspector/inspection.hpp"
#include <emscripten.h>
#include <optional>
#include <stdexcept>

namespace {
EM_ASYNC_JS(int, predict_js, (const float* data, int count, int width, int actions, float* output), {
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

std::vector<stsrl::infra::Prediction> wasm_predict(std::span<const stsrl::infra::Tokens> requests) {
    std::vector<stsrl::infra::Prediction> predictions;
    predictions.reserve(requests.size());
    for (const auto& tokens : requests) {
        std::vector<float> output(tokens.actions + 1);
        if (!predict_js(tokens.data.data(), tokens.count, stsrl::nn::width, tokens.actions, output.data()))
            throw std::runtime_error{"browser inference failed (see console)"};
        predictions.push_back({output.front(), {output.begin() + 1, output.end()}});
    }
    return predictions;
}
}  // namespace

extern "C" EMSCRIPTEN_KEEPALIVE const char* session_json(const char* input) {
    static std::string response;
    static std::optional<stsrl::inspection::Session> session;
    try {
        auto command = nlohmann::json::parse(input);
        const auto op = command.at("op").get<std::string>();
        if (op == "init") {
            session.emplace(stsrl::nn::encoding_from_name(command.at("encoding").get<std::string>()), wasm_predict);
            response = nlohmann::json{{"ok", true}}.dump();
        } else {
            if (!session) throw std::logic_error{"session not initialized; send op \"init\" first"};
            if (op == "open") {
                // Parse saved starts only in C++: JavaScript numbers cannot represent uint64 RNG states.
                for (auto& tree : command.at("trees"))
                    if (tree.contains("start_json"))
                        tree["start"] = nlohmann::json::parse(tree.at("start_json").get<std::string>());
            }
            response = session->run(command).dump();
        }
    } catch (const std::exception& error) {
        response = nlohmann::json{{"error", error.what()}}.dump();
    }
    return response.c_str();
}
