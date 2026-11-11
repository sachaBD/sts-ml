// One decision inspection per JSON line; CPU inference on an exported ONNX model.
// Requests add {"model": ONNX path, "encoding": encoding name} to player::explain's fields.
#include "agents/combat/player/explanation.hpp"
#include "agents/combat/nn/tokens.hpp"
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

int main() {
    std::filesystem::path loaded;
    std::unique_ptr<stsrl::infra::Backend> backend;
    for (std::string line; std::getline(std::cin, line);) {
        if (line.empty()) continue;
        nlohmann::json out;
        try {
            const auto request = nlohmann::json::parse(line);
            const std::filesystem::path model = request.at("model").get<std::string>();
            if (model != loaded) {
                loaded.clear();
                backend = stsrl::infra::cpu_backend(model, 1, 1);
                if (backend->width() != stsrl::nn::width) throw std::invalid_argument{"model token width does not match combat encoder"};
                loaded = model;
            }
            out = stsrl::player::explain(request, stsrl::nn::encoding_from_name(request.at("encoding").get<std::string>()), [&](const stsrl::infra::Tokens& tokens) {
                const stsrl::infra::Input input{&tokens, {}};
                return backend->evaluate({&input, 1}).front();
            });
        } catch (const std::exception& e) {
            out = {{"error", e.what()}};
        }
        std::cout << out.dump() << std::endl;
    }
}
