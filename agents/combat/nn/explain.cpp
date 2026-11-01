// nn_explain: one decision of the NN combat agent per request, for the Gumbel tree viewer.
// stdin: one request per line, explain()'s request plus {"model": ONNX path}; stdout: one response per line.
#include "agents/combat/nn/explanation.hpp"
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
                backend = stsrl::infra::cpu_backend(model);
                if (backend->width() != stsrl::nn::width) throw std::invalid_argument{"model token width does not match combat encoder"};
                loaded = model;
            }
            out = stsrl::nn::explain(request, [&](const stsrl::infra::Tokens& tokens) {
                const stsrl::infra::Input input{&tokens, {}};
                return backend->evaluate({&input, 1}).front();
            });
        } catch (const std::exception& e) {
            out = {{"error", e.what()}};
        }
        std::cout << out.dump() << std::endl;
    }
}
