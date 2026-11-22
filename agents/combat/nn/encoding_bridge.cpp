#include "agents/combat/nn/tokens.hpp"
#include "environments/combat/state_codec.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

namespace py = pybind11;

namespace {
py::list encode(const std::vector<std::string>& states,
                const std::vector<std::vector<std::uint32_t>>& action_orders, std::string_view name) {
    if (states.size() != action_orders.size())
        throw std::invalid_argument{"one action order is required per state"};
    const auto encoding = stsrl::nn::encoding_from_name(name);
    py::list result;
    for (std::size_t i = 0; i < states.size(); ++i) {
        if (action_orders[i].empty()) throw std::invalid_argument{"action orders must not be empty"};
        const std::vector<sts::search::Action> actions(action_orders[i].begin(), action_orders[i].end());
        stsrl::infra::Tokens tokens;
        {
            py::gil_scoped_release release;
            tokens = stsrl::nn::encode(stsrl::combat::unpack(states[i]), actions, encoding);
        }
        result.append(py::bytes(reinterpret_cast<const char*>(tokens.data.data()), tokens.data.size() * sizeof(float)));
    }
    return result;
}
}  // namespace

PYBIND11_MODULE(_encoding_bridge, module) {
    module.def("encode", &encode);
    module.attr("state_codec_version") = stsrl::combat::state_codec_version;
}
