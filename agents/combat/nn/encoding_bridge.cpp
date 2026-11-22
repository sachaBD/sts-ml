#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "agents/combat/nn/tokens.hpp"
#include "environments/combat/state_codec.hpp"
#include <cstring>
#include <exception>
#include <span>
#include <stdexcept>
#include <vector>

namespace {
struct Unlocked {
    PyThreadState* thread = PyEval_SaveThread();
    ~Unlocked() { PyEval_RestoreThread(thread); }
};

PyObject* encode(PyObject*, PyObject* args) {
    PyObject *state_input, *action_input;
    const char* name;
    if (!PyArg_ParseTuple(args, "OOs", &state_input, &action_input, &name)) return nullptr;
    // Own immutable snapshots before releasing the GIL: callers may mutate their lists.
    PyObject* states = PySequence_Tuple(state_input);
    if (!states) return nullptr;
    PyObject* orders = PySequence_Tuple(action_input);
    if (!orders) { Py_DECREF(states); return nullptr; }
    PyObject* result = nullptr;
    try {
        const auto encoding = stsrl::nn::encoding_from_name(name);
        const auto size = PyTuple_GET_SIZE(states);
        if (PyTuple_GET_SIZE(orders) != size) throw std::invalid_argument{"one action order is required per state"};
        std::vector<std::span<const char>> views;
        std::vector<std::vector<sts::search::Action>> actions(size);
        views.reserve(size);
        for (Py_ssize_t i = 0; i < size; ++i) {
            char *state, *order;
            Py_ssize_t state_size, order_size;
            if (PyBytes_AsStringAndSize(PyTuple_GET_ITEM(states, i), &state, &state_size) < 0 ||
                PyBytes_AsStringAndSize(PyTuple_GET_ITEM(orders, i), &order, &order_size) < 0)
                throw std::invalid_argument{"states and action orders must be bytes"};
            if (!order_size || order_size % sizeof(std::uint32_t))
                throw std::invalid_argument{"action orders must contain uint32 action bits"};
            views.emplace_back(state, std::size_t(state_size));
            for (Py_ssize_t at = 0; at < order_size; at += sizeof(std::uint32_t)) {
                std::uint32_t bits;
                std::memcpy(&bits, order + at, sizeof bits);
                actions[i].emplace_back(bits);
            }
        }
        std::vector<stsrl::infra::Tokens> tokens;
        tokens.reserve(size);
        {
            Unlocked unlocked;
            for (Py_ssize_t i = 0; i < size; ++i)
                tokens.push_back(stsrl::nn::encode(stsrl::combat::unpack(views[i]), actions[i], encoding));
        }
        result = PyList_New(size);
        if (result) {
            for (Py_ssize_t i = 0; i < size; ++i) {
                const auto& data = tokens[i].data;
                auto* bytes = PyBytes_FromStringAndSize(reinterpret_cast<const char*>(data.data()),
                                                       Py_ssize_t(data.size() * sizeof(float)));
                if (!bytes) { Py_CLEAR(result); break; }
                PyList_SET_ITEM(result, i, bytes);
            }
        }
    } catch (const std::bad_alloc&) {
        PyErr_NoMemory();
    } catch (const std::exception& error) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, error.what());
    }
    Py_DECREF(states);
    Py_DECREF(orders);
    return result;
}

PyMethodDef methods[] = {{"encode", encode, METH_VARARGS,
    "Encode packed states and uint32 action-order buffers into float32 token buffers."}, {nullptr, nullptr, 0, nullptr}};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_encoding_bridge", nullptr, -1, methods,
                      nullptr, nullptr, nullptr, nullptr};
}  // namespace

PyMODINIT_FUNC PyInit__encoding_bridge() {
    auto* result = PyModule_Create(&module);
    if (result && PyModule_AddIntConstant(result, "state_codec_version", stsrl::combat::state_codec_version) < 0)
        Py_CLEAR(result);
    return result;
}
