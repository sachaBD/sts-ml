#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "agents/combat/nn/training/sample.hpp"
#include <exception>
#include <new>
#include <span>
#include <vector>

namespace {
struct Unlocked {
    PyThreadState* thread = PyEval_SaveThread();
    ~Unlocked() { PyEval_RestoreThread(thread); }
};

PyObject* encode(PyObject*, PyObject* args) {
    PyObject* records;
    const char* name;
    if (!PyArg_ParseTuple(args, "Os", &records, &name)) return nullptr;
    PyObject* sequence = PySequence_Fast(records, "records must be a sequence of bytes");
    if (!sequence) return nullptr;
    try {
        const auto encoding = stsrl::nn::encoding_from_name(name);
        std::vector<std::span<const char>> views;
        const auto size = PySequence_Fast_GET_SIZE(sequence);
        views.reserve(size);
        for (Py_ssize_t i = 0; i < size; ++i) {
            char* bytes;
            Py_ssize_t length;
            if (PyBytes_AsStringAndSize(PySequence_Fast_GET_ITEM(sequence, i), &bytes, &length) < 0) {
                Py_DECREF(sequence);
                return nullptr;
            }
            views.emplace_back(bytes, std::size_t(length));
        }
        std::vector<stsrl::infra::Tokens> tokens;
        tokens.reserve(views.size());
        {
            // The sequence keeps immutable input bytes alive while the learner runs on another thread.
            Unlocked unlocked;
            for (const auto record : views) tokens.push_back(stsrl::nn::training::sample_tokens(record, encoding));
        }
        PyObject* result = PyList_New(size);
        if (!result) { Py_DECREF(sequence); return nullptr; }
        for (Py_ssize_t i = 0; i < size; ++i) {
            const auto& data = tokens[i].data;
            PyObject* bytes = PyBytes_FromStringAndSize(reinterpret_cast<const char*>(data.data()),
                                                       Py_ssize_t(data.size() * sizeof(float)));
            if (!bytes) { Py_DECREF(result); Py_DECREF(sequence); return nullptr; }
            PyList_SET_ITEM(result, i, bytes);
        }
        Py_DECREF(sequence);
        return result;
    } catch (const std::bad_alloc&) {
        Py_DECREF(sequence);
        return PyErr_NoMemory();
    } catch (const std::exception& error) {
        Py_DECREF(sequence);
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
}

PyMethodDef methods[] = {{"encode", encode, METH_VARARGS, nullptr}, {nullptr, nullptr, 0, nullptr}};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_state_replay", nullptr, -1, methods,
                      nullptr, nullptr, nullptr, nullptr};
}  // namespace

PyMODINIT_FUNC PyInit__state_replay() { return PyModule_Create(&module); }
