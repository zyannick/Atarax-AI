// ataraxai/core_ai/src/bindings.cpp

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core_ai/core_ai_service.hh"

namespace py = pybind11;

PYBIND11_MODULE(core_ai_py, m) {
    m.doc() = "Python bindings for the AtaraxAI Core AI C++ engine. Provides access to LLM, STT, and other AI functionalities.";

    py::class_<CoreAIService>(m, "CoreAIService", "Manages AI model interactions, including LLM, STT, etc.")

        .def(py::init<>(), "Default constructor for CoreAIService.")

        .def("initialize", &CoreAIService::initialize,
             "Initializes the AI service and loads the specified LLM model.",
             py::arg("model_path"),
             py::arg("n_gpu_layers") = 0, 
             py::arg("n_ctx") = 2048     
        )

        .def("process_prompt", &CoreAIService::process_prompt,
             "Processes a text prompt using the loaded LLM and returns the generated response.",
             py::arg("prompt_text"),
             py::arg("max_new_tokens") = 128
        )


        .def("is_model_loaded", &CoreAIService::is_model_loaded,
             "Returns true if an LLM model is currently loaded and ready, false otherwise.");

};