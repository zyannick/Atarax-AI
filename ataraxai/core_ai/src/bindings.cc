// ataraxai/core_ai/src/bindings.cpp

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core_ai/core_ai_service.hh"

namespace py = pybind11;

PYBIND11_MODULE(core_ai_py, m)
{
     m.doc() = "Python bindings for the AtaraxAI Core AI C++ engine. Provides access to LLM, STT, and other AI functionalities.";

     py::class_<CoreAIService>(m, "CoreAIService", "Manages AI model interactions, including LLM, STT, etc.")

         .def(py::init<>(), "Default constructor")
         .def("initialize_llama_model", &CoreAIService::initialize_llama_model, "Initialize and load the Llama model",
              py::arg("llama_model_params"))
         .def("unload_llama_model", &CoreAIService::unload_llama_model, "Unload the currently loaded Llama model")
         .def("is_llama_model_loaded", &CoreAIService::is_llama_model_loaded, "Check if a Llama model is loaded")
         .def("process_prompt", &CoreAIService::process_prompt, "Process a text prompt using the Llama model",
              py::arg("prompt_text"), py::arg("llama_generation_params"))
         .def("stream_prompt", &CoreAIService::stream_prompt, "Stream generation of text from a prompt",
              py::arg("prompt_text"), py::arg("llama_generation_params"), py::arg("callback"))
         .def("initialize_whisper_model", &CoreAIService::initialize_whisper_model, "Initialize and load the Whisper model",
              py::arg("whisper_model_params"))
         .def("unload_whisper_model", &CoreAIService::unload_whisper_model, "Unload the currently loaded Whisper model")
         .def("is_whisper_model_loaded", &CoreAIService::is_whisper_model_loaded, "Check if a Whisper model is loaded")
         .def("transcribe_audio_pcm", &CoreAIService::transcribe_audio_pcm, "Transcribe PCM audio data using Whisper",
              py::arg("pcm_f32_data"), py::arg("whisper_model_params"))
         .def("transcribe_audio_file", &CoreAIService::transcribe_audio_file, "Transcribe an audio file using Whisper",
              py::arg("audio_file_path"), py::arg("whisper_model_params"));
};