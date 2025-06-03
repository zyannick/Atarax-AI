// ataraxai/core_ai/src/bindings.cpp

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core_ai/core_ai_service.hh"

namespace py = pybind11;

PYBIND11_MODULE(core_ai_py, m)
{
     m.doc() = "Python bindings for the AtaraxAI Core AI C++ engine. Provides access to LLM, STT, and other AI functionalities.";

     py::class_<LlamaModelParams>(m, "LlamaModelParams", "Parameters for loading a Llama model.")
         .def(py::init<>())
         .def(py::init<const std::string &, int32_t, int32_t, int32_t, bool, bool, bool, bool>(),
              py::arg("model_path") = "",
              py::arg("n_ctx") = 2048,
              py::arg("n_gpu_layers") = 0,
              py::arg("main_gpu") = 0,
              py::arg("tensor_split") = false,
              py::arg("vocab_only") = false,
              py::arg("use_map") = false,
              py::arg("use_mlock") = false)

         .def_readwrite("model_path", &LlamaModelParams::model_path, "Path to the GGUF model file.")
         .def_readwrite("n_gpu_layers", &LlamaModelParams::n_gpu_layers, "Number of layers to offload to GPU.")
         .def_readwrite("n_ctx", &LlamaModelParams::n_ctx, "Context size for the model.")
         .def_readwrite("main_gpu", &LlamaModelParams::main_gpu, "Main GPU index for model loading.")
         .def_readwrite("tensor_split", &LlamaModelParams::tensor_split, "Whether to use tensor splitting for large models.")
         .def_readwrite("vocab_only", &LlamaModelParams::vocab_only, "Load only the vocabulary without the model.")
         .def_readwrite("use_map", &LlamaModelParams::use_map, "Use memory mapping for the model file.")
         .def_readwrite("use_mlock", &LlamaModelParams::use_mlock, "Lock model memory to prevent swapping.")
         .def("__eq__", [](const LlamaModelParams &a, const LlamaModelParams &b)
              { return a == b; })
         .def("__ne__", [](const LlamaModelParams &a, const LlamaModelParams &b)
              { return a != b; })
         .def("__hash__", [](const LlamaModelParams &p)
              { return p.hash(); })
         .def("__str__", [](const LlamaModelParams &p)
              { return p.to_string(); });

     py::class_<GenerationParams>(m, "GenerationParams", "Parameters for Llama text generation.")
         .def(py::init<>())
         .def(py::init<int32_t, float, int32_t, float, float,
                       std::vector<std::string>, int32_t, int32_t>(),
              py::arg("n_predict") = 128,
              py::arg("temp") = 0.8f,
              py::arg("top_k") = 40,
              py::arg("top_p") = 0.95f,
              py::arg("repeat_penalty") = 1.1f,
              py::arg("stop_sequences") = std::vector<std::string>{},
              py::arg("n_batch") = 512,
              py::arg("n_threads") = 0)
         .def_readwrite("n_predict", &GenerationParams::n_predict)
         .def_readwrite("temp", &GenerationParams::temp)
         .def_readwrite("top_k", &GenerationParams::top_k)
         .def_readwrite("top_p", &GenerationParams::top_p)
         .def_readwrite("repeat_penalty", &GenerationParams::repeat_penalty)
         .def_readwrite("stop_sequences", &GenerationParams::stop_sequences)
         .def_readwrite("n_batch", &GenerationParams::n_batch)
         .def_readwrite("n_threads", &GenerationParams::n_threads)
         .def("__eq__", [](const GenerationParams &a, const GenerationParams &b)
              { return a == b; })
         .def("__ne__", [](const GenerationParams &a, const GenerationParams &b)
              { return a != b; })
         .def("__hash__", [](const GenerationParams &p)
              { return p.hash(); })
         .def("__str__", [](const GenerationParams &p)
              { return p.to_string(); });

     py::class_<WhisperModelParams>(m, "WhisperModelParams", "Parameters for loading a Whisper model.")
         .def(py::init<>())
         .def(py::init<const std::string &, bool>(),
              py::arg("model_path") = "",
              py::arg("use_gpu") = true)
         .def_readwrite("model_path", &WhisperModelParams::model_path, "Path to the Whisper GGUF model file.")
         .def_readwrite("use_gpu", &WhisperModelParams::use_gpu, "Whether to use GPU for transcription.")
         .def("__eq__", [](const WhisperModelParams &a, const WhisperModelParams &b)
              { return a == b; })
         .def("__ne__", [](const WhisperModelParams &a, const WhisperModelParams &b)
              { return a != b; })
         .def("__hash__", [](const WhisperModelParams &p)
              { return p.hash(); })
         .def("__str__", [](const WhisperModelParams &p)
              { return p.to_string(); });

     py::class_<WhisperTranscriptionParams>(m, "WhisperTranscriptionParams", "Parameters for Whisper audio transcription.")
         .def(py::init<>())
         .def(py::init<int, const std::string &>(),
              py::arg("n_threads") = 4,
              py::arg("language") = "en")
         .def_readwrite("n_threads", &WhisperTranscriptionParams::n_threads)
         .def_readwrite("language", &WhisperTranscriptionParams::language, "Target language for transcription (e.g., 'en', 'auto').")
         .def_readwrite("translate", &WhisperTranscriptionParams::translate, "Translate to English if true.")
         .def_readwrite("print_special", &WhisperTranscriptionParams::print_special)
         .def_readwrite("print_progress", &WhisperTranscriptionParams::print_progress)
         .def_readwrite("no_context", &WhisperTranscriptionParams::no_context)
         .def_readwrite("max_len", &WhisperTranscriptionParams::max_len)
         .def_readwrite("single_segment", &WhisperTranscriptionParams::single_segment)
         .def_readwrite("temperature", &WhisperTranscriptionParams::temperature)
         .def("__eq__", [](const WhisperTranscriptionParams &a, const WhisperTranscriptionParams &b)
              { return a == b; })
         .def("__ne__", [](const WhisperTranscriptionParams &a, const WhisperTranscriptionParams &b)
              { return a != b; })
         .def("__hash__", [](const WhisperTranscriptionParams &p)
              { return p.hash(); })
         .def("__str__", [](const WhisperTranscriptionParams &p)
              { return p.to_string(); });

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