#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core_ai_service.hh"
#include "model_benchmarker.hh"
#include "memory_locker.hh"

namespace py = pybind11;

PYBIND11_MODULE(hegemonikon_py, m)
{
     m.doc() = "Python bindings for the AtaraxAI Core AI C++ engine. Provides access to LLM, STT, and other AI functionalities.";

     py::class_<HegemonikonLlamaModelParams>(m, "HegemonikonLlamaModelParams", "Parameters for loading a Llama model.")
         .def(py::init<>())
         .def(py::init<const std::string &, int32_t, int32_t, int32_t, int32_t, bool, bool, bool, bool>(),
              py::arg("model_path") = "",
              py::arg("n_ctx") = 2048,
              py::arg("n_gpu_layers") = 0,
              py::arg("main_gpu") = 0,
              py::arg("n_batch") = 1,
              py::arg("tensor_split") = false,
              py::arg("vocab_only") = false,
              py::arg("use_map") = false,
              py::arg("use_mlock") = false)
         .def_static("from_dict", [](const py::dict &d)
                     {
     HegemonikonLlamaModelParams params;
     if (d.contains("model_path")) {
           params.model_path = d["model_path"].cast<std::string>();
     }
     if (d.contains("n_ctx")) {
               params.n_ctx = d["n_ctx"].cast<int32_t>();
     }
     if (d.contains("n_gpu_layers")) {
               params.n_gpu_layers = d["n_gpu_layers"].cast<int32_t>();
     }
     if (d.contains("main_gpu")) {
               params.main_gpu = d["main_gpu"].cast<int32_t>();
     }
     if (d.contains("n_batch")) {
               params.n_batch = d["n_batch"].cast<int32_t>();
     }
     if (d.contains("tensor_split")) {
               params.tensor_split = d["tensor_split"].cast<bool>();
     }
     if (d.contains("vocab_only")) {
               params.vocab_only = d["vocab_only"].cast<bool>();
     }
     if (d.contains("use_map")) {
               params.use_map = d["use_map"].cast<bool>();
     }
     if (d.contains("use_mlock")) {
               params.use_mlock = d["use_mlock"].cast<bool>();
     }
     return params; })
         .def("set_model_path", &HegemonikonLlamaModelParams::set_model_path, "Set the model file path.")
         .def_readwrite("model_path", &HegemonikonLlamaModelParams::model_path, "Path to the GGUF model file.")
         .def_readwrite("n_gpu_layers", &HegemonikonLlamaModelParams::n_gpu_layers, "Number of layers to offload to GPU.")
         .def_readwrite("n_ctx", &HegemonikonLlamaModelParams::n_ctx, "Context size for the model.")
         .def_readwrite("main_gpu", &HegemonikonLlamaModelParams::main_gpu, "Main GPU index for model loading.")
         .def_readwrite("n_batch", &HegemonikonLlamaModelParams::n_batch, "Batch size for model inference.")
         .def_readwrite("tensor_split", &HegemonikonLlamaModelParams::tensor_split, "Whether to use tensor splitting for large models.")
         .def_readwrite("vocab_only", &HegemonikonLlamaModelParams::vocab_only, "Load only the vocabulary without the model.")
         .def_readwrite("use_map", &HegemonikonLlamaModelParams::use_map, "Use memory mapping for the model file.")
         .def_readwrite("use_mlock", &HegemonikonLlamaModelParams::use_mlock, "Lock model memory to prevent swapping.")
         .def("__eq__", [](const HegemonikonLlamaModelParams &a, const HegemonikonLlamaModelParams &b)
              { return a == b; })
         .def("__ne__", [](const HegemonikonLlamaModelParams &a, const HegemonikonLlamaModelParams &b)
              { return a != b; })
         .def("__hash__", [](const HegemonikonLlamaModelParams &p)
              { return p.hash(); })
         .def("__str__", [](const HegemonikonLlamaModelParams &p)
              { return p.to_string(); });

     py::class_<HegemonikonGenerationParams>(m, "HegemonikonGenerationParams", "Parameters for Llama text generation.")
         .def(py::init<>())
         .def(py::init<int32_t, float, int32_t, float, float, int32_t, float, float,
                       std::vector<std::string>, int32_t, int32_t>(),
              py::arg("n_predict") = 128,
              py::arg("temperature") = 0.8f,
              py::arg("top_k") = 40,
              py::arg("top_p") = 0.95f,
              py::arg("repeat_penalty") = 1.1f,
              py::arg("penalty_last_n") = 64,
              py::arg("penalty_freq") = 0.0f,
              py::arg("penalty_present") = 0.0f,
              py::arg("stop_sequences") = std::vector<std::string>{},
              py::arg("n_batch") = 512,
              py::arg("n_threads") = 0)
         .def_static("from_dict", [](const py::dict &d)
                     {   
                         HegemonikonGenerationParams params;
                         if (d.contains("n_predict")) {
                         params.n_predict = d["n_predict"].cast<int32_t>();
                         }
                         if (d.contains("temperature")) {
                              params.temperature = d["temperature"].cast<float>();
                         }
                         if (d.contains("top_k")) {
                                   params.top_k = d["top_k"].cast<int32_t>();
                         }
                         if (d.contains("top_p")) {
                                   params.top_p = d["top_p"].cast<float>();
                         }
                         if (d.contains("repeat_penalty")) {
                                   params.repeat_penalty = d["repeat_penalty"].cast<float>();
                         }
                         if (d.contains("penalty_last_n")) {
                                   params.penalty_last_n = d["penalty_last_n"].cast<int32_t>();
                         }
                         if (d.contains("penalty_freq")) {
                                   params.penalty_freq = d["penalty_freq"].cast<float>();
                         }
                         if (d.contains("penalty_present")) {
                                   params.penalty_present = d["penalty_present"].cast<float>();
                         }
                         if (d.contains("stop_sequences")) {
                         params.stop_sequences = d["stop_sequences"].cast<std::vector<std::string>>();
                         }
                         if (d.contains("n_batch")) {
                              params.n_batch = d["n_batch"].cast<int32_t>();   
                         }
                         if (d.contains("n_threads")) {
                                   params.n_threads = d["n_threads"].cast<int32_t>();
                         }
                         return params; })
         .def_readwrite("n_predict", &HegemonikonGenerationParams::n_predict)
         .def_readwrite("temperature", &HegemonikonGenerationParams::temperature)
         .def_readwrite("top_k", &HegemonikonGenerationParams::top_k)
         .def_readwrite("top_p", &HegemonikonGenerationParams::top_p)
         .def_readwrite("repeat_penalty", &HegemonikonGenerationParams::repeat_penalty)
         .def_readwrite("penalty_last_n", &HegemonikonGenerationParams::penalty_last_n)
         .def_readwrite("penalty_freq", &HegemonikonGenerationParams::penalty_freq)
         .def_readwrite("penalty_present", &HegemonikonGenerationParams::penalty_present)
         .def_readwrite("stop_sequences", &HegemonikonGenerationParams::stop_sequences)
         .def_readwrite("n_batch", &HegemonikonGenerationParams::n_batch)
         .def_readwrite("n_threads", &HegemonikonGenerationParams::n_threads)
         .def("__eq__", [](const HegemonikonGenerationParams &a, const HegemonikonGenerationParams &b)
              { return a == b; })
         .def("__ne__", [](const HegemonikonGenerationParams &a, const HegemonikonGenerationParams &b)
              { return a != b; })
         .def("__hash__", [](const HegemonikonGenerationParams &p)
              { return p.hash(); })
         .def("__str__", [](const HegemonikonGenerationParams &p)
              { return p.to_string(); });

     py::class_<HegemonikonWhisperModelParams>(m, "HegemonikonWhisperModelParams", "Parameters for loading a Whisper model.")
         .def(py::init<>())
         .def(py::init<const std::string &, const std::string &, bool, bool, int32_t, int32_t>(),
              py::arg("model"),
              py::arg("language"),
              py::arg("use_gpu") = true,
              py::arg("flash_attn") = false,
              py::arg("audio_ctx") = 0,
              py::arg("n_threads") = std::min(4, (int32_t)std::thread::hardware_concurrency()))
         .def_static("from_dict", [](const py::dict &d)
                     {
                              HegemonikonWhisperModelParams params;
                              if (d.contains("model")) {
                                   params.model = d["model"].cast<std::string>();
                              }
                                   if (d.contains("language")) {
                                        params.language = d["language"].cast<std::string>();
                                   }
                                   if (d.contains("use_gpu")) {
                                        params.use_gpu = d["use_gpu"].cast<bool>();
                                   }
                                   if (d.contains("flash_attn")) {
                                        params.flash_attn = d["flash_attn"].cast<bool>();
                                   }
                                   if (d.contains("audio_ctx")) {
                                        params.audio_ctx = d["audio_ctx"].cast<int32_t>();
                                   }
                                   if (d.contains("n_threads")) {
                                        params.n_threads = d["n_threads"].cast<int32_t>();
                                   }
                                   return params; })
         .def_readwrite("model", &HegemonikonWhisperModelParams::model, "Path to the Whisper GGUF model file.")
         .def_readwrite("language", &HegemonikonWhisperModelParams::language, "Language for the Whisper model (e.g., 'en', 'auto').")
         .def_readwrite("use_gpu", &HegemonikonWhisperModelParams::use_gpu, "Whether to use GPU for transcription.")
         .def_readwrite("flash_attn", &HegemonikonWhisperModelParams::flash_attn, "Whether to use flash attention for faster processing.")
         .def_readwrite("audio_ctx", &HegemonikonWhisperModelParams::audio_ctx, "Audio context size for the model.")
         .def_readwrite("n_threads", &HegemonikonWhisperModelParams::n_threads, "Number of threads to use for processing.")
         .def("__eq__", [](const HegemonikonWhisperModelParams &a, const HegemonikonWhisperModelParams &b)
              { return a == b; })
         .def("__ne__", [](const HegemonikonWhisperModelParams &a, const HegemonikonWhisperModelParams &b)
              { return a != b; })
         .def("__hash__", [](const HegemonikonWhisperModelParams &p)
              { return p.hash(); })
         .def("__str__", [](const HegemonikonWhisperModelParams &p)
              { return p.to_string(); });

     py::class_<HegemonikonWhisperGenerationParams>(m, "HegemonikonWhisperGenerationParams", "Parameters for Whisper audio transcription.")
         .def(py::init<>())
         .def(py::init<int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, bool, bool, bool, bool, int32_t, int32_t, bool, bool, bool, const std::string &>(),
              py::arg("step_ms_"),
              py::arg("length_ms_"),
              py::arg("keep_ms_"),
              py::arg("capture_id_"),
              py::arg("vad_thold_"),
              py::arg("freq_thold_"),
              py::arg("translate_") = false,
              py::arg("tinydiarize_") = false,
              py::arg("no_fallback_") = false,
              py::arg("no_context_") = false,
              py::arg("max_tokens_") = 0,
              py::arg("beam_size_") = 1,
              py::arg("print_special_") = false,
              py::arg("no_timestamps_") = false,
              py::arg("save_audio_") = false,
              py::arg("fname_out_") = "")
         .def_static("from_dict", [](const py::dict &d)
                     {
     HegemonikonWhisperGenerationParams params;    
     if (d.contains("step_ms")) {
           params.step_ms = d["step_ms"].cast<int32_t>();
     }
     if (d.contains("length_ms")) {
           params.length_ms = d["length_ms"].cast<int32_t>();
          }
     if (d.contains("keep_ms")) {
           params.keep_ms = d["keep_ms"].cast<int32_t>();   
     }
     if (d.contains("capture_id")) {
               params.capture_id = d["capture_id"].cast<int32_t>();
     }
     if (d.contains("vad_thold")) {
               params.vad_thold = d["vad_thold"].cast<float>();
     }
     if (d.contains("freq_thold")) {
               params.freq_thold = d["freq_thold"].cast<float>();
     }
     if (d.contains("translate")) {
               params.translate = d["translate"].cast<bool>();
     }
     if (d.contains("tinydiarize")) {
               params.tinydiarize = d["tinydiarize"].cast<bool>();
     }
     if (d.contains("no_fallback")) {
               params.no_fallback = d["no_fallback"].cast<bool>();
     }
     if (d.contains("no_context")) {
               params.no_context = d["no_context"].cast<bool>();
     }
     if (d.contains("max_tokens")) {
               params.max_tokens = d["max_tokens"].cast<int32_t>();
     }
     if (d.contains("beam_size")) {
               params.beam_size = d["beam_size"].cast<int32_t>();
     }
     if (d.contains("print_special")) {
               params.print_special = d["print_special"].cast<bool>();
     }
     if (d.contains("no_timestamps")) {
               params.no_timestamps = d["no_timestamps"].cast<bool>();
     }
     if (d.contains("save_audio")) {
               params.save_audio = d["save_audio"].cast<bool>();
     }
     if (d.contains("fname_out")) {
               params.fname_out = d["fname_out"].cast<std::string>();
     }
     return params; })
         .def_readwrite("step_ms", &HegemonikonWhisperGenerationParams::step_ms, "Step size in milliseconds for audio processing.")
         .def_readwrite("length_ms", &HegemonikonWhisperGenerationParams::length_ms, "Length of audio segments in milliseconds.")
         .def_readwrite("keep_ms", &HegemonikonWhisperGenerationParams::keep_ms, "Duration to keep audio in milliseconds.")
         .def_readwrite("capture_id", &HegemonikonWhisperGenerationParams::capture_id, "Capture device ID for audio input.")
         .def_readwrite("vad_thold", &HegemonikonWhisperGenerationParams::vad_thold, "Voice Activity Detection threshold.")
         .def_readwrite("freq_thold", &HegemonikonWhisperGenerationParams::freq_thold, "Frequency threshold for audio processing.")
         .def_readwrite("translate", &HegemonikonWhisperGenerationParams::translate, "Whether to translate the audio to English.")
         .def_readwrite("tinydiarize", &HegemonikonWhisperGenerationParams::tinydiarize, "Whether to use tiny diarization for speaker separation.")
         .def_readwrite("no_fallback", &HegemonikonWhisperGenerationParams::no_fallback, "Whether to disable fallback to non-diarized transcription.")
         .def_readwrite("no_context", &HegemonikonWhisperGenerationParams::no_context, "Whether to disable context for transcription.")
         .def_readwrite("max_tokens", &HegemonikonWhisperGenerationParams::max_tokens, "Maximum number of tokens to generate.")
         .def_readwrite("beam_size", &HegemonikonWhisperGenerationParams::beam_size, "Beam size for beam search decoding.")
         .def_readwrite("print_special", &HegemonikonWhisperGenerationParams::print_special, "Whether to print special tokens in the output.")
         .def_readwrite("no_timestamps", &HegemonikonWhisperGenerationParams::no_timestamps, "Whether to disable timestamps in the output.")
         .def_readwrite("save_audio", &HegemonikonWhisperGenerationParams::save_audio, "Whether to save the processed audio.")
         .def_readwrite("fname_out", &HegemonikonWhisperGenerationParams::fname_out, "Output filename for the processed audio.")
         .def("__eq__", [](const HegemonikonWhisperGenerationParams &a, const HegemonikonWhisperGenerationParams &b)
              { return a == b; })
         .def("__ne__", [](const HegemonikonWhisperGenerationParams &a, const HegemonikonWhisperGenerationParams &b)
              { return a != b; })
         .def("__hash__", [](const HegemonikonWhisperGenerationParams &p)
              { return p.hash(); })
         .def("__str__", [](const HegemonikonWhisperGenerationParams &p)
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
              py::arg("audio_file_path"), py::arg("whisper_model_params"))
         .def("tokenization", &CoreAIService::tokenization, "Tokenize text using Llama model parameters",
              py::arg("text"))
         .def("detokenization", &CoreAIService::detokenization, "Detokenize a list of tokens into text",
              py::arg("tokens"));

     py::class_<QuantizedModelInfo>(m, "QuantizedModelInfo", "Information about a quantized model.")
         .def(py::init<>())
         .def_readwrite("model_id", &QuantizedModelInfo::model_id, "Unique identifier for the model.")
         .def_readwrite("file_name", &QuantizedModelInfo::file_name, "File name of the quantized model.")
         .def_readwrite("last_modified", &QuantizedModelInfo::last_modified, "Last modified timestamp of the model file.")
         .def_readwrite("quantization", &QuantizedModelInfo::quantization, "Quantization type (e.g., 'Q4_0', 'Q8_0').")
         .def_readwrite("fileSize", &QuantizedModelInfo::fileSize, "Size of the model file in bytes.")
         .def_static("from_dict", [](const py::dict &d)
                     {
             QuantizedModelInfo info;
             if (d.contains("model_id")) {
                 info.model_id = d["model_id"].cast<std::string>();
             }
             if (d.contains("file_name")) {
                 info.file_name = d["file_name"].cast<std::string>();
             }
             if (d.contains("last_modified")) {
                 info.last_modified = d["last_modified"].cast<std::string>();
             }
             if (d.contains("quantization")) {
                 info.quantization = d["quantization"].cast<std::string>();
             }
             if (d.contains("fileSize")) {
                 info.fileSize = d["fileSize"].cast<size_t>();
             }
             return info; })
         .def("is_valid", &QuantizedModelInfo::isValid, "Check if the model info is valid (non-empty model_id and file_name).")
         .def("__str__", [](const QuantizedModelInfo &info)
              { return info.to_string(); })
         .def("__hash__", [](const QuantizedModelInfo &info)
              { return info.hash(); })
         .def("__repr__", [](const QuantizedModelInfo &info)
              { return info.to_string(); })
         .def("__eq__", [](const QuantizedModelInfo &a, const QuantizedModelInfo &b)
              { return a == b; })
         .def("__ne__", [](const QuantizedModelInfo &a, const QuantizedModelInfo &b)
              { return a != b; });

     py::class_<BenchmarkMetrics>(m, "BenchmarkMetrics", "Metrics collected during model benchmarking.")
         .def(py::init<>())
         .def_readwrite("load_time_ms", &BenchmarkMetrics::load_time_ms, "Time taken to load the model in milliseconds.")
         .def_readwrite("generation_time", &BenchmarkMetrics::generation_time, "Time taken for text generation in seconds.")
         .def_readwrite("total_time", &BenchmarkMetrics::total_time, "Total time for the benchmark in seconds.")
         .def_readwrite("tokens_generated", &BenchmarkMetrics::tokens_generated, "Number of tokens generated during the benchmark.")
         .def_readwrite("tokens_per_second", &BenchmarkMetrics::tokens_per_second, "Average tokens generated per second.")
         .def_readwrite("memory_usage", &BenchmarkMetrics::memory_usage, "Memory usage during the benchmark in MB.")
         .def_readwrite("success", &BenchmarkMetrics::success, "Whether the benchmark was successful.")
         .def_readwrite("errorMessage", &BenchmarkMetrics::errorMessage, "Error message if the benchmark failed.")
         .def_readwrite("generation_times", &BenchmarkMetrics::generation_times, "List of generation times for each run in seconds.")
         .def_readwrite("tokens_per_second_history", &BenchmarkMetrics::tokens_per_second_history, "List of tokens per second for each run.")
         .def_readwrite("avg_ttft_ms", &BenchmarkMetrics::avg_ttft_ms, "Average time to first token in milliseconds.")
         .def_readwrite("avg_prefill_ms", &BenchmarkMetrics::avg_prefill_ms, "Average prefill time in milliseconds.")
         .def_readwrite("avg_decode_tps", &BenchmarkMetrics::avg_decode_tps, "Average decode tokens per second.")
         .def_readwrite("avg_end_to_end_latency_ms", &BenchmarkMetrics::avg_end_to_end_latency_ms, "Average end-to-end latency in milliseconds.")
         .def_readwrite("ttft_history", &BenchmarkMetrics::ttft_history, "List of time to first token for each run in milliseconds.")
         .def_readwrite("end_to_end_latency_history", &BenchmarkMetrics::end_to_end_latency_history, "List of end-to-end latencies for each run in milliseconds.");

     py::class_<HegemonikonBenchmarkResult>(m, "HegemonikonBenchmarkResult", "Result of a model benchmark.")
         .def(py::init<const std::string &>(), "Constructor with model ID")
         .def_readwrite("model_id", &HegemonikonBenchmarkResult::model_id, "ID of the model being benchmarked.")
         .def_readwrite("metrics", &HegemonikonBenchmarkResult::metrics, "Metrics collected during the benchmark.")
         .def_readwrite("generated_text", &HegemonikonBenchmarkResult::generated_text, "Text generated during the benchmark.")
         .def_readwrite("promptUsed", &HegemonikonBenchmarkResult::promptUsed, "Prompt used for the benchmark.")
         .def("calculate_statistics", &HegemonikonBenchmarkResult::calculateStatistics, "Calculate statistical summaries from the benchmark metrics.");

     py::class_<HegemonikonBenchmarkParams>(m, "HegemonikonBenchmarkParams", "Parameters for benchmarking models.")
         .def(py::init<>())
         .def(py::init<int, int, bool, const HegemonikonGenerationParams &>(), "Constructor with parameters",
              py::arg("n_gpu_layers") = 0,
              py::arg("repetitions") = 10,
              py::arg("warmup") = true,
              py::arg("generation_params") = HegemonikonGenerationParams())
         .def_readwrite("n_gpu_layers", &HegemonikonBenchmarkParams::n_gpu_layers, "Number of GPU layers to use during benchmarking.")
         .def_readwrite("repetitions", &HegemonikonBenchmarkParams::repetitions, "Number of times to repeat the benchmark.")
         .def_readwrite("warmup", &HegemonikonBenchmarkParams::warmup, "Whether to perform a warmup run before benchmarking.")
         .def_readwrite("generation_params", &HegemonikonBenchmarkParams::generation_params, "Generation parameters to use during benchmarking.");

     py::class_<HegemonikonLlamaBenchmarker>(m, "HegemonikonLlamaBenchmarker", "Benchmarks LLM models for performance and metrics.")
         .def(py::init<>(), "Default constructor")
         .def(py::init<std::vector<QuantizedModelInfo>, std::vector<std::string>>(), "Constructor with model and prompt lists",
              py::arg("models"), py::arg("prompts"))
         .def("benchmark_single_model", &HegemonikonLlamaBenchmarker::benchmarkSingleModel, "Benchmark a single LLM model",
              py::arg("quantized_model_info"), py::arg("benchmark_params"), py::arg("llama_model_params"));

     py::class_<SecureKey>(m, "SecureKey", "A C++ class to hold sensitive data (like encryption keys) in locked memory.")
         .def("data", [](const SecureKey &self)
              { return py::bytes(reinterpret_cast<const char *>(self.data()), self.size()); }, "Returns the key data as a Python bytes object.")
         .def("size", &SecureKey::size, "Returns the size of the key in bytes.");

     py::class_<SecureString>(m, "SecureString", "A C++ class to hold sensitive strings in locked memory.")
         .def(py::init([](py::bytes b)
                       { return new SecureString(std::string(b)); }),
              "Constructor from a bytes object");

     m.def("derive_and_protect_key", [](const SecureString &secure_password, const py::bytes &salt_bytes)
           {

                char *salt_buffer;
                ssize_t salt_length;
                if (PYBIND11_BYTES_AS_STRING_AND_SIZE(salt_bytes.ptr(), &salt_buffer, &salt_length))
                {
                     throw std::runtime_error("Unable to process salt bytes");
                }
                std::vector<uint8_t> salt(salt_buffer, salt_buffer + salt_length);

                return derive_and_protect_key(secure_password, salt); }, py::arg("password"), py::arg("salt"), "Derives a key from a password using Argon2id and returns it in a protected object.");
};