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
     params.model_path = d.attr("get")("model_path", "").cast<std::string>();
     params.n_ctx = d.attr("get")("n_ctx", 2048).cast<int32_t>();
     params.n_gpu_layers = d.attr("get")("n_gpu_layers", 0).cast<int32_t>();
     params.main_gpu = d.attr("get")("main_gpu", 0).cast<int32_t>();
     params.n_batch = d.attr("get")("n_batch", 1).cast<int32_t>();
     params.tensor_split = d.attr("get")("tensor_split", false).cast<bool>();
     params.vocab_only = d.attr("get")("vocab_only", false).cast<bool>();
     params.use_map = d.attr("get")("use_map", false).cast<bool>();
     params.use_mlock = d.attr("get")("use_mlock", false).cast<bool>();
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
                         params.n_predict = d.attr("get")("n_predict", 128).cast<int32_t>();
                         params.temperature = d.attr("get")("temperature", 0.8f).cast<float>();
                         params.top_k = d.attr("get")("top_k", 40).cast<int32_t>();
                         params.top_p = d.attr("get")("top_p", 0.95f).cast<float>();
                         params.repeat_penalty = d.attr("get")("repeat_penalty", 1.1f).cast<float>();
                         params.penalty_last_n = d.attr("get")("penalty_last_n", 64).cast<int32_t>();
                         params.penalty_freq = d.attr("get")("penalty_freq", 0.0f).cast<float>();
                         params.penalty_present = d.attr("get")("penalty_present", 0.0f).cast<float>();
                         params.stop_sequences = d.attr("get")("stop_sequences", std::vector<std::string>{}).cast<std::vector<std::string>>();
                         params.n_batch = d.attr("get")("n_batch", 512).cast<int32_t>();
                         params.n_threads = d.attr("get")("n_threads", 0).cast<int32_t>();
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
                              params.model = d.attr("get")("model", "").cast<std::string>();
                              params.language = d.attr("get")("language", "en").cast<std::string>();
                              params.use_gpu = d.attr("get")("use_gpu", true).cast<bool>();
                              params.flash_attn = d.attr("get")("flash_attn", false).cast<bool>();
                              params.audio_ctx = d.attr("get")("audio_ctx", 0).cast<int32_t>();
                              params.n_threads = d.attr("get")("n_threads", std::min(4, (int32_t)std::thread::hardware_concurrency())).cast<int32_t>();
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
     params.step_ms = d.attr("get")("step_ms", 300).cast<int32_t>();
     params.length_ms = d.attr("get")("length_ms", 1000).cast<int32_t>();
     params.keep_ms = d.attr("get")("keep_ms", 3000).cast<int32_t>();
     params.capture_id = d.attr("get")("capture_id", 0).cast<int32_t>();
     params.vad_thold = d.attr("get")("vad_thold", 200).cast<int32_t>();
     params.freq_thold = d.attr("get")("freq_thold", 0).cast<int32_t>();
     params.translate = d.attr("get")("translate", false).cast<bool>();
     params.tinydiarize = d.attr("get")("tinydiarize", false).cast<bool>();
     params.no_fallback = d.attr("get")("no_fallback", false).cast<bool>();
     params.no_context = d.attr("get")("no_context", false).cast<bool>();
     params.max_tokens = d.attr("get")("max_tokens", 0).cast<int32_t>();
     params.beam_size = d.attr("get")("beam_size", 1).cast<int32_t>();
     params.print_special = d.attr("get")("print_special", false).cast<bool>();
     params.no_timestamps = d.attr("get")("no_timestamps", false).cast<bool>();
     params.save_audio = d.attr("get")("save_audio", false).cast<bool>();
     params.fname_out = d.attr("get")("fname_out", "").cast<std::string>();
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
         .def("is_llama_model_loaded", &CoreAIService::is_llama_model_loaded)
         .def("process_prompt", &CoreAIService::process_prompt, "Process a text prompt using the Llama model",
              py::arg("prompt_text"), py::arg("llama_generation_params"))
         .def("stream_prompt", &CoreAIService::stream_prompt, "Stream generation of text from a prompt",
              py::arg("prompt_text"), py::arg("llama_generation_params"), py::arg("callback"))
         .def("initialize_whisper_model", &CoreAIService::initialize_whisper_model, "Initialize and load the Whisper model",
              py::arg("whisper_model_params"))
         .def("unload_whisper_model", &CoreAIService::unload_whisper_model, "Unload the currently loaded Whisper model")
         .def("is_whisper_model_loaded", &CoreAIService::is_whisper_model_loaded)
         .def("transcribe_audio_pcm", &CoreAIService::transcribe_audio_pcm, "Transcribe PCM audio data using Whisper",
              py::arg("pcm_f32_data"), py::arg("whisper_model_params"))
         .def("transcribe_audio_file", &CoreAIService::transcribe_audio_file, "Transcribe an audio file using Whisper",
              py::arg("audio_file_path"), py::arg("whisper_model_params"))
         .def("tokenization", &CoreAIService::tokenization, "Tokenize text using Llama model parameters",
              py::arg("text"))
         .def("detokenization", &CoreAIService::detokenization, "Detokenize a list of tokens into text",
              py::arg("tokens"));
              
     py::class_<HegemonikonQuantizedModelInfo>(m, "HegemonikonQuantizedModelInfo", "Information about a quantized model.")
         .def(py::init<>())
         .def_readwrite("model_id", &HegemonikonQuantizedModelInfo::model_id, "Unique identifier for the model.")
         .def_readwrite("local_path", &HegemonikonQuantizedModelInfo::local_path, "File name of the quantized model.")
         .def_readwrite("last_modified", &HegemonikonQuantizedModelInfo::last_modified, "Last modified timestamp of the model file.")
         .def_readwrite("quantization", &HegemonikonQuantizedModelInfo::quantization, "Quantization type (e.g., 'Q4_0', 'Q8_0').")
         .def_readwrite("fileSize", &HegemonikonQuantizedModelInfo::fileSize, "Size of the model file in bytes.")
         .def_static("from_dict", [](const py::dict &d)
                     {
             HegemonikonQuantizedModelInfo info;
             info.model_id = d.attr("get")("model_id", "").cast<std::string>();
             info.local_path = d.attr("get")("local_path", "").cast<std::string>();
             info.last_modified = d.attr("get")("last_modified", 0).cast<std::string>();
             info.quantization = d.attr("get")("quantization", "").cast<std::string>();
             info.fileSize = d.attr("get")("fileSize", 0).cast<int64_t>();
             return info; })
         .def("is_valid", &HegemonikonQuantizedModelInfo::isValid, "Check if the model info is valid (non-empty model_id and local_path).")
         .def("__str__", [](const HegemonikonQuantizedModelInfo &info)
              { return info.to_string(); })
         .def("__hash__", [](const HegemonikonQuantizedModelInfo &info)
              { return info.hash(); })
         .def("__repr__", [](const HegemonikonQuantizedModelInfo &info)
              { return info.to_string(); })
         .def("__eq__", [](const HegemonikonQuantizedModelInfo &a, const HegemonikonQuantizedModelInfo &b)
              { return a == b; })
         .def("__ne__", [](const HegemonikonQuantizedModelInfo &a, const HegemonikonQuantizedModelInfo &b)
              { return a != b; });

     py::class_<HegemonikonBenchmarkMetrics>(m, "HegemonikonBenchmarkMetrics", "Metrics collected during model benchmarking.")
         .def(py::init<>())
         .def_static("from_dict", [](const py::dict &d)
                     {
     HegemonikonBenchmarkMetrics metrics;
     metrics.load_time_ms = d.attr("get")("load_time_ms", 0).cast<int64_t>();
     metrics.generation_time_ms = d.attr("get")("generation_time_ms", 0.0).cast<float>();
     metrics.total_time_ms = d.attr("get")("total_time_ms", 0.0).cast<float>();
     metrics.tokens_generated = d.attr("get")("tokens_generated", 0).cast<int64_t>();
     metrics.tokens_per_second = d.attr("get")("tokens_per_second", 0.0).cast<float>();
     metrics.memory_usage_mb = d.attr("get")("memory_usage_mb", 0.0).cast<float>();
     metrics.success = d.attr("get")("success", false).cast<bool>();
     metrics.errorMessage = d.attr("get")("errorMessage", "").cast<std::string>();
     metrics.generation_time_history_ms = d.attr("get")("generation_time_history_ms", std::vector<float>{}).cast<std::vector<float>>();
     metrics.tokens_per_second_history = d.attr("get")("tokens_per_second_history", std::vector<float>{}).cast<std::vector<float>>();
     metrics.avg_ttft_ms = d.attr("get")("avg_ttft_ms", 0.0).cast<float>();
     metrics.avg_decode_time_ms = d.attr("get")("avg_decode_time_ms", 0.0).cast<float>();
     metrics.avg_end_to_end_time_latency_ms = d.attr("get")("avg_end_to_end_time_latency_ms", 0.0).cast<float>();
     return metrics; })
         .def_readwrite("load_time_ms", &HegemonikonBenchmarkMetrics::load_time_ms, "Time taken to load the model in milliseconds.")
         .def_readwrite("generation_time_ms", &HegemonikonBenchmarkMetrics::generation_time_ms, "Time taken for text generation in seconds.")
         .def_readwrite("total_time_ms", &HegemonikonBenchmarkMetrics::total_time_ms, "Total time for the benchmark in seconds.")
         .def_readwrite("tokens_generated", &HegemonikonBenchmarkMetrics::tokens_generated, "Number of tokens generated during the benchmark.")
         .def_readwrite("tokens_per_second", &HegemonikonBenchmarkMetrics::tokens_per_second, "Average tokens generated per second.")
         .def_readwrite("memory_usage_mb", &HegemonikonBenchmarkMetrics::memory_usage_mb, "Memory usage during the benchmark in MB.")
         .def_readwrite("success", &HegemonikonBenchmarkMetrics::success, "Whether the benchmark was successful.")
         .def_readwrite("errorMessage", &HegemonikonBenchmarkMetrics::errorMessage, "Error message if the benchmark failed.")
         .def_readwrite("generation_time_history_ms", &HegemonikonBenchmarkMetrics::generation_time_history_ms, "List of generation times for each run in seconds.")
         .def_readwrite("tokens_per_second_history", &HegemonikonBenchmarkMetrics::tokens_per_second_history, "List of tokens per second for each run.")
         .def_readwrite("avg_ttft_ms", &HegemonikonBenchmarkMetrics::avg_ttft_ms, "Average time to first token in milliseconds.")
         .def_readwrite("avg_decode_time_ms", &HegemonikonBenchmarkMetrics::avg_decode_time_ms, "Average decode time in milliseconds.")
         .def_readwrite("avg_end_to_end_time_latency_ms", &HegemonikonBenchmarkMetrics::avg_end_to_end_time_latency_ms, "Average end-to-end latency in milliseconds.")
         .def_readwrite("ttft_history_ms", &HegemonikonBenchmarkMetrics::ttft_history_ms, "List of time to first token for each run in milliseconds.")
         .def_readwrite("end_to_end_latency_history_ms", &HegemonikonBenchmarkMetrics::end_to_end_latency_history_ms, "List of end-to-end latencies for each run in milliseconds.")
         .def_readwrite("decode_times_history_ms", &HegemonikonBenchmarkMetrics::decode_times_history_ms, "List of decode times for each run in milliseconds.")
         .def_readwrite("p50_latency_ms", &HegemonikonBenchmarkMetrics::p50_latency_ms, "50th percentile latency in milliseconds.")
         .def_readwrite("p95_latency_ms", &HegemonikonBenchmarkMetrics::p95_latency_ms, "95th percentile latency in milliseconds.")
         .def_readwrite("p99_latency_ms", &HegemonikonBenchmarkMetrics::p99_latency_ms, "99th percentile latency in milliseconds.");

     py::class_<HegemonikonBenchmarkResult>(m, "HegemonikonBenchmarkResult", "Result of a model benchmark.")
         .def(py::init<const std::string &>(), "Constructor with model ID")
         .def_readwrite("metrics", &HegemonikonBenchmarkResult::metrics, "Parameters used for the benchmark.")
     //     .def_readwrite("benchmark_params", &HegemonikonBenchmarkResult::benchmark_params, "Benchmark parameters used during the benchmark.")
     //     .def_readwrite("llama_model_params", &HegemonikonBenchmarkResult::llama_model_params, "Quantized model information used during the benchmark.")
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
         .def_static("from_dict", [](const py::dict &d)
                     {   
                              HegemonikonBenchmarkParams params;
                              params.n_gpu_layers = d.attr("get")("n_gpu_layers", 0).cast<int>();
                              params.repetitions = d.attr("get")("repetitions", 10).cast<int>();
                              params.warmup = d.attr("get")("warmup", true).cast<bool>();
                              if (d.contains("generation_params") && py::isinstance<py::dict>(d["generation_params"]))
                              {
                                  params.generation_params = HegemonikonGenerationParams();
                                  params.generation_params.n_predict = d["generation_params"].attr("get")("n_predict", 128).cast<int32_t>();
                                  params.generation_params.temperature = d["generation_params"].attr("get")("temperature", 0.8f).cast<float>();
                                  params.generation_params.top_k = d["generation_params"].attr("get")("top_k", 40).cast<int32_t>();
                                  params.generation_params.top_p = d["generation_params"].attr("get")("top_p", 0.95f).cast<float>();
                                  params.generation_params.repeat_penalty = d["generation_params"].attr("get")("repeat_penalty", 1.1f).cast<float>();
                                  params.generation_params.penalty_last_n = d["generation_params"].attr("get")("penalty_last_n", 64).cast<int32_t>();
                                  params.generation_params.penalty_freq = d["generation_params"].attr("get")("penalty_freq", 0.0).cast<float>();
                                  params.generation_params.penalty_present = d["generation_params"].attr("get")("penalty_present", 0.0).cast<float>();
                                  params.generation_params.stop_sequences = d["generation_params"].attr("get")("stop_sequences", std::vector<std::string>{}).cast<std::vector<std::string>>();
                                  params.generation_params.n_batch = d["generation_params"].attr("get")("n_batch", 512).cast<int32_t>();
                                  params.generation_params.n_threads = d["generation_params"].attr("get")("n_threads", 0).cast<int32_t>();
                              }
                              return params; })
         .def_readwrite("n_gpu_layers", &HegemonikonBenchmarkParams::n_gpu_layers, "Number of GPU layers to use during benchmarking.")
         .def_readwrite("repetitions", &HegemonikonBenchmarkParams::repetitions, "Number of times to repeat the benchmark.")
         .def_readwrite("warmup", &HegemonikonBenchmarkParams::warmup, "Whether to perform a warmup run before benchmarking.")
         .def_readwrite("generation_params", &HegemonikonBenchmarkParams::generation_params, "Generation parameters to use during benchmarking.");

     py::class_<HegemonikonLlamaBenchmarker>(m, "HegemonikonLlamaBenchmarker", "Benchmarks LLM models for performance and metrics.")
         .def(py::init<>(), "Default constructor")
         .def("benchmark_single_model", &HegemonikonLlamaBenchmarker::benchmarkSingleModel, "Benchmark a single LLM model",
              py::arg("quantized_model_info"), py::arg("benchmark_params"), py::arg("llama_model_params"), 
              "Runs a benchmark for a single model.",
              py::call_guard<py::gil_scoped_release>())
          .def("request_cancellation", &HegemonikonLlamaBenchmarker::requestCancellation, "Request cancellation of an ongoing benchmark.");

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