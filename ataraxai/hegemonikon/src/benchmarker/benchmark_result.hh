#pragma once
#include <llama.h>
#include <string>
#include <filesystem>

#include "system_infos/cpu_info.hh"
#include "system_infos/gpu_info.hh"
#include "system_infos/memory_usage.hh"

struct BenchmarkResult
{
    std::string model_name;
    float load_time_ms;
    float pps;
    float token_gen_time_ms;
    float time_to_first_token_ms;
    float sample_time_ms;
    float peak_ram_bytes;
    long ram_peak_kb;
    float perplexity;
    float latency_ms;
    float throughput;
    float ttft_ms;
};

struct BenchmarkSettings
{
    std::string model_path;
    std::string input_text;
    int n_threads;
    int n_past;
    int n_predict;
    bool use_mlock;
    
};

BenchmarkResult benchmark_model(const std::string &model_path, llama_model_params &model_params, const llama_context_params &cparams, const std::string &prompt_text, int tokens_to_generate)
{
    CPUInfoCollection cpu_info_list = CPUInfoCollection();
    GPUInfoCollection gpu_info_list = GPUInfoCollection();

    BenchmarkResult result;
    result.model_name = std::filesystem::path(model_path).filename().string();

    long ram_before_load = get_current_memory_usage();

    ggml_backend_load_all();

} 