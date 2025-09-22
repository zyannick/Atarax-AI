#pragma once
#include <string>
#include <vector>

struct HegemonikonBenchmarkResult {
    std::string model_name;
    double prompt_eval_ms;
    double generation_eval_ms;
    double sample_ms;
    double total_time_ms;
};

std::string run_benchmark(const std::string& model_path, int threads, int n_tokens);
HegemonikonBenchmarkResult parse_output(const std::string& output);
void write_report(const std::vector<HegemonikonBenchmarkResult>& results, const std::string& output_path);
