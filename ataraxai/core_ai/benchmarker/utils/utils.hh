#pragma once
#include <string>
#include <vector>

struct BenchmarkResult {
    std::string model_name;
    double prompt_eval_ms;
    double generation_eval_ms;
    double sample_ms;
    double total_time_ms;
};

std::string run_benchmark(const std::string& model_path, int threads, int n_tokens);
BenchmarkResult parse_output(const std::string& output);
void write_report(const std::vector<BenchmarkResult>& results, const std::string& output_path);
