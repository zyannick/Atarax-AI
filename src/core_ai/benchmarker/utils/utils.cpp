#include "utils.hpp"
#include <cstdlib>
#include <cstdio>
#include <sstream>
#include <regex>
#include <fstream>

std::string run_benchmark(const std::string& model_path, int threads, int n_tokens) {
    std::ostringstream cmd;
    cmd << "./llama-bench -m \"" << model_path << "\" -t " << threads << " -n " << n_tokens;
    std::string command = cmd.str();

    std::array<char, 128> buffer;
    std::string result;
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) throw std::runtime_error("Failed to run command");

    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
        result += buffer.data();
    }
    pclose(pipe);
    return result;
}

BenchmarkResult parse_output(const std::string& output) {
    BenchmarkResult result;
    std::smatch match;

    std::regex prompt_re(R"(prompt eval time =\s+([\d.]+))");
    std::regex eval_re(R"(eval time =\s+([\d.]+))");
    std::regex sample_re(R"(sample time =\s+([\d.]+))");
    std::regex total_re(R"(total time =\s+([\d.]+))");

    if (std::regex_search(output, match, prompt_re)) result.prompt_eval_ms = std::stod(match[1]);
    if (std::regex_search(output, match, eval_re)) result.generation_eval_ms = std::stod(match[1]);
    if (std::regex_search(output, match, sample_re)) result.sample_ms = std::stod(match[1]);
    if (std::regex_search(output, match, total_re)) result.total_time_ms = std::stod(match[1]);

    return result;
}

void write_report(const std::vector<BenchmarkResult>& results, const std::string& output_path) {
    std::ofstream file(output_path);
    file << "# LLaMA Model Benchmark Report\n\n";

    for (const auto& r : results) {
        file << "## Model: `" << r.model_name << "`\n";
        file << "- Prompt eval time: " << r.prompt_eval_ms << " ms\n";
        file << "- Generation eval time: " << r.generation_eval_ms << " ms\n";
        file << "- Sampling time: " << r.sample_ms << " ms\n";
        file << "- Total time: " << r.total_time_ms << " ms\n\n";
    }
}
