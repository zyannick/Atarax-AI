
#include <fstream>
#include <iostream>
#include <string>
#include <cstring>
#include <filesystem>
#include <system_error>
#include "llama.h"

#include <iostream>
#include <cstdlib>

#include "core_ai.hh"
#include "json.hpp"
#include "io_utils/directory_utils.hh"

using json = nlohmann::json;
namespace fs = std::filesystem;

int main(int argc, char **argv)
{
    const char *env_path = std::getenv("ATARAXIA_PATH") ? std::getenv("ATARAXIA_PATH") : "..";
    std::string output_path = std::string(env_path) + "/" + "output";
    create_directory(output_path);
    setenv("ATARAXIA_OUTPUT_DIR", output_path.c_str(), 1);

    if (!env_path)
    {
        std::cerr << "Environment variable ATARAXIA_PATH not set!" << std::endl;
        return 1;
    }

    std::string model_jsons_path = std::string(env_path) + "/data/last_models/text.json";

    if (!std::filesystem::exists(model_jsons_path))
    {
        std::cerr << "File does not exist: " << model_jsons_path << std::endl;
        return 1;
    }

    std::cout << "Full path: " << model_jsons_path << std::endl;

    LlamaBenchmarker benchmarker(model_jsons_path);

    BenchmarkParams params;
    params.n_gpu_layers = 32;
    params.repetitions = 10;
    params.warmup = true;
    params.parallel = false;

    benchmarker.setBenchmarkParams(params);

    auto results = benchmarker.benchmarkAllModels(params);

    benchmarker.exportResults(results, "benchmark_results.json");

    return 0;
}