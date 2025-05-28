#include "core_ai/core_ai.hh"
#include "core_ai/json.hpp"
#include <fstream>
#include <iostream>
#include <string>

#include "llama.h"

#include <iostream>
#include <cstdlib>

using json = nlohmann::json;

int main(int argc, char **argv)
{
    const char *env_path = std::getenv("ATARAXIA_PATH") ? std::getenv("ATARAXIA_PATH") : "..";

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

    CPUInfoCollection cpu_info_collection = CPUInfoCollection();
    GPUInfoCollection gpu_info_collection = GPUInfoCollection();

    std::cout << "Collecting CPU information..." << std::endl;
    std::cout << "-------------------------------------------------------------------------------------------------" << std::endl;

    for (const auto &cpu : cpu_info_collection.cpus)
    {
        std::cout << cpu << std::endl;
    }

    std::cout << "Collecting GPU information..." << std::endl;
    std::cout << "-------------------------------------------------------------------------------------------------" << std::endl;

    for (const auto &gpu : gpu_info_collection.gpus)
    {
        std::cout << gpu << std::endl;
    }

    std::cout << "--------------------------------------------------------------------------------------------------" << std::endl;

    std::cout << "Start benchmarking models..." << std::endl;
    ModelBenchmarker model_benchmarker = ModelBenchmarker(model_jsons_path);
    model_benchmarker.benchmark_models(8, 256);

    return 0;
}