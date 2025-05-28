#include "core_ai/core_ai.hh"
#include "core_ai/json.hpp"
#include <fstream>
#include <iostream>
#include <string>

#include "llama.h"

#include <iostream>
#include <cstdlib>


using json = nlohmann::json;

/* 
g++ main.cc -o ataraxia \
    -Ithird_party/llama.cpp/include \
    -Ithird_party/llama.cpp/ggml/include \
    -I. \
    -Icore_ai \
    -L../build/third_party/llama.cpp/build/bin \
    -lllava_shared \
    -Wl,-rpath=../build/third_party/llama.cpp/build/bin \
    -std=c++17
     */
int main(int argc, char **argv)
{
    const char* env_path = std::getenv("ATARAXIA_PATH") ? std::getenv("ATARAXIA_PATH") : "..";

    if (!env_path) {
        std::cerr << "Environment variable ATARAXIA_PATH not set!" << std::endl;
        return 1;
    }

    std::string model_jsons_path = std::string(env_path) + "/data/last_models/text.json";

    // Check if the file exists
    if (!std::filesystem::exists(model_jsons_path)) {
        std::cerr << "File does not exist: " << model_jsons_path << std::endl;
        return 1;
    }

    std::cout << "Full path: " << model_jsons_path << std::endl;

    ModelBenchmarker model_benchmarker = ModelBenchmarker(model_jsons_path);
    model_benchmarker.benchmark_models(8, 100);

    return 0;
}