#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>

#include "utils/utils.hh"

#include "json.hpp"

using namespace std::chrono;
using json = nlohmann::json;

struct QuantizedModelInfo
{
    std::string modelId;
    std::string fileName;
    std::string lastModified;
};

struct ModelBenchmarker
{
    std::vector<QuantizedModelInfo> quantized_models;
    std::string json_file_model_paths;
    std::vector<std::string> input_texts;

    ModelBenchmarker()
    {
    }

    ModelBenchmarker(const std::string &json_file)
    {
        json_file_model_paths = json_file;
        load_model_paths_from_json(json_file);
        input_texts = {
            "Testing my local model.",
            "What is the meaning of life?",
            "How do you feel today?",
            "What is the capital of Burkina Faso?",
        };
    }

    void load_model_paths_from_json(const std::string &json_file)
    {
        std::ifstream file(json_file);
        if (!file.is_open())
        {
            std::cerr << "Error opening JSON file: " << json_file << std::endl;
            return;
        }

        std::ifstream f(json_file);
        json data = json::parse(f);

        for (auto& [model_key, model_val] : data.items()) {
            QuantizedModelInfo model_info;
            model_info.modelId = model_val["modelID"];
            model_info.fileName = model_val["fileName"];
            model_info.lastModified = model_val["lastModified"];
            quantized_models.push_back(model_info);
        }
    }

    void benchmark_models(int n_threads, int n_predict)
    {
        std::cout << "Benchmarking models... " <<  quantized_models.size() << std::endl;
        std::cout << "Number of threads: " << n_threads << std::endl;
        std::cout << "Number of predictions: " << n_predict << std::endl;
        for (const QuantizedModelInfo &quantized_model_info : quantized_models)
        {
            for (const auto &input_text : input_texts)
            {
                std::string result = llama_bench_model(quantized_model_info, input_text, n_threads, n_predict);
                std::cout << result << std::endl;
            }
        }
    }

    std::string llama_bench_model(QuantizedModelInfo quantized_model_info, const std::string &input_text, int n_threads, int n_predict)
    {
        const char* env_path = std::getenv("ATARAXIA_PATH");

        if (!env_path) {
            std::cerr << "Environment variable ATARAXIA_PATH not set!" << std::endl;
            return "";
        }

        std::string model_file = std::string(env_path) + "/" + quantized_model_info.fileName;
        std::string cmd = "third_party/llama.cpp/build/bin/llama-bench -m \"" + model_file + "\" -t " + std::to_string(n_threads) + " -n " + std::to_string(n_predict);
        std::string command = cmd;
        std::array<char, 128> buffer;
        std::string result;
        FILE *pipe = popen(command.c_str(), "r");
        if (!pipe)
            throw std::runtime_error("Failed to run command");
        while (fgets(buffer.data(), buffer.size(), pipe) != nullptr)
        {
            result += buffer.data();
        }
        pclose(pipe);
        return result;
    }
};