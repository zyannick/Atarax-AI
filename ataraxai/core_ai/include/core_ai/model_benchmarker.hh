#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <memory>
#include <iomanip>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <thread>
#include <future>

#include "core_ai/llama_interface.hh"
#include "json.hpp"

using namespace std::chrono;
using json = nlohmann::json;

template <typename T>
T avg(const std::vector<T>& v) {
    if (v.empty()) return T(0);
    return std::accumulate(v.begin(), v.end(), T(0)) / v.size();
}

template <typename T>
T stdev(const std::vector<T>& v) {
    if (v.size() <= 1) return T(0);
    T mean = avg(v);
    T sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), T(0));
    return std::sqrt(sq_sum / (v.size() - 1) - mean * mean * v.size() / (v.size() - 1));
}

struct QuantizedModelInfo
{
    std::string modelId;
    std::string fileName;
    std::string lastModified;
    std::string quantization;
    size_t fileSize = 0;  

    bool isValid() const { return !modelId.empty() && !fileName.empty(); }
};

struct BenchmarkMetrics
{
    double loadTime = 0.0;
    double generationTime = 0.0;
    double totalTime = 0.0;
    int tokensGenerated = 0;
    double tokensPerSecond = 0.0;
    double memoryUsage = 0.0;    
    bool success = false;
    std::string errorMessage;
    
    std::vector<double> generationTimes;
    std::vector<double> tokensPerSecondHistory;
};

struct BenchmarkResult
{
    std::string modelId;
    BenchmarkMetrics metrics;
    std::string generatedText;
    std::string promptUsed;

    BenchmarkResult(const std::string &id) : modelId(id) {}
    
    void calculateStatistics() {
        if (!metrics.generationTimes.empty()) {
            metrics.generationTime = avg(metrics.generationTimes);
        }
        if (!metrics.tokensPerSecondHistory.empty()) {
            metrics.tokensPerSecond = avg(metrics.tokensPerSecondHistory);
        }
    }
};

struct BenchmarkParams
{
    int n_gpu_layers = 0;
    int n_prompt = 512;
    int n_gen = 128;
    int n_threads = 4;
    int repetitions = 5;
    bool warmup = true;           
    bool parallel = false;        
    bool detailed_stats = false; 
    float temperature = 0.7f;
    int top_k = 40;
    float top_p = 0.9f;
};

class LlamaBenchmarker
{
private:
    std::vector<QuantizedModelInfo> quantized_models;
    std::vector<std::string> benchmark_prompts;
    std::string ataraxia_path;
    BenchmarkParams default_params;

public:
    LlamaBenchmarker()
    {
        initializeDefaultPrompts();
        const char *env_path = std::getenv("ATARAXIA_PATH");
        if (env_path) {
            ataraxia_path = std::string(env_path);
        } else {
            std::cerr << "Warning: ATARAXIA_PATH not set. Model paths must be absolute." << std::endl;
        }
    }

    explicit LlamaBenchmarker(const std::string &json_file) : LlamaBenchmarker()
    {
        loadModelPathsFromJson(json_file);
    }

    void initializeDefaultPrompts()
    {
        benchmark_prompts = {
            "What are the main advantages of using C++ for system programming?",
            "Where is Ouagadougou located?",
            "What is the capital of Burkina Faso?",
            "Write a short poem about Askia Mohammed.",
            "Explain the concept of recursion in programming with an example.",
            "What are the key differences between machine learning and deep learning?",
        };
    }

    bool loadModelPathsFromJson(const std::string &json_file)
    {
        try {
            std::ifstream file(json_file);
            if (!file.is_open())
                throw std::runtime_error("Could not open JSON file: " + json_file);
            
            json data = json::parse(file);
            quantized_models.clear();
            
            for (const auto &[key, val] : data.items()) {
                QuantizedModelInfo info;
                info.modelId = val.value("modelID", "");
                info.fileName = val.value("fileName", "");
                info.lastModified = val.value("lastModified", "");
                info.quantization = val.value("quantization", "unknown");
                info.fileSize = val.value("fileSize", 0);
                
                if (info.isValid()) {
                    quantized_models.push_back(info);
                }
            }
            
            std::cout << "Loaded " << quantized_models.size() << " models from JSON." << std::endl;
            return true;
        }
        catch (const std::exception &e) {
            std::cerr << "Error loading models: " << e.what() << std::endl;
            return false;
        }
    }

    BenchmarkResult benchmarkSingleModel(const QuantizedModelInfo &model_info, const BenchmarkParams &params)
    {
        BenchmarkResult result(model_info.modelId);
        auto total_start = high_resolution_clock::now();

        try {
            LlamaInterface interface;

            LlamaModelParams model_params;
            model_params.model_path = ataraxia_path.empty() ? 
                model_info.fileName : ataraxia_path + "/" + model_info.fileName;
            model_params.n_gpu_layers = params.n_gpu_layers;

            auto load_start = high_resolution_clock::now();
            if (!interface.load_model(model_params)) {
                throw std::runtime_error("Failed to load model via LlamaInterface");
            }
            result.metrics.loadTime = duration_cast<milliseconds>(
                high_resolution_clock::now() - load_start).count();

            GenerationParams gen_params;
            gen_params.n_predict = params.n_gen;
            gen_params.temp = params.temperature;
            gen_params.top_k = params.top_k;
            gen_params.top_p = params.top_p;

            if (params.warmup) {
                std::cout << "  Running warmup..." << std::endl;
                interface.generate_completion("Hello", gen_params);
            }

            for (int i = 0; i < params.repetitions; ++i) {
                const std::string &prompt = benchmark_prompts[i % benchmark_prompts.size()];
                if (i == 0) result.promptUsed = prompt;

                auto generation_start = high_resolution_clock::now();
                std::string generated = interface.generate_completion(prompt, gen_params);
                auto generation_time = duration_cast<milliseconds>(
                    high_resolution_clock::now() - generation_start).count();

                result.metrics.generationTimes.push_back(generation_time);
                
                int tokensGenerated = generated.length() / 4; // Rough approximation
                double tokensPerSec = (generation_time > 0) ? 
                    (tokensGenerated * 1000.0) / generation_time : 0.0;
                result.metrics.tokensPerSecondHistory.push_back(tokensPerSec);

                if (i == 0) {
                    result.generatedText = generated;
                    result.metrics.tokensGenerated = tokensGenerated;
                }
            }

            result.calculateStatistics();
            result.metrics.success = true;
        }
        catch (const std::exception &e) {
            result.metrics.success = false;
            result.metrics.errorMessage = e.what();
        }

        result.metrics.totalTime = duration_cast<milliseconds>(
            high_resolution_clock::now() - total_start).count();
        return result;
    }

    std::vector<BenchmarkResult> benchmarkAllModels(const BenchmarkParams &params)
    {
        std::vector<BenchmarkResult> results;
        std::cout << "Benchmarking " << quantized_models.size() << " models..." << std::endl;
        std::cout << "Parameters: GPU Layers=" << params.n_gpu_layers 
                  << ", Repetitions=" << params.repetitions << std::endl;
        std::cout << std::string(80, '=') << std::endl;

        if (params.parallel && quantized_models.size() > 1) {
            // Parallel execution
            std::vector<std::future<BenchmarkResult>> futures;
            for (const auto &model_info : quantized_models) {
                futures.push_back(std::async(std::launch::async, 
                    [this, model_info, params]() {
                        return benchmarkSingleModel(model_info, params);
                    }));
            }

            for (auto &future : futures) {
                BenchmarkResult result = future.get();
                results.push_back(result);
                printBenchmarkResult(result);
                std::cout << std::string(80, '-') << std::endl;
            }
        } else {
            for (const auto &model_info : quantized_models) {
                std::cout << "Benchmarking: " << model_info.modelId << std::endl;
                BenchmarkResult result = benchmarkSingleModel(model_info, params);
                results.push_back(result);
                printBenchmarkResult(result);
                std::cout << std::string(80, '-') << std::endl;
            }
        }

        printSummary(results);
        return results;
    }

    void printBenchmarkResult(const BenchmarkResult &result)
    {
        if (!result.metrics.success) {
            std::cout << "  FAILED: " << result.metrics.errorMessage << std::endl;
            return;
        }

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  • Load time:        " << result.metrics.loadTime << " ms" << std::endl;
        std::cout << "  • Generation time:  " << result.metrics.generationTime << " ms";
        
        if (result.metrics.generationTimes.size() > 1) {
            double stddev = stdev(result.metrics.generationTimes);
            std::cout << " (±" << stddev << " ms)";
        }
        std::cout << std::endl;
        
        std::cout << "  • Speed (approx):   " << result.metrics.tokensPerSecond << " tokens/sec";
        if (result.metrics.tokensPerSecondHistory.size() > 1) {
            double stddev = stdev(result.metrics.tokensPerSecondHistory);
            std::cout << " (±" << stddev << ")";
        }
        std::cout << std::endl;
        
        std::cout << "  • Tokens generated: " << result.metrics.tokensGenerated << std::endl;
    }

    void printSummary(const std::vector<BenchmarkResult> &results)
    {
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "BENCHMARK SUMMARY" << std::endl;
        std::cout << std::string(80, '=') << std::endl;

        auto fastest = std::max_element(results.begin(), results.end(),
            [](const BenchmarkResult &a, const BenchmarkResult &b) {
                return a.metrics.success && b.metrics.success ? 
                    a.metrics.tokensPerSecond < b.metrics.tokensPerSecond : !a.metrics.success;
            });

        if (fastest != results.end() && fastest->metrics.success) {
            std::cout << "Fastest model: " << fastest->modelId 
                      << " (" << fastest->metrics.tokensPerSecond << " tokens/sec)" << std::endl;
        }

        int successful = std::count_if(results.begin(), results.end(),
            [](const BenchmarkResult &r) { return r.metrics.success; });
        std::cout << "Success rate: " << successful << "/" << results.size() 
                  << " (" << (100.0 * successful / results.size()) << "%)" << std::endl;
    }

    void exportResults(const std::vector<BenchmarkResult> &results, const std::string &filename)
    {
        json output;
        output["benchmark_timestamp"] = std::time(nullptr);
        output["benchmark_params"] = {
            {"n_gpu_layers", default_params.n_gpu_layers},
            {"n_gen", default_params.n_gen},
            {"repetitions", default_params.repetitions},
            {"temperature", default_params.temperature},
            {"top_k", default_params.top_k},
            {"top_p", default_params.top_p}
        };
        output["results"] = json::array();

        for (const auto &result : results) {
            json result_json;
            result_json["model_id"] = result.modelId;
            result_json["success"] = result.metrics.success;
            result_json["prompt_used"] = result.promptUsed;
            
            if (!result.metrics.success) {
                result_json["error_message"] = result.metrics.errorMessage;
            } else {
                result_json["metrics"] = {
                    {"load_time_ms", result.metrics.loadTime},
                    {"generation_time_ms", result.metrics.generationTime},
                    {"total_time_ms", result.metrics.totalTime},
                    {"tokens_generated", result.metrics.tokensGenerated},
                    {"tokens_per_second", result.metrics.tokensPerSecond}
                };

                if (!result.metrics.generationTimes.empty()) {
                    result_json["metrics"]["generation_times_ms"] = result.metrics.generationTimes;
                    result_json["metrics"]["generation_time_stddev"] = stdev(result.metrics.generationTimes);
                }
                if (!result.metrics.tokensPerSecondHistory.empty()) {
                    result_json["metrics"]["tokens_per_second_history"] = result.metrics.tokensPerSecondHistory;
                    result_json["metrics"]["tokens_per_second_stddev"] = stdev(result.metrics.tokensPerSecondHistory);
                }
            }

            output["results"].push_back(result_json);
        }

        std::ofstream file(filename);
        file << output.dump(4);
        std::cout << "Results exported to: " << filename << std::endl;
    }

    void setBenchmarkParams(const BenchmarkParams &params) {
        default_params = params;
    }

    void setBenchmarkPrompts(const std::vector<std::string> &prompts) {
        benchmark_prompts = prompts;
    }

    BenchmarkParams getDefaultParams() const {
        return default_params;
    }

    void addModel(const QuantizedModelInfo &model) {
        quantized_models.push_back(model);
    }

    void clearModels() {
        quantized_models.clear();
    }

    size_t getModelCount() const {
        return quantized_models.size();
    }

    std::vector<std::string> getModelIds() const {
        std::vector<std::string> ids;
        for (const auto &model : quantized_models) {
            ids.push_back(model.modelId);
        }
        return ids;
    }
};