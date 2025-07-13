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
#include "core_ai/json.hpp"

using namespace std::chrono;
using json = nlohmann::json;

/**
 * @brief Computes the average value of the elements in a vector.
 *
 * This function calculates the arithmetic mean of the elements in the given vector.
 * If the vector is empty, it returns a default-constructed value of type T (typically zero).
 *
 * @tparam T The type of the elements in the vector. Must support addition, division, and default construction.
 * @param v The vector of elements to average.
 * @return The average value of the elements in the vector, or T(0) if the vector is empty.
 */
template <typename T>
T avg(const std::vector<T>& v) {
    if (v.empty()) return T(0);
    return std::accumulate(v.begin(), v.end(), T(0)) / v.size();
}

/**
 * @brief Calculates the sample standard deviation of a vector of values.
 *
 * This function computes the standard deviation using the sample formula (dividing by N-1),
 * where N is the number of elements in the input vector. If the vector contains one or zero elements,
 * the function returns zero.
 *
 * @tparam T Numeric type of the vector elements.
 * @param v The input vector containing values to compute the standard deviation for.
 * @return The sample standard deviation of the input vector.
 */
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

    bool operator==(const QuantizedModelInfo &other) const {
        return modelId == other.modelId && fileName == other.fileName &&
               lastModified == other.lastModified && quantization == other.quantization &&
               fileSize == other.fileSize;
    }

    bool operator!=(const QuantizedModelInfo &other) const {
        return !(*this == other);
    }

    size_t hash() const {
        return std::hash<std::string>()(modelId) ^ std::hash<std::string>()(fileName) ^
               std::hash<std::string>()(lastModified) ^ std::hash<std::string>()(quantization) ^
               std::hash<size_t>()(fileSize);
    }

    std::string to_string() const {
        return "QuantizedModelInfo(modelId=" + modelId + ", fileName=" + fileName +
               ", lastModified=" + lastModified + ", quantization=" + quantization +
               ", fileSize=" + std::to_string(fileSize) + ")";
    }
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

    bool operator==(const BenchmarkMetrics &other) const {
        return loadTime == other.loadTime && generationTime == other.generationTime &&
               totalTime == other.totalTime && tokensGenerated == other.tokensGenerated &&
               tokensPerSecond == other.tokensPerSecond && memoryUsage == other.memoryUsage &&
               success == other.success && errorMessage == other.errorMessage &&
               generationTimes == other.generationTimes && tokensPerSecondHistory == other.tokensPerSecondHistory;
    }

    bool operator!=(const BenchmarkMetrics &other) const {
        return !(*this == other);
    }

    size_t hash() const {
        return std::hash<double>()(loadTime) ^ std::hash<double>()(generationTime) ^
               std::hash<double>()(totalTime) ^ std::hash<int>()(tokensGenerated) ^
               std::hash<double>()(tokensPerSecond) ^ std::hash<double>()(memoryUsage) ^
               std::hash<bool>()(success) ^ std::hash<std::string>()(errorMessage);
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2);
        oss << "BenchmarkMetrics(loadTime=" << loadTime << "ms, generationTime=" << generationTime
            << "ms, totalTime=" << totalTime << "ms, tokensGenerated=" << tokensGenerated
            << ", tokensPerSecond=" << tokensPerSecond << ", memoryUsage=" << memoryUsage
            << "MB, success=" << std::boolalpha << success << ", errorMessage='" << errorMessage
            << "', generationTimes.size()=" << generationTimes.size()
            << ", tokensPerSecondHistory.size()=" << tokensPerSecondHistory.size() << ")";
        return oss.str();
    }
};

struct BenchmarkResult
{
    std::string modelId;
    BenchmarkMetrics metrics;
    std::string generatedText;
    std::string promptUsed;

    BenchmarkResult(const std::string &id) : modelId(id) {}
    
    /**
     * @brief Calculates and updates statistical metrics for model benchmarking.
     *
     * This method computes the average generation time and average tokens per second
     * from their respective historical data vectors, if they are not empty, and updates
     * the corresponding metric fields.
     */
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
    /**
     * @brief Constructs a LlamaBenchmarker object.
     *
     * Initializes the default prompts for benchmarking. Attempts to retrieve the
     * ATARAXIA_PATH environment variable to set the base path for model files.
     * If the environment variable is not set, a warning is printed and model paths
     * must be provided as absolute paths.
     */
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

    /**
     * @brief Constructs a LlamaBenchmarker and loads model paths from a JSON file.
     *
     * This constructor initializes the LlamaBenchmarker by first calling the default constructor,
     * then loading model paths from the specified JSON file.
     *
     * @param json_file The path to the JSON file containing model paths.
     */
    explicit LlamaBenchmarker(const std::string &json_file) : LlamaBenchmarker()
    {
        loadModelPathsFromJson(json_file);
    }

    /**
     * @brief Initializes the benchmark_prompts vector with a set of default prompts.
     *
     * This method populates the benchmark_prompts container with a predefined list of
     * questions and tasks. These prompts are intended to be used for benchmarking or
     * evaluating the performance and capabilities of AI models.
     */
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

    /**
     * @brief Loads quantized model information from a JSON file.
     *
     * This function reads the specified JSON file, parses its contents, and populates
     * the `quantized_models` container with valid `QuantizedModelInfo` objects extracted
     * from the file. Each entry in the JSON is expected to contain fields such as
     * "modelID", "fileName", "lastModified", "quantization", and "fileSize".
     * Only valid models (as determined by `QuantizedModelInfo::isValid()`) are added.
     *
     * @param json_file The path to the JSON file containing model information.
     * @return true if the models were loaded successfully; false otherwise.
     */
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

    /**
     * @brief Benchmarks a single quantized model using specified parameters.
     *
     * This function loads a quantized model, optionally performs a warmup generation,
     * and then runs multiple prompt generations to measure performance metrics such as
     * load time, generation time, and tokens per second. The results, including statistics
     * and any errors encountered, are returned in a BenchmarkResult object.
     *
     * @param model_info Information about the quantized model to benchmark, including model ID and file name.
     * @param params Benchmarking parameters, such as number of GPU layers, number of generations,
     *        temperature, top_k, top_p, number of repetitions, and whether to perform a warmup.
     * @return BenchmarkResult containing performance metrics, generated text, and success status.
     */
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

    /**
     * @brief Benchmarks all quantized models using the specified parameters.
     *
     * This function iterates over all quantized models and benchmarks each one using the provided
     * BenchmarkParams. If parallel execution is enabled and there is more than one model, the
     * benchmarking is performed concurrently using std::async. Otherwise, benchmarking is done
     * sequentially. The results for each model are printed, and a summary of all results is displayed
     * at the end.
     *
     * @param params The parameters to use for benchmarking, including GPU layers, repetitions, and
     *               whether to run in parallel.
     * @return std::vector<BenchmarkResult> A vector containing the benchmarking results for all models.
     */
    std::vector<BenchmarkResult> benchmarkAllModels(const BenchmarkParams &params)
    {
        std::vector<BenchmarkResult> results;
        std::cout << "Benchmarking " << quantized_models.size() << " models..." << std::endl;
        std::cout << "Parameters: GPU Layers=" << params.n_gpu_layers 
                  << ", Repetitions=" << params.repetitions << std::endl;
        std::cout << std::string(80, '=') << std::endl;

        if (params.parallel && quantized_models.size() > 1) {
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

    /**
     * @brief Prints the benchmark results for a model execution.
     *
     * This function outputs detailed benchmark metrics to the standard output,
     * including load time, generation time (with optional standard deviation),
     * approximate token generation speed (with optional standard deviation),
     * and the total number of tokens generated. If the benchmark failed,
     * it prints the error message instead.
     *
     * @param result The BenchmarkResult structure containing all relevant metrics and status.
     */
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

    /**
     * @brief Prints a summary of benchmark results to the standard output.
     *
     * This function outputs a formatted summary including:
     * - A header section.
     * - The fastest model among the provided benchmark results (based on tokens per second, considering only successful runs).
     * - The overall success rate of the benchmarks.
     *
     * @param results A vector of BenchmarkResult objects containing the results to summarize.
     */
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

    /**
     * @brief Exports benchmark results to a JSON file.
     *
     * This function serializes the provided benchmark results and associated parameters
     * into a JSON structure and writes it to the specified file. The output includes
     * a timestamp, benchmark parameters, and detailed metrics for each result.
     *
     * @param results   A vector of BenchmarkResult objects containing the results to export.
     * @param filename  The path to the file where the JSON output will be written.
     */
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

    /**
     * @brief Sets the default benchmark parameters.
     *
     * This method updates the internal benchmark parameters with the provided values.
     *
     * @param params The BenchmarkParams object containing the new benchmark settings.
     */
    void setBenchmarkParams(const BenchmarkParams &params) {
        default_params = params;
    }

    /**
     * @brief Sets the benchmark prompts to be used for model evaluation.
     *
     * This function assigns the provided list of prompts to the internal
     * benchmark prompts storage. These prompts are used during benchmarking
     * to evaluate the model's performance.
     *
     * @param prompts A vector of strings containing the prompts for benchmarking.
     */
    void setBenchmarkPrompts(const std::vector<std::string> &prompts) {
        benchmark_prompts = prompts;
    }

    /**
     * @brief Retrieves the default benchmark parameters.
     *
     * This function returns the default set of parameters used for benchmarking models.
     *
     * @return BenchmarkParams The default benchmark parameters.
     */
    BenchmarkParams getDefaultParams() const {
        return default_params;
    }

    /**
     * @brief Adds a quantized model to the list of benchmarked models.
     *
     * This function appends the provided QuantizedModelInfo object to the internal
     * collection of quantized models for benchmarking purposes.
     *
     * @param model The QuantizedModelInfo instance representing the model to add.
     */
    void addModel(const QuantizedModelInfo &model) {
        quantized_models.push_back(model);
    }

    /**
     * @brief Removes all quantized models from the collection.
     *
     * This function clears the internal container holding the quantized models,
     * effectively removing all currently stored models and resetting the collection to an empty state.
     */
    void clearModels() {
        quantized_models.clear();
    }

    /**
     * @brief Returns the number of quantized models.
     *
     * @return The total count of quantized models currently stored.
     */
    size_t getModelCount() const {
        return quantized_models.size();
    }

    /**
     * @brief Retrieves the list of model IDs from the quantized models.
     *
     * Iterates through the collection of quantized models and extracts their unique
     * identifiers, returning them as a vector of strings.
     *
     * @return std::vector<std::string> A vector containing the IDs of all quantized models.
     */
    std::vector<std::string> getModelIds() const {
        std::vector<std::string> ids;
        for (const auto &model : quantized_models) {
            ids.push_back(model.modelId);
        }
        return ids;
    }
};