#pragma once

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

#include "llama_interface.hh"
#include "json.hpp"

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
T avg(const std::vector<T> &v)
{
    if (v.empty())
        return T(0);
    return std::accumulate(v.begin(), v.end(), T(0)) / v.size();
}

template <typename T>
T percentile(std::vector<T> v, double p)
{
    if (v.empty())
        return T(0);
    std::sort(v.begin(), v.end());
    int index = static_cast<int>(p * (v.size() - 1));
    return v[index];
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
T stdev(const std::vector<T> &v)
{
    if (v.size() <= 1)
        return T(0);
    T mean = avg(v);
    T sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), T(0));
    return std::sqrt(sq_sum / (v.size() - 1) - mean * mean * v.size() / (v.size() - 1));
}

struct HegemonikonQuantizedModelInfo
{
    std::string model_id;
    std::string local_path;
    std::string last_modified;
    std::string quantization;
    size_t fileSize = 0;

    bool isValid() const { return !model_id.empty() && !local_path.empty(); }

    bool operator==(const HegemonikonQuantizedModelInfo &other) const
    {
        return model_id == other.model_id && local_path == other.local_path &&
               last_modified == other.last_modified && quantization == other.quantization &&
               fileSize == other.fileSize;
    }

    bool operator!=(const HegemonikonQuantizedModelInfo &other) const
    {
        return !(*this == other);
    }

    size_t hash() const
    {
        return std::hash<std::string>()(model_id) ^ std::hash<std::string>()(local_path) ^
               std::hash<std::string>()(last_modified) ^ std::hash<std::string>()(quantization) ^
               std::hash<size_t>()(fileSize);
    }

    std::string to_string() const
    {
        return "HegemonikonQuantizedModelInfo(model_id=" + model_id + ", file_name=" + local_path +
               ", last_modified=" + last_modified + ", quantization=" + quantization +
               ", fileSize=" + std::to_string(fileSize) + ")";
    }
};
    
struct HegemonikonBenchmarkMetrics
{
    float load_time_ms = 0.0f;
    float generation_time = 0.0f;
    float total_time = 0.0f;
    int tokens_generated = 0;
    float tokens_per_second = 0.0f;
    float memory_usage = 0.0f;
    bool success = false;
    std::string errorMessage;

    std::vector<float> generation_times;
    std::vector<float> tokens_per_second_history;

    float avg_ttft_ms = 0.0f;
    float avg_decode_tps = 0.0f;
    float avg_end_to_end_latency_ms = 0.0f;

    std::vector<float> ttft_history;
    std::vector<float> end_to_end_latency_history;
    std::vector<float> decode_tps_history;

    float p50_latency_ms = 0.0f;
    float p95_latency_ms = 0.0f;
    float p99_latency_ms = 0.0f;
};

struct HegemonikonBenchmarkParams
{
    int n_gpu_layers = 0;
    int repetitions = 10;
    bool warmup = true;
    HegemonikonGenerationParams generation_params;

    HegemonikonBenchmarkParams() = default;
    HegemonikonBenchmarkParams(int gpu_layers, int reps, bool do_warmup, const HegemonikonGenerationParams &gen_params)
        : n_gpu_layers(gpu_layers), repetitions(reps), warmup(do_warmup), generation_params(gen_params) {}
};

struct HegemonikonBenchmarkResult
{
    std::string model_id;
    HegemonikonBenchmarkMetrics metrics;
    // HegemonikonBenchmarkParams benchmark_params;
    // HegemonikonLlamaModelParams llama_model_params;
    std::string generated_text;
    std::string promptUsed;
    std::string errorMessage;

    HegemonikonBenchmarkResult(const std::string &id) : model_id(id) {}

    void calculateStatistics()
    {
        if (!metrics.ttft_history.empty())
        {
            metrics.avg_ttft_ms = avg(metrics.ttft_history);
        }
        if (!metrics.decode_tps_history.empty())
        {
            metrics.avg_decode_tps = avg(metrics.decode_tps_history);
        }
        if (!metrics.end_to_end_latency_history.empty())
        {
            metrics.avg_end_to_end_latency_ms = avg(metrics.end_to_end_latency_history);
            if (metrics.end_to_end_latency_history.size() > 1)
            {
                metrics.p50_latency_ms = percentile(metrics.end_to_end_latency_history, 0.50);
                metrics.p95_latency_ms = percentile(metrics.end_to_end_latency_history, 0.95);
                metrics.p99_latency_ms = percentile(metrics.end_to_end_latency_history, 0.99);
            }
        }
    }
};



class HegemonikonLlamaBenchmarker
{
private:
    std::vector<HegemonikonQuantizedModelInfo> quantized_models;
    std::vector<std::string> benchmark_prompts;

public:
    HegemonikonLlamaBenchmarker()
    {
        initializeDefaultPrompts();
    }

    // HegemonikonLlamaBenchmarker(std::vector<HegemonikonQuantizedModelInfo> models, std::vector<std::string> prompts)
    //     : quantized_models(std::move(models)), benchmark_prompts(std::move(prompts))
    // {
    //     initializeDefaultPrompts();
    // }

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

    HegemonikonBenchmarkResult benchmarkSingleModel(const HegemonikonQuantizedModelInfo &quantized_model_info, const HegemonikonBenchmarkParams &benchmark_params, HegemonikonLlamaModelParams llama_model_params)
    {
        HegemonikonBenchmarkResult result(quantized_model_info.model_id);
        // result.benchmark_params = benchmark_params;
        // result.llama_model_params = llama_model_params;

        try
        {
            LlamaInterface interface;

            llama_model_params.n_gpu_layers = benchmark_params.n_gpu_layers;

            auto load_start = high_resolution_clock::now();
            if (!interface.load_model(llama_model_params))
            {
                throw std::runtime_error("Failed to load model via LlamaInterface");
            }
            result.metrics.load_time_ms = duration_cast<milliseconds>(
                                              high_resolution_clock::now() - load_start)
                                              .count();

            HegemonikonGenerationParams gen_params = benchmark_params.generation_params;

            if (benchmark_params.warmup)
            {
                double ttft_ms = 0.0;
                double decode_duration_ms = 0.0;
                int32_t tokens_generated = 0;
                std::cout << "  Running warmup..." << std::endl;
                interface.generate_completion("Hello", gen_params, ttft_ms, decode_duration_ms, tokens_generated);
            }

            for (int i = 0; i < benchmark_params.repetitions; ++i)
            {
                auto e2e_start = high_resolution_clock::now();
                const std::string &prompt = benchmark_prompts[i % benchmark_prompts.size()];
                if (i == 0)
                    result.promptUsed = prompt;

                double ttft_ms = 0.0;
                double decode_duration_ms = 0.0;
                int32_t tokens_generated = 0;

                std::string generated_text = interface.generate_completion(prompt, gen_params, ttft_ms, decode_duration_ms, tokens_generated);

                auto e2e_latency_ms = duration_cast<microseconds>(
                                          high_resolution_clock::now() - e2e_start)
                                          .count() /
                                      1000.0;

                result.metrics.end_to_end_latency_history.push_back(e2e_latency_ms);
                result.metrics.ttft_history.push_back(ttft_ms);

                double decode_tps = (decode_duration_ms > 0) ? (tokens_generated * 1000.0) / decode_duration_ms : 0.0;
                result.metrics.decode_tps_history.push_back(decode_tps);

                if (i == 0)
                {
                    result.generated_text = generated_text;
                }
            }
            result.metrics.success = true;
        }
        catch (const std::exception &e)
        {
            result.metrics.success = false;
            result.metrics.errorMessage = e.what();
        }

        result.calculateStatistics();
        return result;
    }

    void printBenchmarkResult(const HegemonikonBenchmarkResult &result)
    {
        if (!result.metrics.success)
        {
            std::cout << "  FAILED: " << result.metrics.errorMessage << std::endl;
            return;
        }

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  Load Time:          " << result.metrics.load_time_ms << " ms" << std::endl;
        std::cout << "  Avg TTFT:           " << result.metrics.avg_ttft_ms << " ms" << std::endl;
        std::cout << "  Avg Decode Speed:   " << result.metrics.avg_decode_tps << " tokens/sec" << std::endl;
        std::cout << "  Avg E2E Latency:    " << result.metrics.avg_end_to_end_latency_ms << " ms" << std::endl;
        std::cout << "  Latency (P50/P95/P99): "
                  << result.metrics.p50_latency_ms << " / "
                  << result.metrics.p95_latency_ms << " / "
                  << result.metrics.p99_latency_ms << " ms" << std::endl;
    }

    void printSummary(const std::vector<HegemonikonBenchmarkResult> &results)
    {
        std::cout << "\n"
                  << std::string(80, '=') << std::endl;
        std::cout << "BENCHMARK SUMMARY" << std::endl;
        std::cout << std::string(80, '=') << std::endl;

        auto fastest = std::max_element(results.begin(), results.end(),
                                        [](const HegemonikonBenchmarkResult &a, const HegemonikonBenchmarkResult &b)
                                        {
                                            return a.metrics.success && b.metrics.success ? a.metrics.avg_decode_tps < b.metrics.avg_decode_tps : !a.metrics.success;
                                        });

        if (fastest != results.end() && fastest->metrics.success)
        {
            std::cout << "Fastest model (by decode TPS): " << fastest->model_id
                      << " (" << fastest->metrics.avg_decode_tps << " tokens/sec)" << std::endl;
        }

        int successful = std::count_if(results.begin(), results.end(),
                                       [](const HegemonikonBenchmarkResult &r)
                                       { return r.metrics.success; });
        std::cout << "Success rate: " << successful << "/" << results.size()
                  << " (" << (100.0 * successful / results.size()) << "%)" << std::endl;
    }

    void setBenchmarkPrompts(const std::vector<std::string> &prompts)
    {
        benchmark_prompts = prompts;
    }

    std::vector<std::string> getBenchmarkPrompts() const
    {
        return benchmark_prompts;
    }

    std::vector<HegemonikonQuantizedModelInfo> getQuantizedModels() const
    {
        return quantized_models;
    }

    void setQuantizedModels(const std::vector<HegemonikonQuantizedModelInfo> &models)
    {
        quantized_models = models;
    }
};