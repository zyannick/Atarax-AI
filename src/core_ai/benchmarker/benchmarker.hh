#include <string>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>

using namespace std::chrono;
using namespace std;

#include <thread>
unsigned int num_cores = std::thread::hardware_concurrency();

struct BenchmarkResult
{
    std:string model_name;
    float load_time_ms;
    float pps;
    float token_gen_time_ms;
    float time_to_first_token_ms;
    float sample_time_ms;
    long ram_peak_kb;
    float perplexity;
    float latency_ms;
    float throughput;
};

struct BenchmarkSettings
{
    std::string model_path;
    std::string input_text;
    int n_threads;
    int n_past;
    int n_predict;
    bool use_mlock;
};

BenchmarkResult benchmark_model(const std::string &model_path, const std::string &input_text)
{

}