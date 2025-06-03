#pragma once

#include <string>
#include <vector>
#include <functional>
#include <llama.h>

struct llama_model;
struct llama_context;
struct llama_context_params;

using llama_token_callback = std::function<bool(const std::string &token_text)>;

struct LlamaModelParams
{
    std::string model_path;
    int32_t n_ctx = 2048;
    int32_t n_gpu_layers = 0;
    int32_t main_gpu = 0;
    bool tensor_split = false;
    bool vocab_only = false;
    bool use_map = false;
    bool use_mlock = false;

    LlamaModelParams() = default;

    LlamaModelParams(const std::string &path, int32_t ctx = 2048, int32_t gpu_layers = 0,
                     int32_t main_gpu = 0, bool split = false, bool only = false,
                     bool map = false, bool mlock = false)
        : model_path(path), n_ctx(ctx), n_gpu_layers(gpu_layers), main_gpu(main_gpu),
          tensor_split(split), vocab_only(only), use_map(map), use_mlock(mlock) {}

    LlamaModelParams &set_model_path(const std::string &path)
    {
        model_path = path;
        return *this;
    }

    LlamaModelParams &set_n_ctx(int32_t ctx)
    {
        n_ctx = ctx;
        return *this;
    }

    LlamaModelParams &set_n_gpu_layers(int32_t gpu_layers)
    {
        n_gpu_layers = gpu_layers;
        return *this;
    }

    LlamaModelParams &set_main_gpu(int32_t gpu)
    {
        main_gpu = gpu;
        return *this;
    }

    LlamaModelParams &set_tensor_split(bool split)
    {
        tensor_split = split;
        return *this;
    }

    LlamaModelParams &set_vocab_only(bool only)
    {
        vocab_only = only;
        return *this;
    }

    LlamaModelParams &set_use_map(bool map)
    {
        use_map = map;
        return *this;
    }

    LlamaModelParams &set_use_mlock(bool mlock)
    {
        use_mlock = mlock;
        return *this;
    }

    bool operator==(const LlamaModelParams &other) const
    {
        return model_path == other.model_path &&
               n_ctx == other.n_ctx &&
               n_gpu_layers == other.n_gpu_layers &&
               main_gpu == other.main_gpu &&
               tensor_split == other.tensor_split &&
               vocab_only == other.vocab_only &&
               use_map == other.use_map &&
               use_mlock == other.use_mlock;
    }
    bool operator!=(const LlamaModelParams &other) const
    {
        return !(*this == other);
    }
    std::size_t hash() const
    {
        return std::hash<std::string>()(model_path) ^
               std::hash<int32_t>()(n_ctx) ^
               std::hash<int32_t>()(n_gpu_layers) ^
               std::hash<int32_t>()(main_gpu) ^
               std::hash<bool>()(tensor_split) ^
               std::hash<bool>()(vocab_only) ^
               std::hash<bool>()(use_map) ^
               std::hash<bool>()(use_mlock);
    }
    std::string to_string() const
    {
        return "LlamaModelParams(model_path='" + model_path +
               "', n_ctx=" + std::to_string(n_ctx) +
               ", n_gpu_layers=" + std::to_string(n_gpu_layers) +
               ", main_gpu=" + std::to_string(main_gpu) +
               ", tensor_split=" + (tensor_split ? "true" : "false") +
               ", vocab_only=" + (vocab_only ? "true" : "false") +
               ", use_map=" + (use_map ? "true" : "false") +
               ", use_mlock=" + (use_mlock ? "true" : "false") + ")";
    }
};

// #ifndef NO_PYBIND
// namespace pybind11
// {
//     class dict;
// }
// #endif

struct GenerationParams
{
    int32_t n_predict = 128;
    float temp = 0.8f;
    int32_t top_k = 40;
    float top_p = 0.95f;
    float repeat_penalty = 1.1f;
    std::vector<std::string> stop_sequences;
    int32_t n_batch = 512;
    int32_t n_threads = 0;

    GenerationParams() = default;

    GenerationParams(int32_t predict, float temperature = 0.8f, int32_t top_k = 40,
                     float top_p = 0.95f, float repeat_penalty = 1.1f,
                     std::vector<std::string> stop_seqs = {}, int32_t batch_size = 512,
                     int32_t threads = 0)
        : n_predict(predict), temp(temperature), top_k(top_k), top_p(top_p),
          repeat_penalty(repeat_penalty), stop_sequences(std::move(stop_seqs)),
          n_batch(batch_size), n_threads(threads) {}

    bool operator==(const GenerationParams &other) const
    {
        return n_predict == other.n_predict &&
               temp == other.temp &&
               top_k == other.top_k &&
               top_p == other.top_p &&
               repeat_penalty == other.repeat_penalty &&
               stop_sequences == other.stop_sequences &&
               n_batch == other.n_batch &&
               n_threads == other.n_threads;
    }
    bool operator!=(const GenerationParams &other) const
    {
        return !(*this == other);
    }

    std::size_t hash() const
    {
        std::size_t h = std::hash<int32_t>()(n_predict) ^
                        std::hash<float>()(temp) ^
                        std::hash<int32_t>()(top_k) ^
                        std::hash<float>()(top_p) ^
                        std::hash<float>()(repeat_penalty) ^
                        std::hash<int32_t>()(n_batch) ^
                        std::hash<int32_t>()(n_threads);
        for (const auto &s : stop_sequences)
            h ^= std::hash<std::string>()(s);
        return h;
    }

    std::string to_string() const
    {
        return "GenerationParams(n_predict=" + std::to_string(n_predict) +
               ", temp=" + std::to_string(temp) +
               ", top_k=" + std::to_string(top_k) +
               ", top_p=" + std::to_string(top_p) +
               ", repeat_penalty=" + std::to_string(repeat_penalty) +
               ", stop_sequences=[" + (stop_sequences.empty() ? "" : stop_sequences[0]) +
               (stop_sequences.size() > 1 ? ", ..." : "") + "]" +
               ", n_batch=" + std::to_string(n_batch) +
               ", n_threads=" + std::to_string(n_threads) + ")";
    }

// #ifndef NO_PYBIND
//     static GenerationParams from_dict(const pybind11::dict &d);
// #endif

    GenerationParams &set_n_predict(int32_t predict)
    {
        n_predict = predict;
        return *this;
    }
    GenerationParams &set_temp(float temperature)
    {
        temp = temperature;
        return *this;
    }
    GenerationParams &set_top_k(int32_t k)
    {
        top_k = k;
        return *this;
    }

    GenerationParams &set_top_p(float p)
    {
        top_p = p;
        return *this;
    }

    GenerationParams &set_repeat_penalty(float penalty)
    {
        repeat_penalty = penalty;
        return *this;
    }

    GenerationParams &set_stop_sequences(const std::vector<std::string> &sequences)
    {
        stop_sequences = sequences;
        return *this;
    }

    GenerationParams &set_n_batch(int32_t batch_size)
    {
        n_batch = batch_size;
        return *this;
    }

    GenerationParams &set_n_threads(int32_t threads)
    {
        n_threads = threads;
        return *this;
    }
};

class LlamaInterface
{
public:
    LlamaInterface();
    LlamaInterface(LlamaInterface &&other) noexcept;
    LlamaInterface &operator=(LlamaInterface &&other) noexcept;
    ~LlamaInterface();

    bool load_model(const LlamaModelParams &params);
    void unload_model();
    bool is_model_loaded() const;
    llama_sampler *create_sampler(const GenerationParams &params);
    bool check_stop_sequences(const std::string &text, const std::vector<std::string> &stop_sequences);
    int get_context_size() const;
    int get_vocab_size() const;
    std::string get_model_info() const;
    // std::vector<int32_t> tokenize(const std::string &text, bool add_bos = true, bool special) const;
    std::string generate_completion(const std::string &prompt_text, const GenerationParams &params);
    bool generate_completion_streaming(const std::string &prompt_text,
                                       const GenerationParams &params,
                                       llama_token_callback callback);
    std::vector<float> get_embeddings(const std::string &text);

    static void init_backend();
    static void free_backend();

private:
    llama_model *model_ = nullptr;
    llama_context *ctx_ = nullptr;
    const llama_vocab *vocab_ = nullptr;

    LlamaModelParams current_model_params_;

    std::vector<int32_t> tokenize(const std::string &text, bool add_bos, bool special) const;
    std::string detokenize_token(int32_t token) const;
    std::string detokenize_sequence(const std::vector<int32_t> &tokens) const;

    llama_sampler *create_default_sampler(const GenerationParams &params);
};
