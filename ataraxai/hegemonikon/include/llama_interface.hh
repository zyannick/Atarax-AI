#pragma once

#include <string>
#include <vector>
#include <functional>
#include <llama.h>
#include <stdexcept>
// #include <model_benchmarker.hh>

struct llama_model;
struct llama_context;
struct llama_context_params;

using llama_token_callback = std::function<bool(const std::string &token_text)>;


struct HegemonikonLlamaModelParams
{
    std::string model_path;
    int32_t n_ctx = 2048;
    int32_t n_gpu_layers = 0;
    int32_t main_gpu = 0;
    int32_t n_batch = 1;
    bool tensor_split = false;
    bool vocab_only = false;
    bool use_map = false;
    bool use_mlock = false;

    HegemonikonLlamaModelParams() = default;

    /**
     * @brief Constructs HegemonikonLlamaModelParams with specified model loading options.
     *
     * @param path         Path to the model file.
     * @param ctx          Context size (number of tokens to keep in context), default is 2048.
     * @param gpu_layers   Number of layers to offload to GPU, default is 0 (CPU only).
     * @param main_gpu     Index of the main GPU to use, default is 0.
     * @param split        Whether to split tensors across multiple GPUs, default is false.
     * @param only         Load only the vocabulary without model weights, default is false.
     * @param map          Use memory-mapped file for model loading, default is false.
     * @param mlock        Lock model memory to prevent swapping, default is false.
     */
    HegemonikonLlamaModelParams(const std::string &path, int32_t ctx = 2048, int32_t gpu_layers = 0,
                     int32_t main_gpu = 0, int32_t n_batch = 1, bool split = false, bool only = false,
                     bool map = false, bool mlock = false)
        : model_path(path), n_ctx(ctx), n_gpu_layers(gpu_layers), main_gpu(main_gpu),
          n_batch(n_batch), tensor_split(split), vocab_only(only), use_map(map), use_mlock(mlock) {}

    /**
     * @brief Sets the file path to the model.
     *
     * This method assigns the specified file path to the model_path member variable.
     *
     * @param path The file system path to the model file.
     * @return Reference to the current HegemonikonLlamaModelParams object to allow method chaining.
     */
    HegemonikonLlamaModelParams &set_model_path(const std::string &path)
    {
        model_path = path;
        return *this;
    }

    /**
     * @brief Sets the context window size for the model.
     *
     * This function sets the number of context tokens (n_ctx) to be used by the model.
     *
     * @param ctx The desired context window size (number of tokens).
     * @return Reference to the current HegemonikonLlamaModelParams object for method chaining.
     */
    HegemonikonLlamaModelParams &set_n_ctx(int32_t ctx)
    {
        n_ctx = ctx;
        return *this;
    }

    /**
     * @brief Sets the number of layers to be offloaded to the GPU.
     *
     * This method configures how many layers of the model should be processed on the GPU,
     * which can help optimize performance depending on the available hardware.
     *
     * @param gpu_layers The number of layers to offload to the GPU.
     * @return Reference to the current HegemonikonLlamaModelParams object for method chaining.
     */
    HegemonikonLlamaModelParams &set_n_gpu_layers(int32_t gpu_layers)
    {
        n_gpu_layers = gpu_layers;
        return *this;
    }

    /**
     * @brief Sets the main GPU to be used by the model.
     *
     * This method assigns the specified GPU index to the main_gpu member variable.
     * It allows for method chaining by returning a reference to the current object.
     *
     * @param gpu The index of the GPU to set as the main GPU.
     * @return Reference to the current HegemonikonLlamaModelParams object for method chaining.
     */
    HegemonikonLlamaModelParams &set_main_gpu(int32_t gpu)
    {
        main_gpu = gpu;
        return *this;
    }

    /**
     * @brief Sets the tensor splitting option for the model parameters.
     *
     * This method enables or disables tensor splitting by setting the internal
     * tensor_split flag. Tensor splitting can be used to distribute tensors
     * across multiple devices or processes, depending on the implementation.
     *
     * @param split Boolean value indicating whether to enable (true) or disable (false) tensor splitting.
     * @return Reference to the current HegemonikonLlamaModelParams object for method chaining.
     */
    HegemonikonLlamaModelParams &set_tensor_split(bool split)
    {
        tensor_split = split;
        return *this;
    }

    /**
     * @brief Sets whether to load only the vocabulary for the model.
     *
     * This method configures the model parameters to load only the vocabulary,
     * without loading the full model weights. Useful for scenarios where only
     * tokenization or vocabulary information is needed.
     *
     * @param only If true, loads only the vocabulary; if false, loads the full model.
     * @return Reference to the current HegemonikonLlamaModelParams object for method chaining.
     */
    HegemonikonLlamaModelParams &set_vocab_only(bool only)
    {
        vocab_only = only;
        return *this;
    }

    /**
     * @brief Sets whether to use memory-mapped file access for the model.
     *
     * @param map If true, enables memory-mapped file access; otherwise, disables it.
     * @return Reference to the current HegemonikonLlamaModelParams object for method chaining.
     */
    HegemonikonLlamaModelParams &set_use_map(bool map)
    {
        use_map = map;
        return *this;
    }

    /**
     * @brief Sets whether to use memory locking (mlock) for the model.
     *
     * This function enables or disables the use of mlock, which can prevent the model's memory
     * from being swapped out to disk. This may improve performance or security in certain scenarios.
     *
     * @param mlock Boolean flag indicating whether to use mlock (true to enable, false to disable).
     * @return Reference to the current HegemonikonLlamaModelParams object for method chaining.
     */
    HegemonikonLlamaModelParams &set_use_mlock(bool mlock)
    {
        use_mlock = mlock;
        return *this;
    }

    /**
     * @brief Equality operator for HegemonikonLlamaModelParams.
     *
     * Compares this instance with another HegemonikonLlamaModelParams object for equality.
     * Returns true if all member variables (model_path, n_ctx, n_gpu_layers, main_gpu,
     * tensor_split, vocab_only, use_map, use_mlock) are equal; otherwise, returns false.
     *
     * @param other The HegemonikonLlamaModelParams object to compare with.
     * @return true if all parameters are equal, false otherwise.
     */
    bool operator==(const HegemonikonLlamaModelParams &other) const
    {
        return model_path == other.model_path &&
               n_ctx == other.n_ctx &&
               n_gpu_layers == other.n_gpu_layers &&
               main_gpu == other.main_gpu &&
               n_batch == other.n_batch &&
               tensor_split == other.tensor_split &&
               vocab_only == other.vocab_only &&
               use_map == other.use_map &&
               use_mlock == other.use_mlock;
    }

    /**
     * @brief Inequality operator for HegemonikonLlamaModelParams.
     *
     * Compares this instance with another HegemonikonLlamaModelParams object for inequality.
     * Returns true if the two objects are not equal, false otherwise.
     *
     * @param other The HegemonikonLlamaModelParams object to compare with.
     * @return true if the objects are not equal, false otherwise.
     */
    bool operator!=(const HegemonikonLlamaModelParams &other) const
    {
        return !(*this == other);
    }

    /**
     * @brief Computes a hash value for the current object.
     *
     * This function combines the hash values of several member variables,
     * including model_path, n_ctx, n_gpu_layers, main_gpu, tensor_split,
     * vocab_only, use_map, and use_mlock, to produce a unique hash for the object.
     *
     * @return std::size_t The computed hash value.
     */
    std::size_t hash() const
    {
        return std::hash<std::string>()(model_path) ^
               std::hash<int32_t>()(n_ctx) ^
               std::hash<int32_t>()(n_gpu_layers) ^
               std::hash<int32_t>()(main_gpu) ^
               std::hash<int32_t>()(n_batch) ^
               std::hash<bool>()(tensor_split) ^
               std::hash<bool>()(vocab_only) ^
               std::hash<bool>()(use_map) ^
               std::hash<bool>()(use_mlock);
    }

    /**
     * @brief Returns a string representation of the HegemonikonLlamaModelParams object.
     *
     * The returned string includes the values of all member variables in a readable format,
     * such as model_path, n_ctx, n_gpu_layers, main_gpu, tensor_split, vocab_only, use_map, and use_mlock.
     *
     * @return std::string A string describing the current state of the HegemonikonLlamaModelParams object.
     */
    std::string to_string() const
    {
        return "HegemonikonLlamaModelParams(model_path='" + model_path +
               "', n_ctx=" + std::to_string(n_ctx) +
               ", n_gpu_layers=" + std::to_string(n_gpu_layers) +
               ", main_gpu=" + std::to_string(main_gpu) +
               ", n_batch=" + std::to_string(n_batch) +
               ", tensor_split=" + (tensor_split ? "true" : "false") +
               ", vocab_only=" + (vocab_only ? "true" : "false") +
               ", use_map=" + (use_map ? "true" : "false") +
               ", use_mlock=" + (use_mlock ? "true" : "false") + ")";
    }
};

struct HegemonikonGenerationParams
{
    int32_t n_predict = 128;
    float temperature = 0.8f;
    int32_t top_k = 40;
    float top_p = 0.95f;
    float repeat_penalty = 1.1f;
    int32_t penalty_last_n = 64;
    float penalty_freq = 0.0f;
    float penalty_present = 0.0f;
    std::vector<std::string> stop_sequences;
    int32_t n_batch = 1024;
    int32_t n_threads = 0;
    bool add_bos = true;
    bool parse_special = false;

    HegemonikonGenerationParams() = default;

    /**
     * @brief Constructs HegemonikonGenerationParams with specified generation settings.
     *
     * @param predict         Number of tokens to predict/generate.
     * @param temperature     Sampling temperature; higher values increase randomness (default: 0.8f).
     * @param top_k           Limits sampling to the top_k most probable tokens (default: 40).
     * @param top_p           Nucleus sampling probability threshold (default: 0.95f).
     * @param repeat_penalty  Penalty for repeated tokens to reduce repetition (default: 1.1f).
     * @param stop_seqs       List of stop sequences to terminate generation early (default: empty).
     * @param batch_size      Number of tokens to process in a batch (default: 512).
     * @param threads         Number of threads to use for generation (default: 0, meaning auto-detect).
     */
    HegemonikonGenerationParams(int32_t predict, float temperature = 0.8f, int32_t top_k = 40,
                     float top_p = 0.95f, float repeat_penalty = 1.1f, int32_t penalty_last_n = 64,
                     float penalty_freq = 0.0f, float penalty_present = 0.0f,
                     std::vector<std::string> stop_seqs = {}, int32_t batch_size = 1024,
                     int32_t threads = 0)
        : n_predict(predict), temperature(temperature), top_k(top_k), top_p(top_p),
          repeat_penalty(repeat_penalty), penalty_last_n(penalty_last_n),
          penalty_freq(penalty_freq), penalty_present(penalty_present),
          stop_sequences(std::move(stop_seqs)), n_batch(batch_size), n_threads(threads) {}

    
          

    /**
     * @brief Equality operator for HegemonikonGenerationParams.
     *
     * Compares this HegemonikonGenerationParams object with another for equality.
     * Returns true if all generation parameters (n_predict, temperature, top_k, top_p,
     * repeat_penalty, stop_sequences, n_batch, n_threads) are equal between the two objects.
     *
     * @param other The HegemonikonGenerationParams object to compare with.
     * @return true if all parameters are equal, false otherwise.
     */
    bool operator==(const HegemonikonGenerationParams &other) const
    {
        return n_predict == other.n_predict &&
               temperature == other.temperature &&
               top_k == other.top_k &&
               top_p == other.top_p &&
               repeat_penalty == other.repeat_penalty &&
               penalty_last_n == other.penalty_last_n &&
               penalty_freq == other.penalty_freq &&
               penalty_present == other.penalty_present &&
               stop_sequences == other.stop_sequences &&
               n_batch == other.n_batch &&
               n_threads == other.n_threads;
    }

    /**
     * @brief Inequality operator for HegemonikonGenerationParams.
     *
     * Compares this HegemonikonGenerationParams object with another for inequality.
     * Returns true if the two objects are not equal, as determined by the equality operator.
     *
     * @param other The HegemonikonGenerationParams object to compare against.
     * @return true if the objects are not equal, false otherwise.
     */
    bool operator!=(const HegemonikonGenerationParams &other) const
    {
        return !(*this == other);
    }

    /**
     * @brief Hash function for HegemonikonGenerationParams.
     *
     * Computes a hash value for the HegemonikonGenerationParams object by combining the hash values
     * of its individual parameters (n_predict, temperature, top_k, top_p, repeat_penalty,
     * stop_sequences, n_batch, n_threads).
     *
     * @return A std::size_t hash value representing the HegemonikonGenerationParams object.
     */
    std::size_t hash() const
    {
        std::size_t h = std::hash<int32_t>()(n_predict) ^
                        std::hash<float>()(temperature) ^
                        std::hash<int32_t>()(top_k) ^
                        std::hash<float>()(top_p) ^
                        std::hash<float>()(repeat_penalty) ^
                        std::hash<int32_t>()(penalty_last_n) ^
                        std::hash<float>()(penalty_freq) ^
                        std::hash<float>()(penalty_present) ^
                        std::hash<int32_t>()(n_batch) ^
                        std::hash<int32_t>()(n_threads);
        for (const auto &s : stop_sequences)
            h ^= std::hash<std::string>()(s);
        return h;
    }

    /**
     * @brief Returns a string representation of the HegemonikonGenerationParams object.
     *
     * This method constructs a human-readable string that describes the current state
     * of the HegemonikonGenerationParams object, including all its parameters such as n_predict,
     * temperature, top_k, top_p, repeat_penalty, stop_sequences, n_batch, and n_threads.
     *
     * @return A string summarizing the generation parameters.
     */
    std::string to_string() const
    {
        return "HegemonikonGenerationParams(n_predict=" + std::to_string(n_predict) +
               ", temperature=" + std::to_string(temperature) +
               ", top_k=" + std::to_string(top_k) +
               ", top_p=" + std::to_string(top_p) +
               ", repeat_penalty=" + std::to_string(repeat_penalty) +
               ", penalty_last_n=" + std::to_string(penalty_last_n) +
               ", penalty_freq=" + std::to_string(penalty_freq) +
               ", penalty_present=" + std::to_string(penalty_present) +
               ", stop_sequences=[" + [&]()
        {
            std::string seqs;
            for (const auto &seq : stop_sequences)
                seqs += "'" + seq + "', ";
            if (!seqs.empty())
                seqs.pop_back(), seqs.pop_back(); // remove trailing comma and space
            return seqs;
        }() + "], n_batch=" +
               std::to_string(n_batch) + ", n_threads=" + std::to_string(n_threads) + ")";
    }

    // #ifndef NO_PYBIND
    //     static HegemonikonGenerationParams from_dict(const pybind11::dict &d);
    // #endif

    /**
     * @brief Sets the number of tokens to predict during generation.
     *
     * @param predict The desired number of tokens to generate.
     * @return Reference to the current HegemonikonGenerationParams object for method chaining.
     */
    HegemonikonGenerationParams &set_n_predict(int32_t predict)
    {
        n_predict = predict;
        return *this;
    }

    /**
     * @brief Sets the temperature parameter for text generation.
     *
     * The temperature controls the randomness of the generated output.
     * Higher values (e.g., 1.0) produce more random results, while lower values (e.g., 0.2) make the output more deterministic.
     *
     * @param temperature The temperature value to set.
     * @return Reference to the current HegemonikonGenerationParams object for method chaining.
     */
    HegemonikonGenerationParams &set_temp(float temperature)
    {
        temperature = temperature;
        return *this;
    }

    /**
     * @brief Sets the top-k sampling parameter for generation.
     *
     * This method sets the value of top_k, which determines the number of highest probability
     * vocabulary tokens to keep for top-k filtering during text generation. A higher value of k
     * allows for more diverse outputs, while a lower value restricts the model to more likely tokens.
     *
     * @param k The number of top tokens to consider during sampling.
     * @return Reference to the current HegemonikonGenerationParams object for method chaining.
     */
    HegemonikonGenerationParams &set_top_k(int32_t k)
    {
        top_k = k;
        return *this;
    }

    /**
     * @brief Sets the nucleus sampling probability (top-p) parameter for text generation.
     *
     * This method updates the top-p value, which controls the cumulative probability
     * threshold for token selection during generation. A lower value results in more
     * focused and deterministic outputs, while a higher value allows for more diverse
     * and creative responses.
     *
     * @param p The new top-p value (typically between 0.0 and 1.0).
     * @return Reference to the current HegemonikonGenerationParams object for method chaining.
     */
    HegemonikonGenerationParams &set_top_p(float p)
    {
        top_p = p;
        return *this;
    }

    /**
     * @brief Sets the repeat penalty parameter for text generation.
     *
     * The repeat penalty is used to penalize repeated tokens during generation,
     * encouraging more diverse output. A higher value increases the penalty for
     * repeating tokens.
     *
     * @param penalty The repeat penalty value to set.
     * @return Reference to the current HegemonikonGenerationParams object for method chaining.
     */
    HegemonikonGenerationParams &set_repeat_penalty(float penalty)
    {
        repeat_penalty = penalty;
        return *this;
    }

    /**
     * @brief Sets the stop sequences for text generation.
     *
     * This method assigns the provided list of stop sequences to the generation parameters.
     * When any of these sequences are encountered during generation, the process will stop.
     *
     * @param sequences A vector of strings representing the stop sequences.
     * @return Reference to the current HegemonikonGenerationParams object for method chaining.
     */
    HegemonikonGenerationParams &set_stop_sequences(const std::vector<std::string> &sequences)
    {
        stop_sequences = sequences;
        return *this;
    }

    /**
     * @brief Sets the batch size for generation.
     *
     * This method updates the n_batch parameter with the specified batch size.
     * It enables method chaining by returning a reference to the current HegemonikonGenerationParams object.
     *
     * @param batch_size The desired batch size for generation.
     * @return Reference to the updated HegemonikonGenerationParams object.
     */
    HegemonikonGenerationParams &set_n_batch(int32_t batch_size)
    {
        n_batch = batch_size;
        return *this;
    }

    /**
     * @brief Sets the number of threads to be used for generation.
     *
     * This method updates the internal thread count parameter, which determines
     * how many threads will be utilized during the generation process.
     *
     * @param threads The desired number of threads.
     * @return Reference to the current HegemonikonGenerationParams object for method chaining.
     */
    HegemonikonGenerationParams &set_n_threads(int32_t threads)
    {
        n_threads = threads;
        return *this;
    }
};

class LlamaContextWrapper
{
private:
    llama_context *ctx_;

public:
    LlamaContextWrapper(llama_model *model, const llama_context_params &params)
        : ctx_(llama_init_from_model(model, params))
    {
        if (!ctx_)
        {
            throw std::runtime_error("Failed to create llama context");
        }
    }

    ~LlamaContextWrapper()
    {
        if (ctx_)
        {
            llama_free(ctx_);
        }
    }

    LlamaContextWrapper(const LlamaContextWrapper &) = delete;
    LlamaContextWrapper &operator=(const LlamaContextWrapper &) = delete;

    LlamaContextWrapper(LlamaContextWrapper &&other) noexcept
        : ctx_(other.ctx_)
    {
        other.ctx_ = nullptr;
    }

    llama_context *get() const { return ctx_; }
    operator llama_context *() const { return ctx_; } // Implicit conversion
};

class LlamaInterface
{
public:
    LlamaInterface();
    LlamaInterface(LlamaInterface &&other) noexcept;
    LlamaInterface &operator=(LlamaInterface &&other) noexcept;
    ~LlamaInterface();

    virtual bool load_model(const HegemonikonLlamaModelParams &params);
    virtual void unload_model();
    bool is_model_loaded() const;
    llama_sampler *create_sampler(const HegemonikonGenerationParams &params);
    bool check_stop_sequences(const std::string &text, const std::vector<std::string> &stop_sequences);
    int get_context_size() const;
    int get_vocab_size() const;
    std::string get_model_info() const;
    std::vector<int32_t> tokenization(const std::string &text) const;
    std::string detokenization(const std::vector<int32_t> &tokens) const;
    virtual std::string generate_completion(const std::string &prompt_text, const HegemonikonGenerationParams &params, double &ttft_ms,
        double &decode_duration_ms,
        int32_t &tokens_generated);
    virtual bool generate_completion_streaming(const std::string &prompt_text,
                                               const HegemonikonGenerationParams &params,
                                               llama_token_callback callback);
    std::vector<float> get_embeddings(const std::string &text);


    static void init_backend();
    static void free_backend();

private:
    llama_model *model_ = nullptr;
    llama_context *ctx_ = nullptr;
    const llama_vocab *vocab_ = nullptr;

    HegemonikonLlamaModelParams current_model_params_;

    static constexpr double FAST_TTFT_MS = 200.0;
    static constexpr double ACCEPTABLE_TTFT_MS = 500.0;
    static constexpr double MIN_TOKENS_PER_SEC = 5.0;
    static constexpr double GOOD_TOKENS_PER_SEC = 15.0;

    std::vector<int32_t> tokenize(const std::string &text, bool add_bos, bool special) const;
    std::string detokenize_token(int32_t token) const;
    std::string detokenize_sequence(const std::vector<int32_t> &tokens) const;

    llama_sampler *create_default_sampler(const HegemonikonGenerationParams &params);
};
