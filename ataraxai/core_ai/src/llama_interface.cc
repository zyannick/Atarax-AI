#include <atomic>
#include "core_ai/llama_interface.hh"
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <memory>
#include <mutex>
#include <thread>

static std::once_flag backend_init_flag;
static std::atomic<bool> backend_initialized{false};

/**
 * @brief Initializes the backend for the LlamaInterface.
 *
 * This function ensures that the backend initialization routines are executed only once,
 * even if called from multiple threads. It loads all required GGML backends, initializes
 * the Llama backend, and sets the logging function to null. Once initialization is complete,
 * it sets the backend_initialized flag to true and logs a message to standard error.
 *
 * Thread-safe: Uses std::call_once to guarantee single initialization.
 */
void LlamaInterface::init_backend()
{
    std::call_once(backend_init_flag, []()
                   {
        ggml_backend_load_all();
        llama_backend_init();
        // only print errors
        llama_log_set([](enum ggml_log_level level, const char * text, void * /* user_data */) {
            if (level >= GGML_LOG_LEVEL_ERROR) {
                fprintf(stderr, "%s", text);
            }
        }, nullptr);
        backend_initialized.store(true);
        std::cerr << "LlamaInterface: Backend initialized." << std::endl; });
}

/**
 * @brief Frees the resources associated with the Llama backend if it has been initialized.
 *
 * This function checks whether the backend has been initialized. If so, it releases
 * the backend resources by calling `llama_backend_free()`, updates the initialization
 * status, and logs the action to standard error output.
 */
void LlamaInterface::free_backend()
{
    if (backend_initialized.load())
    {
        llama_backend_free();
        backend_initialized.store(false);
        std::cerr << "LlamaInterface: Backend freed." << std::endl;
    }
}

/**
 * @brief Constructs a LlamaInterface object and initializes its members.
 *
 * This constructor initializes the internal pointers (model_, ctx_, vocab_) to nullptr
 * and calls the init_backend() function to set up the backend environment required
 * for the LlamaInterface to operate.
 */
LlamaInterface::LlamaInterface() : model_(nullptr), ctx_(nullptr), vocab_(nullptr)
{
    init_backend();
}

/**
 * @brief Destructor for the LlamaInterface class.
 *
 * This destructor ensures that any resources or models loaded by the interface
 * are properly released by calling the unload_model() method.
 */
LlamaInterface::~LlamaInterface()
{
    unload_model();
}

/**
 * @brief Move constructor for LlamaInterface.
 *
 * Transfers ownership of the internal model, context, and vocabulary pointers,
 * as well as the current model parameters, from another LlamaInterface instance.
 * After the move, the source instance's pointers are set to nullptr to prevent
 * double deletion.
 *
 * @param other The LlamaInterface instance to move from.
 */
LlamaInterface::LlamaInterface(LlamaInterface &&other) noexcept
    : model_(other.model_), ctx_(other.ctx_), vocab_(other.vocab_),
      current_model_params_(std::move(other.current_model_params_))
{
    other.model_ = nullptr;
    other.ctx_ = nullptr;
    other.vocab_ = nullptr;
}

/**
 * @brief Move assignment operator for LlamaInterface.
 *
 * Transfers ownership of the internal resources from another LlamaInterface instance
 * to this instance. If this instance already holds resources, they are released first.
 * After the move, the source instance is left in a valid but unspecified state.
 *
 * @param other The LlamaInterface instance to move from.
 * @return Reference to this LlamaInterface instance.
 */
LlamaInterface &LlamaInterface::operator=(LlamaInterface &&other) noexcept
{
    if (this != &other)
    {
        unload_model();
        model_ = other.model_;
        ctx_ = other.ctx_;
        vocab_ = other.vocab_;
        current_model_params_ = std::move(other.current_model_params_);
        other.model_ = nullptr;
        other.ctx_ = nullptr;
        other.vocab_ = nullptr;
    }
    return *this;
}

/**
 * @brief Loads a Llama model with the specified parameters.
 *
 * This function attempts to load a Llama model from the file path specified in the given
 * LlamaModelParams. It first unloads any currently loaded model. It validates the input
 * parameters, sets up model and context parameters, and initializes the model and context.
 * If any step fails, it logs an error message and returns false.
 *
 * @param params The parameters for loading the model, including model path, context size,
 *               and GPU layer configuration.
 * @return true if the model was loaded successfully; false otherwise.
 */
bool LlamaInterface::load_model(const LlamaModelParams &params)
{
    if (model_)
    {
        unload_model();
    }

    if (params.model_path.empty())
    {
        std::cerr << "LlamaInterface Error: model path is empty" << std::endl;
        return false;
    }

    if (params.n_ctx <= 0 || params.n_ctx > 32768)
    {
        std::cerr << "LlamaInterface Error: invalid context size: " << params.n_ctx << std::endl;
        return false;
    }

    current_model_params_ = params;

    llama_model_params model_p = llama_model_default_params();
    model_p.n_gpu_layers = current_model_params_.n_gpu_layers;

    if (model_p.n_gpu_layers < 0)
    {
        std::cerr << "LlamaInterface Warning: negative GPU layers, setting to 0" << std::endl;
        model_p.n_gpu_layers = 0;
    }

    model_ = llama_model_load_from_file(current_model_params_.model_path.c_str(), model_p);
    if (!model_)
    {
        std::cerr << "LlamaInterface Error: unable to load model from " << current_model_params_.model_path << std::endl;
        return false;
    }

    vocab_ = llama_model_get_vocab(model_);
    if (!vocab_)
    {
        std::cerr << "LlamaInterface Error: failed to get vocabulary" << std::endl;
        llama_model_free(model_);
        model_ = nullptr;
        return false;
    }

    llama_context_params ctx_p = llama_context_default_params();
    ctx_p.n_ctx = current_model_params_.n_ctx;
    ctx_p.n_batch = std::min(512, current_model_params_.n_ctx / 4); 
    ctx_p.offload_kqv = true;
    ctx_p.n_threads = std::max(1u, std::thread::hardware_concurrency() / 2);   
    ctx_p.n_threads_batch = std::max(1u, std::thread::hardware_concurrency()); 

    ctx_ = llama_init_from_model(model_, ctx_p);
    if (!ctx_)
    {
        std::cerr << "LlamaInterface Error: failed to create llama_context" << std::endl;
        llama_model_free(model_);
        model_ = nullptr;
        vocab_ = nullptr;
        return false;
    }

    std::cerr << "LlamaInterface: Model loaded successfully: " << current_model_params_.model_path
              << " (ctx: " << ctx_p.n_ctx << ", gpu_layers: " << model_p.n_gpu_layers << ")" << std::endl;
    return true;
}

/**
 * @brief Unloads the currently loaded Llama model and releases associated resources.
 *
 * This function frees the context and model objects if they are loaded,
 * sets their pointers to nullptr, and resets the vocabulary pointer.
 * It also logs a message indicating that the model has been unloaded.
 */
void LlamaInterface::unload_model()
{
    if (ctx_)
    {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_)
    {
        llama_model_free(model_);
        model_ = nullptr;
    }
    vocab_ = nullptr;
    std::cerr << "LlamaInterface: Model unloaded." << std::endl;
}

/**
 * @brief Checks if the Llama model and its required components are loaded.
 *
 * This function verifies that the model, context, and vocabulary pointers are all non-null,
 * indicating that the model and its dependencies have been successfully loaded and initialized.
 *
 * @return true if the model, context, and vocabulary are loaded; false otherwise.
 */
bool LlamaInterface::is_model_loaded() const
{
    return model_ != nullptr && ctx_ != nullptr && vocab_ != nullptr;
}

/**
 * @brief Tokenizes the given text using the loaded Llama model vocabulary.
 *
 * This function converts the input string into a sequence of llama_token values,
 * optionally adding a beginning-of-sequence (BOS) token and handling special tokens.
 * It first checks if the model is loaded, then estimates the required buffer size
 * for tokenization. If the initial buffer is insufficient, it resizes and retries.
 *
 * @param text The input string to tokenize.
 * @param add_bos If true, prepends a BOS token to the output.
 * @param special If true, enables special token handling during tokenization.
 * @return std::vector<llama_token> The sequence of tokens representing the input text.
 *         Returns an empty vector if the model is not loaded or tokenization fails.
 */
std::vector<llama_token> LlamaInterface::tokenize(const std::string &text, bool add_bos, bool special) const
{
    if (!is_model_loaded())
    {
        std::cerr << "LlamaInterface Error: model not loaded for tokenization" << std::endl;
        return {};
    }

    if (text.empty())
    {
        return add_bos ? std::vector<int32_t>{llama_vocab_bos(vocab_)} : std::vector<int32_t>{};
    }

    // Better size estimation
    int n_tokens_estimated = static_cast<int>(text.length() * 1.5) + (add_bos ? 1 : 0) + 64;
    std::vector<llama_token> result(n_tokens_estimated);

    int n_tokens = llama_tokenize(vocab_, text.c_str(), text.length(), result.data(), result.size(), add_bos, special);

    if (n_tokens < 0)
    {
        int required_size = -n_tokens;
        result.resize(required_size);
        n_tokens = llama_tokenize(vocab_, text.c_str(), text.length(), result.data(), result.size(), add_bos, special);
        if (n_tokens < 0)
        {
            std::cerr << "LlamaInterface Error: tokenization failed even after resize" << std::endl;
            return {};
        }
    }

    result.resize(n_tokens);
    return result;
}

/**
 * @brief Converts a token ID to its corresponding string representation.
 *
 * This function takes an integer token ID and returns the detokenized string
 * using the loaded vocabulary. If the model is not loaded, it returns an empty string.
 * If the conversion fails, it logs an error and returns "[Error]".
 *
 * @param token The token ID to be detokenized.
 * @return The string representation of the token, or "[Error]" if conversion fails,
 *         or an empty string if the model is not loaded.
 */
std::string LlamaInterface::detokenize_token(int32_t token) const
{
    if (!is_model_loaded())
    {
        return "";
    }

    constexpr size_t buf_size = 256; 
    char buf[buf_size];
    int n = llama_token_to_piece(vocab_, token, buf, buf_size, 0, true);

    if (n < 0)
    {
        std::cerr << "LlamaInterface Error: failed to convert token " << token << " to piece" << std::endl;
        return "[Error]";
    }

    return std::string(buf, n);
}

/**
 * @brief Converts a sequence of token IDs into a string by detokenizing each token.
 *
 * This function takes a vector of integer token IDs and reconstructs the original
 * string by detokenizing each token using the detokenize_token method. If the input
 * vector is empty, an empty string is returned.
 *
 * @param tokens A vector of integer token IDs to be detokenized.
 * @return The detokenized string corresponding to the input token sequence.
 */
std::string LlamaInterface::detokenize_sequence(const std::vector<int32_t> &tokens) const
{
    if (tokens.empty())
        return "";

    std::string result;
    result.reserve(tokens.size() * 4);

    for (int32_t token : tokens)
    {
        result += detokenize_token(token);
    }
    return result;
}

/**
 * @brief Creates and configures a llama_sampler instance based on the provided generation parameters.
 *
 * This function initializes a new llama_sampler chain and sequentially adds various sampling strategies
 * to it, such as penalties for repetition, minimum probability threshold, top-k and top-p sampling,
 * temperature scaling, and distribution initialization. The configuration is determined by the values
 * in the provided GenerationParams structure.
 *
 * @param params The generation parameters specifying sampler configuration, including penalties,
 *               top-k, top-p, temperature, and other relevant settings.
 * @return A pointer to the configured llama_sampler instance.
 */
llama_sampler *LlamaInterface::create_sampler(const GenerationParams &params)
{
    llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());

    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
                                      params.penalty_last_n,
                                      params.repeat_penalty,
                                      params.penalty_freq,
                                      params.penalty_present));

    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1)); 

    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(params.top_k));

    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(params.top_p, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(params.temp));

    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    return smpl;
}

/**
 * @brief Generates a text completion based on the provided prompt and generation parameters.
 *
 * This function uses the currently loaded Llama model to generate a completion for the given prompt text.
 * It handles model loading checks, prompt tokenization, context management, and sampling according to the specified
 * generation parameters. The function supports stopping generation based on a maximum number of tokens or custom stop sequences.
 *
 * @param prompt_text The input prompt for which to generate a completion.
 * @param gen_params  The parameters controlling text generation (e.g., number of tokens, stop sequences).
 * @return The generated completion as a string. Returns an error message string if the model is not loaded,
 *         the prompt is empty, the context size is exceeded, or an exception occurs during generation.
 */
std::string LlamaInterface::generate_completion(const std::string &prompt_text, const GenerationParams &gen_params)
{
    if (!is_model_loaded())
    {
        return "[Error: Model not loaded]";
    }

    if (prompt_text.empty())
    {
        return "[Error: Empty prompt]";
    }

    const llama_vocab *vocab = llama_model_get_vocab(model_);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = current_model_params_.n_ctx;
    ctx_params.n_batch = current_model_params_.n_batch;

    try
    {
        LlamaContextWrapper ctx(model_, ctx_params); 
        llama_sampler *sampler = create_sampler(gen_params);
        std::vector<llama_token> prompt_tokens = tokenize(prompt_text, true, false);

        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
        llama_token new_token_id;

        std::string completion_text;

        int current_nb_predict = 0;

        while (true)
        {
            int n_ctx = llama_n_ctx(ctx);
            int n_ctx_used = llama_kv_self_seq_pos_max(ctx, 0);
            if (n_ctx_used + batch.n_tokens > n_ctx)
            {
                printf("\033[0m\n");
                fprintf(stderr, "context size exceeded\n");
                return "[Error: Context size exceeded]";
            }

            if (llama_decode(ctx, batch))
            {
                GGML_ABORT("failed to decode\n");
            }

            new_token_id = llama_sampler_sample(sampler, ctx, -1);

            if (llama_vocab_is_eog(vocab, new_token_id))
            {
                break;
            }

            char buf[256];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0)
            {
                GGML_ABORT("failed to convert token to piece\n");
            }
            std::string piece(buf, n);
            completion_text += piece;

            current_nb_predict++;
            if (current_nb_predict >= gen_params.n_predict)
            {
                printf("\033[0m\n");
                break; 
            }

            bool stopped_by_sequence = false;
            if (gen_params.stop_sequences.size() > 0)
            { 
                for (size_t i = 0; i < gen_params.stop_sequences.size(); ++i)
                {
                    const std::string &stop_seq = gen_params.stop_sequences[i]; 
                    if (!stop_seq.empty() && completion_text.length() >= stop_seq.length())
                    {
                        if (completion_text.rfind(stop_seq) == (completion_text.length() - stop_seq.length()))
                        {
                            completion_text.erase(completion_text.length() - stop_seq.length());
                            stopped_by_sequence = true;
                            break;
                        }
                    }
                }
            }
            if (stopped_by_sequence)
            {
                break;
            }

            batch = llama_batch_get_one(&new_token_id, 1);
        }

        llama_sampler_free(sampler); 
        // llama_free(ctx);

        return completion_text;
    }
    catch (const std::exception &e)
    {
        return "[Error: " + std::string(e.what()) + "]";
    }
}

/**
 * @brief Generates a completion for the given prompt in a streaming fashion.
 *
 * This function takes a prompt string and generation parameters, tokenizes the prompt,
 * and streams the generated completion tokens to the provided callback function.
 * It performs several checks, including whether the model is loaded, whether the callback
 * is valid, and whether tokenization of the prompt succeeds. If any of these checks fail,
 * an error message is sent to the callback or logged, and the function returns false.
 *
 * @param prompt_text The input prompt to generate a completion for.
 * @param gen_params  The parameters controlling the generation process.
 * @param callback    A function to be called with each generated token or error message.
 * @return true if the streaming generation process was successfully started, false otherwise.
 */
bool LlamaInterface::generate_completion_streaming(
    const std::string &prompt_text,
    const GenerationParams &gen_params,
    llama_token_callback callback)
{
    if (!is_model_loaded())
    {
        callback("[Error: Model not loaded]");
        return false;
    }

    if (!callback)
    {
        std::cerr << "LlamaInterface Error: callback is null" << std::endl;
        return false;
    }

    std::vector<llama_token> prompt_tokens = tokenize(prompt_text, true, false);
    if (prompt_tokens.empty() && !prompt_text.empty())
    {
        callback("[Error: Prompt tokenization failed]");
        return false;
    }

    return true;
}

/**
 * @brief Checks if the given text ends with any of the specified stop sequences.
 *
 * Iterates through the provided list of stop sequences and determines if the input
 * text ends with any non-empty stop sequence. Returns true if a match is found,
 * otherwise returns false.
 *
 * @param text The text to check for stop sequences.
 * @param stop_sequences A vector of stop sequences to check against the end of the text.
 * @return true if the text ends with any of the stop sequences, false otherwise.
 */
bool LlamaInterface::check_stop_sequences(const std::string &text, const std::vector<std::string> &stop_sequences)
{
    for (const auto &stop_seq : stop_sequences)
    {
        if (stop_seq.empty())
            continue;

        if (text.length() >= stop_seq.length())
        {
            if (text.compare(text.length() - stop_seq.length(), stop_seq.length(), stop_seq) == 0)
            {
                return true;
            }
        }
    }
    return false;
}

/**
 * @brief Retrieves the context size of the currently loaded model.
 *
 * This function returns the context size (number of tokens) as specified
 * in the parameters of the currently loaded model. If no model is loaded,
 * the function returns 0.
 *
 * @return int The context size of the loaded model, or 0 if no model is loaded.
 */
int LlamaInterface::get_context_size() const
{
    if (!is_model_loaded())
        return 0;
    return current_model_params_.n_ctx;
}

/**
 * @brief Returns the size of the vocabulary for the loaded model.
 *
 * This function checks if a model is currently loaded. If not, it returns 0.
 * Otherwise, it retrieves and returns the number of tokens in the model's vocabulary.
 *
 * @return int The number of tokens in the vocabulary, or 0 if no model is loaded.
 */
int LlamaInterface::get_vocab_size() const
{
    if (!is_model_loaded())
        return 0;
    return llama_vocab_n_tokens(vocab_);
}

/**
 * @brief Retrieves information about the currently loaded model.
 *
 * This function returns a string containing details about the loaded model,
 * including the model path, context size, number of GPU layers, and vocabulary size.
 * If no model is loaded, it returns "No model loaded".
 *
 * @return std::string A string describing the loaded model or indicating that no model is loaded.
 */
std::string LlamaInterface::get_model_info() const
{
    if (!is_model_loaded())
        return "No model loaded";

    return "Model: " + current_model_params_.model_path +
           ", Context: " + std::to_string(current_model_params_.n_ctx) +
           ", GPU Layers: " + std::to_string(current_model_params_.n_gpu_layers) +
           ", Vocab Size: " + std::to_string(get_vocab_size());
}