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

// #ifndef NO_PYBIND
// #include <pybind11/pybind11.h>
// using namespace pybind11;

// GenerationParams GenerationParams::from_dict(const py::dict &d) {
//     GenerationParams p;
//     if (d.contains("n_predict")) p.n_predict = d["n_predict"].cast<int32_t>();
//     if (d.contains("temp")) p.temp = d["temp"].cast<float>();
//     if (d.contains("top_k")) p.top_k = d["top_k"].cast<int32_t>();
//     if (d.contains("top_p")) p.top_p = d["top_p"].cast<float>();
//     if (d.contains("repeat_penalty")) p.repeat_penalty = d["repeat_penalty"].cast<float>();
//     if (d.contains("stop_sequences")) p.stop_sequences = d["stop_sequences"].cast<std::vector<std::string>>();
//     if (d.contains("n_batch")) p.n_batch = d["n_batch"].cast<int32_t>();
//     if (d.contains("n_threads")) p.n_threads = d["n_threads"].cast<int32_t>();
//     return p;
// }
// #endif

void LlamaInterface::init_backend()
{
    std::call_once(backend_init_flag, []()
                   {
        ggml_backend_load_all();
        llama_backend_init();
        llama_log_set(nullptr, nullptr);
        backend_initialized.store(true);
        std::cerr << "LlamaInterface: Backend initialized." << std::endl; });
}

void LlamaInterface::free_backend()
{
    if (backend_initialized.load())
    {
        llama_backend_free();
        backend_initialized.store(false);
        std::cerr << "LlamaInterface: Backend freed." << std::endl;
    }
}

LlamaInterface::LlamaInterface() : model_(nullptr), ctx_(nullptr), vocab_(nullptr)
{
    init_backend();
}

LlamaInterface::~LlamaInterface()
{
    unload_model();
}

LlamaInterface::LlamaInterface(LlamaInterface &&other) noexcept
    : model_(other.model_), ctx_(other.ctx_), vocab_(other.vocab_),
      current_model_params_(std::move(other.current_model_params_))
{
    other.model_ = nullptr;
    other.ctx_ = nullptr;
    other.vocab_ = nullptr;
}

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

bool LlamaInterface::load_model(const LlamaModelParams &params)
{
    if (model_)
    {
        unload_model();
    }

    // Validate parameters
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

    // Load model with error checking
    llama_model_params model_p = llama_model_default_params();
    model_p.n_gpu_layers = current_model_params_.n_gpu_layers;

    // Add validation for GPU layers
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

    // Create context with better defaults
    llama_context_params ctx_p = llama_context_default_params();
    ctx_p.n_ctx = current_model_params_.n_ctx;
    ctx_p.n_batch = std::min(512, current_model_params_.n_ctx / 4); // Adaptive batch size
    ctx_p.offload_kqv = true;
    ctx_p.n_threads = std::max(1u, std::thread::hardware_concurrency() / 2);   // Use half CPU cores
    ctx_p.n_threads_batch = std::max(1u, std::thread::hardware_concurrency()); // All cores for batch

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

bool LlamaInterface::is_model_loaded() const
{
    return model_ != nullptr && ctx_ != nullptr && vocab_ != nullptr;
}

std::vector<int32_t> LlamaInterface::tokenize(const std::string &text, bool add_bos, bool special) const
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
        // Buffer too small, resize and retry
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

std::string LlamaInterface::detokenize_token(int32_t token) const
{
    if (!is_model_loaded())
    {
        return "";
    }

    constexpr size_t buf_size = 256; // Increased buffer size for UTF-8 sequences
    char buf[buf_size];
    int n = llama_token_to_piece(vocab_, token, buf, buf_size, 0, true);

    if (n < 0)
    {
        std::cerr << "LlamaInterface Error: failed to convert token " << token << " to piece" << std::endl;
        return "[Error]";
    }

    return std::string(buf, n);
}

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

llama_sampler *LlamaInterface::create_sampler(const GenerationParams &params)
{
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;

    llama_sampler *smpl = llama_sampler_chain_init(sparams);
    if (!smpl)
    {
        throw std::runtime_error("Failed to initialize sampler chain");
    }

    // TODO : implement sampler configuration based on params

    return smpl;
}

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

    std::string completion_text;

    return completion_text;
}

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

int LlamaInterface::get_context_size() const
{
    if (!is_model_loaded())
        return 0;
    return current_model_params_.n_ctx;
}

int LlamaInterface::get_vocab_size() const
{
    if (!is_model_loaded())
        return 0;
    return llama_vocab_n_tokens(vocab_);
}

std::string LlamaInterface::get_model_info() const
{
    if (!is_model_loaded())
        return "No model loaded";

    return "Model: " + current_model_params_.model_path +
           ", Context: " + std::to_string(current_model_params_.n_ctx) +
           ", GPU Layers: " + std::to_string(current_model_params_.n_gpu_layers) +
           ", Vocab Size: " + std::to_string(get_vocab_size());
}