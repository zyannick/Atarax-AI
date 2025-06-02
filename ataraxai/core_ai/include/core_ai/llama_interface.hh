#pragma once

#ifndef LLAMA_INTERFACE_HH
#define LLAMA_INTERFACE_HH

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
};

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

#endif