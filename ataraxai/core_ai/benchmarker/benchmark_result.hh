

/* struct BenchmarkResult
{
    std::string model_name;
    float load_time_ms;
    float pps;
    float token_gen_time_ms;
    float time_to_first_token_ms;
    float sample_time_ms;
    float peak_ram_bytes;
    long ram_peak_kb;
    float perplexity;
    float latency_ms;
    float throughput;
    float ttft_ms;
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

BenchmarkResult benchmark_model(const std::string &model_path, llama_model_params &model_params, const llama_context_params &cparams, const std::string &prompt_text, int tokens_to_generate)
{
    cpu_info_collection cpu_info_list = cpu_info_collection();
    gpu_info_collection gpu_info_list = gpu_info_collection();

    BenchmarkResult result;
    result.model_name = std::filesystem::path(model_path).filename().string();

    long ram_before_load = get_current_memory_usage();

    // --- Load Model ---

    // load dynamic backends

    ggml_backend_load_all();

    // Initialize the model
    auto t_load_start = std::chrono::high_resolution_clock::now();
    llama_model *model = llama_load_model_from_file(model_path.c_str(), model_params);
    if (!model)
    {
        std::cerr << "Failed to load model: " << model_path << std::endl;
        return result;
    }
    auto t_load_end = std::chrono::high_resolution_clock::now();


    result.load_time_ms = std::chrono::duration<double, std::milli>(t_load_end - t_load_start).count();

    llama_context *ctx = llama_new_context_with_model(model, cparams);
    if (!ctx)
    {
        std::cerr << "Failed to create context for: " << model_path << std::endl;
        llama_free_model(model);
        return result;
    }

    long ram_after_load = get_current_memory_usage();
    result.peak_ram_bytes = std::max(result.peak_ram_bytes, float(ram_after_load - ram_before_load));

    // --- Tokenize Prompt ---
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> prompt_tokens;
    prompt_tokens.resize(cparams.n_ctx);       
    // find the number of tokens in the prompt 
                                                                                                         // Max possible size
    int n_prompt_tokens = llama_tokenize(vocab, prompt_text.c_str(), prompt_text.length(), prompt_tokens.data(), prompt_tokens.size(), true, false); // add_bos = true, special = false
    if (n_prompt_tokens < 1)
    {
        std::cerr << "Failed to tokenize prompt or prompt too long." << std::endl;
        llama_free(ctx);
        llama_free_model(model);
        return result;
    }
    prompt_tokens.resize(n_prompt_tokens);

    // prepare a batch for the prompt

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    // --- Prompt Processing (PPS & TTFT) ---
    auto t_prompt_eval_start = std::chrono::high_resolution_clock::now();
    if (llama_decode(ctx, batch) != 0)
    { // n_past = 0
        std::cerr << "Failed to eval prompt for " << model_path << std::endl;
        return result;
    }
    // The first token is now ready to be sampled
    auto t_first_token_ready = std::chrono::high_resolution_clock::now();
    result.ttft_ms = std::chrono::duration<double, std::milli>(t_first_token_ready - t_prompt_eval_start).count(); // More accurately, time from start of eval to first sample
    result.pps = (n_prompt_tokens > 0 && result.ttft_ms > 0) ? (n_prompt_tokens / (result.ttft_ms / 1000.0)) : 0.0;

    // --- Token Generation (TPS) ---
    std::vector<llama_token> generated_tokens;
    auto t_gen_start = std::chrono::high_resolution_clock::now();
    int current_n_past = n_prompt_tokens;


    // --- Cleanup ---
    llama_free(ctx);
    llama_free_model(model);

    return result;
} */