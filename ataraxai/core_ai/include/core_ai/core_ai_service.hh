#pragma once
#include <string>
#include <memory> 


#ifndef CORE_AI_SERVICE_HH
#define CORE_AI_SERVICE_HH

class CoreAIService {
public:
    CoreAIService();
    ~CoreAIService(); 

    bool initialize(const std::string& model_path, int n_gpu_layers, int n_ctx);
    std::string process_prompt(const std::string& prompt_text, int max_new_tokens);
    bool is_model_loaded() const;

private:
    bool model_loaded_ = false;
};

#endif // CORE_AI_SERVICE_HH