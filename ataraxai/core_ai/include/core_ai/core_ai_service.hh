#pragma once

#include <string>
#include <vector>
#include <memory>

#include "llama_interface.hh"
#include "whisper_interface.hh"

class CoreAIService
{
public:
    CoreAIService();
    CoreAIService(const LlamaModelParams &llama_model_params_, const WhisperModelParams &whisper_model_params_);
    CoreAIService(std::unique_ptr<LlamaInterface> llama_interface_,
                  std::unique_ptr<WhisperInterface> whisper_interface_);
    ~CoreAIService();

    virtual bool initialize_llama_model(const LlamaModelParams &llama_model_params_);

    virtual void unload_llama_model();

    virtual bool is_llama_model_loaded() const;

    std::string process_prompt(const std::string &prompt_text, const GenerationParams &llama_generation_params_);

    bool stream_prompt(const std::string &prompt_text,
                       const GenerationParams &llama_generation_params,
                       llama_token_callback callback);

    bool initialize_whisper_model(const WhisperModelParams &whisper_model_params_);

    void unload_whisper_model();

    bool is_whisper_model_loaded() const;

    std::string transcribe_audio_pcm(const std::vector<float> &pcm_f32_data,
                                     const WhisperGenerationParams &whisper_transcription_params);

    std::string transcribe_audio_file(const std::string &audio_file_path,
                                      const WhisperGenerationParams &whisper_transcription_params);

    static void initialize_global_backends();

    static void free_global_backends();

    void set_llama_interface(std::unique_ptr<LlamaInterface> llama_interface)
    {
        llama_interface_ = std::move(llama_interface);
    }

    void set_whisper_interface(std::unique_ptr<WhisperInterface> whisper_interface)
    {
        whisper_interface_ = std::move(whisper_interface);
    }

private:
    std::unique_ptr<LlamaInterface> llama_interface_;
    std::unique_ptr<WhisperInterface> whisper_interface_;

    bool llama_model_loaded_ = false;
    bool whisper_model_loaded_ = false;

    LlamaModelParams llama_model_params;
    WhisperModelParams whisper_model_params;

    std::vector<float> convert_audio_file_to_pcm_f32(const std::string &audio_file_path);
};