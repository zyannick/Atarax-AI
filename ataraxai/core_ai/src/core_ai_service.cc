#include "core_ai/core_ai_service.hh"

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#define MINIAUDIO_IMPLEMENTATION
#include "core_ai/miniaudio.h"

CoreAIService::CoreAIService()
    : llama_interface_(nullptr),
      whisper_interface_(nullptr),
      llama_model_loaded_(false),
      whisper_model_loaded_(false)
{
}

CoreAIService::CoreAIService(
    std::unique_ptr<LlamaInterface> llama_interface,
    std::unique_ptr<WhisperInterface> whisper_interface)
    : llama_interface_(std::move(llama_interface)),
      whisper_interface_(std::move(whisper_interface)),
      llama_model_loaded_(false),
      whisper_model_loaded_(false)
{
}

CoreAIService::~CoreAIService()
{
    unload_llama_model();
    unload_whisper_model();
}

bool CoreAIService::initialize_llama_model(const LlamaModelParams &params)
{
    if (!llama_interface_)
    {
        llama_interface_ = std::make_unique<LlamaInterface>();
    }
    llama_model_loaded_ = llama_interface_->load_model(params);
    return llama_model_loaded_;
}

bool CoreAIService::initialize_whisper_model(const WhisperModelParams &params)
{
    if (!whisper_interface_)
    {
        whisper_interface_ = std::make_unique<WhisperInterface>();
    }
    whisper_model_loaded_ = whisper_interface_->load_model(params);
    return whisper_model_loaded_;
}

void CoreAIService::unload_llama_model()
{
    if (llama_interface_)
    {
        llama_interface_->unload_model();
        llama_model_loaded_ = false;
    }
}

bool CoreAIService::is_llama_model_loaded() const
{
    return llama_interface_ && llama_model_loaded_;
}

std::string CoreAIService::process_prompt(const std::string &prompt_text, const GenerationParams &llama_generation_params_)
{
    if (is_llama_model_loaded())
    {
        return llama_interface_->generate_completion(prompt_text, llama_generation_params_);
    }
    else
    {
        return "[Error: Llama model not loaded]";
    }
}

bool CoreAIService::stream_prompt(const std::string &prompt_text,
                                  const GenerationParams &llama_generation_params,
                                  llama_token_callback callback)
{
    if (is_llama_model_loaded())
    {
        return llama_interface_->generate_completion_streaming(prompt_text, llama_generation_params, callback);
    }
    else
    {
        if (callback)
        {
            callback("[Error: Llama model not loaded]");
        }
        return false;
    }
}

bool CoreAIService::is_whisper_model_loaded() const
{
    if (whisper_interface_)
    {
        return whisper_model_loaded_;
    }
    return false;
}

void CoreAIService::unload_whisper_model()
{
    if (whisper_interface_)
    {
        whisper_interface_->unload_model();
        whisper_interface_.reset();
    }
}

std::string CoreAIService::transcribe_audio_pcm(const std::vector<float> &pcm_f32_data, const WhisperGenerationParams &whisper_model_params_)
{
    if (is_whisper_model_loaded())
    {
        return whisper_interface_->transcribe_pcm(pcm_f32_data, whisper_model_params_);
    }
    else
    {
        return "[Error: Whisper model not loaded]";
    }
}

std::vector<float> CoreAIService::convert_audio_file_to_pcm_f32(const std::string &audio_file_path)
{
    ma_result result;
    ma_decoder_config config = ma_decoder_config_init(ma_format_f32, 1, 16000); // Target: f32, 1 channel, 16kHz
    ma_decoder decoder;

    result = ma_decoder_init_file(audio_file_path.c_str(), &config, &decoder);
    if (result != MA_SUCCESS)
    {
        std::cerr << "Failed to initialize audio decoder for: " << audio_file_path << " Error: " << ma_result_description(result) << std::endl;
        return {};
    }

    ma_uint64 total_frames;
    result = ma_decoder_get_length_in_pcm_frames(&decoder, &total_frames);
    if (result != MA_SUCCESS)
    {
        std::cerr << "Failed to get audio length for: " << audio_file_path << " Error: " << ma_result_description(result) << std::endl;
        ma_decoder_uninit(&decoder);
        return {};
    }

    if (total_frames == 0)
    {
        std::cerr << "Audio file has zero length: " << audio_file_path << std::endl;
        ma_decoder_uninit(&decoder);
        return {};
    }

    std::vector<float> pcm_data(total_frames);
    ma_uint64 frames_read;

    result = ma_decoder_read_pcm_frames(&decoder, pcm_data.data(), total_frames, &frames_read);
    if (result != MA_SUCCESS || frames_read != total_frames)
    {
        std::cerr << "Failed to read PCM frames for: " << audio_file_path << " Read " << frames_read << "/" << total_frames << ". Error: " << ma_result_description(result) << std::endl;
        ma_decoder_uninit(&decoder);
        return {};
    }

    ma_decoder_uninit(&decoder);
    std::cout << "Successfully decoded and converted " << audio_file_path << " to 16kHz mono F32 PCM." << std::endl;
    return pcm_data;
}

std::string CoreAIService::transcribe_audio_file(const std::string &audio_file_path, const WhisperGenerationParams &whisper_model_params_)
{
    std::vector<float> pcm_f32_data;
    pcm_f32_data = convert_audio_file_to_pcm_f32(audio_file_path);
    if (pcm_f32_data.empty())
    {
        return "[Error: Failed to load audio file]";
    }
    return transcribe_audio_pcm(pcm_f32_data, whisper_model_params_);
}

void CoreAIService::initialize_global_backends()
{
    LlamaInterface::init_backend();
}

void CoreAIService::free_global_backends()
{
    LlamaInterface::free_backend();
}