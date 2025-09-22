#include "core_ai_service.hh"

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

/**
 * @brief Constructs a CoreAIService object and initializes member variables.
 *
 * Initializes the llama_interface_ and whisper_interface_ pointers to nullptr,
 * and sets the model loaded flags (llama_model_loaded_ and whisper_model_loaded_)
 * to false, indicating that no models are loaded at construction.
 */
CoreAIService::CoreAIService()
    : llama_interface_(nullptr),
      whisper_interface_(nullptr),
      llama_model_loaded_(false),
      whisper_model_loaded_(false)
{
}

/**
 * @brief Constructs a CoreAIService object with the given Llama and Whisper interfaces.
 *
 * Initializes the CoreAIService by taking ownership of the provided LlamaInterface and
 * WhisperInterface instances. Also initializes the model loaded flags to false.
 *
 * @param llama_interface Unique pointer to a LlamaInterface implementation.
 * @param whisper_interface Unique pointer to a WhisperInterface implementation.
 */
CoreAIService::CoreAIService(
    std::unique_ptr<LlamaInterface> llama_interface,
    std::unique_ptr<WhisperInterface> whisper_interface)
    : llama_interface_(std::move(llama_interface)),
      whisper_interface_(std::move(whisper_interface)),
      llama_model_loaded_(false),
      whisper_model_loaded_(false)
{
}

/**
 * @brief Destructor for the CoreAIService class.
 *
 * This destructor ensures that all resources associated with the CoreAIService
 * instance are properly released. Specifically, it unloads the Llama and Whisper
 * models to free up memory and other resources before the object is destroyed.
 */
CoreAIService::~CoreAIService()
{
    unload_llama_model();
    unload_whisper_model();
}

/**
 * @brief Initializes the Llama model with the specified parameters.
 *
 * This function ensures that the Llama interface is instantiated and attempts to load
 * the Llama model using the provided parameters. The result of the model loading operation
 * is stored internally and returned to indicate success or failure.
 *
 * @param params The parameters required to load the Llama model.
 * @return true if the model was loaded successfully; false otherwise.
 */
bool CoreAIService::initialize_llama_model(const LlamaModelParams &params)
{
    if (!llama_interface_)
    {
        llama_interface_ = std::make_unique<LlamaInterface>();
    }
    llama_model_loaded_ = llama_interface_->load_model(params);
    return llama_model_loaded_;
}

/**
 * @brief Initializes the Whisper model with the specified parameters.
 *
 * This function ensures that the Whisper interface is instantiated and attempts to load
 * the Whisper model using the provided parameters. The result of the model loading
 * operation is stored and returned.
 *
 * @param params The parameters required to load the Whisper model.
 * @return true if the Whisper model was successfully loaded; false otherwise.
 */
bool CoreAIService::initialize_whisper_model(const WhisperModelParams &params)
{
    if (!whisper_interface_)
    {
        whisper_interface_ = std::make_unique<WhisperInterface>();
    }
    whisper_model_loaded_ = whisper_interface_->load_model(params);
    return whisper_model_loaded_;
}

/**
 * @brief Unloads the currently loaded Llama model from the service.
 *
 * This function checks if the Llama interface is initialized. If so, it calls
 * the interface's unload_model() method to release the model resources and
 * updates the internal state to indicate that no model is loaded.
 */
void CoreAIService::unload_llama_model()
{
    if (llama_interface_)
    {
        llama_interface_->unload_model();
        llama_model_loaded_ = false;
    }
}

/**
 * @brief Checks if the Llama model is currently loaded.
 *
 * This function returns true if both the Llama interface is initialized
 * and the Llama model has been successfully loaded into memory.
 *
 * @return true if the Llama model is loaded and ready for use, false otherwise.
 */
bool CoreAIService::is_llama_model_loaded() const
{
    return llama_interface_ && llama_model_loaded_;
}

/**
 * @brief Processes the given prompt text using the loaded Llama model and returns the generated completion.
 *
 * This function checks if a Llama model is loaded. If so, it generates a completion for the provided prompt text
 * using the specified generation parameters. If the model is not loaded, it returns an error message.
 *
 * @param prompt_text The input prompt text to be processed by the Llama model.
 * @param llama_generation_params_ The parameters to control the generation behavior of the Llama model.
 * @return std::string The generated completion from the Llama model, or an error message if the model is not loaded.
 */
std::string CoreAIService::process_prompt(const std::string &prompt_text, const GenerationParams &llama_generation_params_)
{
    if (is_llama_model_loaded())
    {
        double ttft_ms = 0.0;
        double decode_duration_ms = 0.0;
        int32_t tokens_generated = 0;
        return llama_interface_->generate_completion(prompt_text, llama_generation_params_, ttft_ms, decode_duration_ms, tokens_generated);
    }
    else
    {
        return "[Error: Llama model not loaded]";
    }
}

std::vector<int32_t> CoreAIService::tokenization(const std::string &text)
{
    if (is_llama_model_loaded())
    {
        return llama_interface_->tokenization(text);
    }
    else
    {
        return {};
    }
}

std::string CoreAIService::detokenization(const std::vector<int32_t> &tokens) const
{
    if (is_llama_model_loaded())
    {
        return llama_interface_->detokenization(tokens);
    }
    else
    {
        return "[Error: Llama model not loaded]";
    }
}

/**
 * @brief Streams a prompt to the Llama model and returns generated completions via a callback.
 *
 * This function checks if the Llama model is loaded. If so, it streams the prompt text to the model
 * using the specified generation parameters, invoking the provided callback with each generated token.
 * If the model is not loaded, the callback is invoked with an error message and the function returns false.
 *
 * @param prompt_text The input prompt to be sent to the Llama model.
 * @param llama_generation_params Parameters controlling the generation behavior of the model.
 * @param callback A function to be called with each generated token or error message.
 * @return true if the streaming started successfully, false otherwise.
 */
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

/**
 * @brief Checks if the Whisper model is currently loaded.
 *
 * This method returns true if the Whisper interface exists and the model has been successfully loaded.
 * Otherwise, it returns false.
 *
 * @return true if the Whisper model is loaded; false otherwise.
 */
bool CoreAIService::is_whisper_model_loaded() const
{
    if (whisper_interface_)
    {
        return whisper_model_loaded_;
    }
    return false;
}

/**
 * @brief Unloads the currently loaded Whisper model, if any.
 *
 * This function checks if a Whisper model interface exists. If so, it calls
 * the unload_model() method to release any resources associated with the model,
 * and then resets the interface pointer to ensure proper cleanup.
 */
void CoreAIService::unload_whisper_model()
{
    if (whisper_interface_)
    {
        whisper_interface_->unload_model();
        whisper_interface_.reset();
    }
}

/**
 * @brief Transcribes audio data in PCM float32 format using the loaded Whisper model.
 *
 * This function takes a vector of 32-bit floating point PCM audio samples and transcribes
 * them into text using the Whisper model, if it is loaded. If the model is not loaded,
 * an error message is returned.
 *
 * @param pcm_f32_data A vector containing the PCM audio data as float values.
 * @param whisper_model_params_ Parameters to configure the Whisper model's transcription behavior.
 * @return std::string The transcribed text if successful, or an error message if the model is not loaded.
 */
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

/**
 * @brief Converts an audio file to 16kHz mono 32-bit floating point PCM data.
 *
 * This function uses the miniaudio library to decode the specified audio file,
 * resampling and converting it to a single channel (mono) with a sample rate of 16,000 Hz
 * and 32-bit floating point samples. The resulting PCM data is returned as a std::vector<float>.
 *
 * @param audio_file_path The path to the audio file to be converted.
 * @return std::vector<float> The decoded PCM data in 16kHz mono F32 format.
 *         Returns an empty vector if decoding fails or the file has zero length.
 *
 * @note Supported audio formats depend on the miniaudio build configuration.
 * @note Error messages are printed to std::cerr on failure.
 */
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

/**
 * @brief Transcribes an audio file to text using the specified Whisper model parameters.
 *
 * This function loads an audio file from the given file path, converts it to
 * 32-bit floating point PCM data, and then transcribes the audio using the
 * provided Whisper model parameters.
 *
 * @param audio_file_path The path to the audio file to be transcribed.
 * @param whisper_model_params_ The parameters to configure the Whisper model for transcription.
 * @return The transcribed text as a std::string. Returns an error message if the audio file could not be loaded.
 */
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

/**
 * @brief Initializes all global AI backends required by the CoreAIService.
 *
 * This function is responsible for setting up and initializing any global
 * backend dependencies needed for the AI service to function correctly.
 * Currently, it initializes the Llama backend interface.
 *
 * Should be called during the startup phase before using any backend-dependent features.
 */
void CoreAIService::initialize_global_backends()
{
    LlamaInterface::init_backend();
}

/**
 * @brief Frees resources allocated by global backend instances.
 *
 * This function calls the static method LlamaInterface::free_backend()
 * to release any resources or memory held by the global backend.
 * It should be invoked during shutdown or cleanup to prevent resource leaks.
 */
void CoreAIService::free_global_backends()
{
    LlamaInterface::free_backend();
}