
#include <string>
#include <memory>
#include <iostream>
#include "core_ai/core_ai_service.hh"

#define MINIAUDIO_IMPLEMENTATION
#include "core_ai/miniaudio.h"


/**
 * @brief Constructs a CoreAIService object.
 *
 * Initializes the CoreAIService instance by setting the internal pointers
 * for the Llama and Whisper interfaces to nullptr and marking both
 * the Llama and Whisper models as not loaded.
 */
CoreAIService::CoreAIService() : llama_interface_(nullptr), whisper_interface_(nullptr),
                                 llama_model_loaded_(false), whisper_model_loaded_(false)
{
}

/**
 * @brief Constructs a CoreAIService object with specified model parameters.
 *
 * Initializes the CoreAIService with the provided Llama and Whisper model parameters.
 *
 * @param llama_model_params_ The parameters for configuring the Llama model.
 * @param whisper_model_params_ The parameters for configuring the Whisper model.
 */
CoreAIService::CoreAIService(const LlamaModelParams &llama_model_params_, const WhisperModelParams &whisper_model_params_)
{
    llama_model_params = llama_model_params_;
    whisper_model_params = whisper_model_params_;
}

/**
 * @brief Destructor for the CoreAIService class.
 *
 * This destructor ensures that all resources associated with the CoreAIService
 * are properly released. Specifically, it unloads the Llama and Whisper models
 * by calling the respective unload functions. This helps prevent memory leaks
 * and ensures clean shutdown of the service.
 */
CoreAIService::~CoreAIService()
{
    unload_llama_model();
    unload_whisper_model();
}

/**
 * @brief Initializes the Llama model with the specified parameters.
 *
 * This function checks if a Llama model is already loaded. If so, it unloads the existing model
 * before creating a new instance of LlamaInterface. It then attempts to load the model using
 * the provided parameters.
 *
 * @param llama_model_params The parameters required to load the Llama model.
 * @return true if the model was loaded successfully; false otherwise.
 */
bool CoreAIService::initialize_llama_model(const LlamaModelParams &llama_model_params)
{
    if (llama_interface_)
    {
        unload_llama_model(); 
    }
    llama_interface_ = std::make_unique<LlamaInterface>();
    llama_model_loaded_ = llama_interface_->load_model(llama_model_params);
    return llama_model_loaded_;
}

/**
 * @brief Initializes the Whisper model with the specified parameters.
 *
 * This function checks if a Whisper model is already loaded. If so, it unloads the existing model
 * before creating a new WhisperInterface instance. It then attempts to load the Whisper model
 * using the provided parameters.
 *
 * @param whisper_model_params_ The parameters required to load the Whisper model.
 * @return true if the model was loaded successfully; false otherwise.
 */
bool CoreAIService::initialize_whisper_model(const WhisperModelParams &whisper_model_params_)
{
    if (whisper_interface_)
    {
        unload_whisper_model();
    }
    whisper_interface_ = std::make_unique<WhisperInterface>();
    return whisper_interface_->load_model(whisper_model_params_);
}

/**
 * @brief Unloads the currently loaded Llama model and releases associated resources.
 *
 * This function checks if the Llama interface is initialized. If so, it calls
 * the unload_model() method to properly unload the model and then resets the
 * interface pointer to release any held resources.
 */
void CoreAIService::unload_llama_model()
{
    if (llama_interface_)
    {
        llama_interface_->unload_model();
        llama_interface_.reset();
    }
}

/**
 * @brief Checks if the Llama model is currently loaded.
 *
 * This method returns true if the Llama interface exists and the model has been successfully loaded,
 * otherwise returns false.
 *
 * @return true if the Llama model is loaded, false otherwise.
 */
bool CoreAIService::is_llama_model_loaded() const
{
    if (llama_interface_)
    {
        return llama_model_loaded_;
    }
    return false;
}

/**
 * @brief Processes the given prompt text using the loaded Llama model.
 *
 * This method checks if a Llama model is loaded. If so, it generates a completion
 * for the provided prompt text using the specified generation parameters. If the
 * model is not loaded, it returns an error message.
 *
 * @param prompt_text The input prompt to be processed by the Llama model.
 * @param llama_generation_params_ The parameters to control the generation behavior.
 * @return std::string The generated completion from the Llama model, or an error message if the model is not loaded.
 */
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

/**
 * @brief Streams a prompt to the loaded Llama model and processes generated tokens via a callback.
 *
 * This function checks if a Llama model is loaded. If so, it streams the given prompt text to the model
 * using the specified generation parameters. Generated tokens are delivered incrementally through the provided
 * callback function. If no model is loaded, an error message is sent to the callback (if provided), and the function returns false.
 *
 * @param prompt_text The input prompt to be processed by the Llama model.
 * @param llama_generation_params Parameters controlling the generation behavior of the Llama model.
 * @param callback A function to be called with each generated token or error message.
 * @return true if streaming was successfully started; false if no model is loaded or an error occurred.
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
 * @brief Checks if the Whisper model is loaded.
 *
 * This method returns true if the Whisper interface exists and the model has been successfully loaded.
 * Otherwise, it returns false.
 *
 * @return true if the Whisper model is loaded, false otherwise.
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
 * @brief Unloads the currently loaded Whisper model and releases associated resources.
 *
 * This method checks if a Whisper interface instance exists. If so, it calls
 * the unload_model() method on the interface to properly unload the model,
 * and then resets the interface pointer to release any held resources.
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
 * @brief Transcribes audio data from PCM float32 format using the loaded Whisper model.
 *
 * This function takes a vector of 32-bit floating point PCM audio samples and transcribes
 * them into text using the Whisper model, if it is loaded. If the model is not loaded,
 * an error message is returned.
 *
 * @param pcm_f32_data A vector containing the PCM audio data as float values.
 * @param whisper_model_params_ Parameters for configuring the Whisper transcription process.
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
 * @brief Converts an audio file to 32-bit floating point PCM data (mono, 16kHz).
 *
 * This function takes the path to an audio file, decodes it using the miniaudio library,
 * and converts it to a vector of 32-bit floating point PCM samples with a target format
 * of mono channel and 16kHz sample rate.
 *
 * @param audio_file_path The path to the audio file to be converted.
 * @return std::vector<float> A vector containing the decoded PCM samples in f32 format.
 *         Returns an empty vector if the file cannot be decoded or an error occurs.
 *
 * @note The function prints error messages to std::cerr if decoding fails at any stage.
 *       On success, a message is printed to std::cout.
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
 * This function loads an audio file from the given path, converts it to 32-bit floating point PCM data,
 * and then performs transcription using the provided Whisper model parameters.
 *
 * @param audio_file_path The path to the audio file to be transcribed.
 * @param whisper_model_params_ The parameters to configure the Whisper transcription model.
 * @return The transcribed text if successful, or an error message if the audio file could not be loaded.
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
 * This function is responsible for setting up and initializing any global backend
 * dependencies needed for the AI service to function correctly. Currently, it
 * initializes the Llama backend by invoking LlamaInterface::init_backend().
 * 
 * Call this method before using any backend-dependent features of CoreAIService.
 */
void CoreAIService::initialize_global_backends()
{
    LlamaInterface::init_backend();
}

/**
 * @brief Frees resources associated with global AI backends.
 *
 * This function releases any resources or memory allocated by the global backend
 * interfaces used by the CoreAIService. It should be called during shutdown or
 * cleanup to ensure proper resource management.
 */
void CoreAIService::free_global_backends()
{
    LlamaInterface::free_backend();
}