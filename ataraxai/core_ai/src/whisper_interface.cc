#include "core_ai/whisper_interface.hh"
#include "whisper.h"
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <memory>
#include <mutex>
#include <thread>

WhisperInterface::WhisperInterface() : ctx_(nullptr)
{

    std::cerr << "WhisperInterface created. Ensure whisper backend (if any specific) is initialized if needed." << std::endl;
}

WhisperInterface::~WhisperInterface()
{
    unload_model();
}

/**
 * @brief Loads a Whisper model with the specified parameters.
 *
 * This function attempts to load a Whisper model using the provided parameters.
 * If a model is already loaded, it will be unloaded before loading the new one.
 * The function initializes the model context with the given parameters, including
 * GPU usage if specified. If the model fails to load, an error message is printed
 * to standard error and the function returns false. On success, a confirmation
 * message is printed and the function returns true.
 *
 * @param params The parameters to use for loading the Whisper model.
 * @return true if the model was loaded successfully, false otherwise.
 */
bool WhisperInterface::load_model(const WhisperModelParams &params)
{
    if (ctx_)
    {
        unload_model();
    }
    current_model_params_ = params;

    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = current_model_params_.use_gpu;

    ctx_ = whisper_init_from_file_with_params(current_model_params_.model.c_str(), cparams);

    if (!ctx_)
    {
        std::cerr << "WhisperInterface Error: Failed to load model from " << current_model_params_.model << std::endl;
        return false;
    }

    std::cerr << "WhisperInterface: Model loaded successfully: " << current_model_params_.model << std::endl;
    return true;
}

/**
 * @brief Unloads the currently loaded Whisper model and releases associated resources.
 *
 * This function checks if a model context (`ctx_`) is loaded. If so, it frees the context
 * using `whisper_free` and sets the context pointer to `nullptr` to prevent dangling references.
 * It also logs a message to standard error indicating that the model has been unloaded.
 */
void WhisperInterface::unload_model()
{
    if (ctx_)
    {
        whisper_free(ctx_);
        ctx_ = nullptr;
    }
    std::cerr << "WhisperInterface: Model unloaded." << std::endl;
}

/**
 * @brief Checks if the Whisper model context is loaded.
 *
 * This function returns true if the internal model context pointer (ctx_) is not null,
 * indicating that a Whisper model has been successfully loaded and is ready for use.
 *
 * @return true if the model context is loaded, false otherwise.
 */
bool WhisperInterface::is_model_loaded() const
{
    return ctx_ != nullptr;
}

void WhisperInterface::static_new_segment_callback(struct whisper_context * /*w_ctx*/, struct whisper_state * /*state*/, int /*n_new*/, void *user_data)
{
    if (user_data)
    {
        auto *instance = static_cast<WhisperInterface *>(user_data);
        if (instance->current_segment_callback_)
        {
            // TODO: Implement logic to retrieve the new segment text and timestamps
        }
    }
}

void WhisperInterface::static_progress_callback(struct whisper_context * /*w_ctx*/, struct whisper_state * /*state*/, int progress, void *user_data)
{
    if (user_data)
    {
        auto *instance = static_cast<WhisperInterface *>(user_data);
        if (instance->current_progress_callback_)
        {
            instance->current_progress_callback_(progress);
        }
    }
}

/**
 * @brief Transcribes PCM audio data using the loaded Whisper model.
 *
 * This function takes a vector of 32-bit floating point PCM audio samples and transcribes
 * the audio into text using the Whisper model. It supports various transcription parameters,
 * including language selection, translation, threading, and callback functions for progress
 * and new segment notifications.
 *
 * @param pcm_f32_data A vector containing the PCM audio data as 32-bit floats.
 * @param transcription_params Parameters controlling the transcription process, such as language,
 *        threading, translation, and callback functions.
 * @return The transcribed text as a std::string. Returns an error message string if the model
 *         is not loaded, the audio data is empty, or transcription fails.
 *
 * @note The function requires that a Whisper model is loaded before being called.
 * @note If callback functions are provided in the parameters, they will be invoked during transcription.
 */
std::string WhisperInterface::transcribe_pcm(const std::vector<float> &pcm_f32_data, const WhisperGenerationParams &transcription_params)
{
    if (!is_model_loaded())
    {
        std::cerr << "WhisperInterface Error: Model not loaded for transcription." << std::endl;
        return "[Error: Model not loaded]";
    }
    if (pcm_f32_data.empty())
    {
        std::cerr << "WhisperInterface Error: Empty audio data provided." << std::endl;
        return "[Error: Empty audio data]";
    }


}