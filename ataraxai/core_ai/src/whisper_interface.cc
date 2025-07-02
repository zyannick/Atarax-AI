#include "core_ai/whisper_interface.hh"
#include "whisper.h"
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <memory>
#include <mutex>
#include <atomic>
#include <thread>
#include <fstream>

static std::once_flag backend_whisper_init_flag;
static std::atomic<bool> backend_whisper_initialized{false};

WhisperInterface::WhisperInterface() : ctx_(nullptr)
{

    std::cerr << "WhisperInterface created. Ensure whisper backend (if any specific) is initialized if needed." << std::endl;
}

WhisperInterface::~WhisperInterface()
{
    unload_model();
}

void WhisperInterface::init_backend()
{
    // std::call_once(backend_whisper_init_flag, []() {
    //     whisper_backend_init();
    //     std::cerr << "Whisper backend initialized." << std::endl;
    // });
}

void WhisperInterface::free_backend()
{
    // TODO
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
    cparams.flash_attn = current_model_params_.flash_attn;

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

    std::cout << "WhisperInterface: Starting transcription..." << std::endl;

    std::vector<whisper_token> prompt_tokens;

    // run the inference
    {
        whisper_full_params wparams = whisper_full_default_params(transcription_params.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY);

        std::cout << "Start 1" << std::endl;
        wparams.print_progress = false;
        wparams.print_special = transcription_params.print_special;
        wparams.print_realtime = false;
        wparams.print_timestamps = !transcription_params.no_timestamps;
        wparams.translate = transcription_params.translate;
        wparams.single_segment = true;
        wparams.max_tokens = transcription_params.max_tokens;
        wparams.language = current_model_params_.language.c_str();
        wparams.n_threads = current_model_params_.n_threads;
        wparams.beam_search.beam_size = transcription_params.beam_size;

        wparams.audio_ctx = transcription_params.audio_ctx;

        wparams.tdrz_enable = transcription_params.tinydiarize; // [TDRZ]

        // disable temperature fallback
        wparams.temperature_inc = transcription_params.no_fallback ? 0.0f : wparams.temperature_inc;
        wparams.duration_ms = 1000.0f * pcm_f32_data.size() / 16000.0f;


        wparams.prompt_tokens = transcription_params.no_context ? nullptr : prompt_tokens.data();
        wparams.prompt_n_tokens = transcription_params.no_context ? 0 : prompt_tokens.size();

        std::cout << "pcm_f32_data.size()  " << pcm_f32_data.size() << std::endl;

        if (whisper_full(ctx_, wparams, pcm_f32_data.data(), pcm_f32_data.size()) != 0)
        {
            return "[Error: Whisper full processing failed]";
        }

        std::cout << "whisper_full if passed  " << pcm_f32_data.size() << std::endl;

        std::string result;
        std::ofstream fout;

        

        if (transcription_params.fname_out.length() > 0)
        {
            fout.open(transcription_params.fname_out);
            if (!fout.is_open())
            {
                std::cerr << "Warning: Could not open output file: " << transcription_params.fname_out << std::endl;
            }
        }

        const int n_segments = whisper_full_n_segments(ctx_);
        for (int i = 0; i < n_segments; ++i)
        {
            const char *text = whisper_full_get_segment_text(ctx_, i);

            std::cout << "WhisperInterface: Segment " << i << ": " << text << std::endl;

            if (transcription_params.no_timestamps)
            {
                result += text;

                // // Optional: still print to stdout if needed
                // if (transcription_params.print_to_stdout)
                // {
                //     printf("%s", text);
                //     fflush(stdout);
                // }

                if (fout.is_open())
                {
                    fout << text;
                }
            }
            else
            {
                const int64_t t0 = whisper_full_get_segment_t0(ctx_, i);
                const int64_t t1 = whisper_full_get_segment_t1(ctx_, i);

                // Format with timestamps
                char timestamp_buffer[64];
                snprintf(timestamp_buffer, sizeof(timestamp_buffer), "[%02d:%02d.%03d --> %02d:%02d.%03d] ",
                         (int)(t0 / 600000), (int)(t0 / 10000) % 60, (int)(t0 % 10000) / 10,
                         (int)(t1 / 600000), (int)(t1 / 10000) % 60, (int)(t1 % 10000) / 10);

                std::string output = std::string(timestamp_buffer) + text;

                if (whisper_full_get_segment_speaker_turn_next(ctx_, i))
                {
                    output += " [SPEAKER_TURN]";
                }

                output += "\n";
                result += output;

                // if (transcription_params.print_to_stdout)
                // {
                //     printf("%s", output.c_str());
                //     fflush(stdout);
                // }

                if (fout.is_open())
                {
                    fout << output;
                }
            }
        }

        if (fout.is_open())
        {
            fout.close();
        }

        return result;
    }
}