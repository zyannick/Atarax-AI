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

bool WhisperInterface::load_model(const WhisperModelParams &params)
{
    if (ctx_)
    {
        unload_model();
    }
    current_model_params_ = params;

    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = current_model_params_.use_gpu;

    ctx_ = whisper_init_from_file_with_params(current_model_params_.model_path.c_str(), cparams);

    if (!ctx_)
    {
        std::cerr << "WhisperInterface Error: Failed to load model from " << current_model_params_.model_path << std::endl;
        return false;
    }

    std::cerr << "WhisperInterface: Model loaded successfully: " << current_model_params_.model_path << std::endl;
    return true;
}

void WhisperInterface::unload_model()
{
    if (ctx_)
    {
        whisper_free(ctx_);
        ctx_ = nullptr;
    }
    std::cerr << "WhisperInterface: Model unloaded." << std::endl;
}

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

std::string WhisperInterface::transcribe_pcm(const std::vector<float> &pcm_f32_data,
                                             const WhisperTranscriptionParams &params)
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

    current_segment_callback_ = params.new_segment_callback;
    current_progress_callback_ = params.progress_callback;

    whisper_full_params wfp = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wfp.n_threads = params.n_threads;
    wfp.language = params.language.c_str();
    wfp.translate = params.translate;
    wfp.print_progress = params.print_progress;
    wfp.no_context = params.no_context;
    wfp.max_len = params.max_len;
    wfp.single_segment = params.single_segment;
    wfp.temperature = params.temperature;

    if (params.new_segment_callback)
    {
        wfp.new_segment_callback = WhisperInterface::static_new_segment_callback;
        wfp.new_segment_callback_user_data = this;
    }
    if (params.progress_callback)
    {
        wfp.progress_callback = WhisperInterface::static_progress_callback;
        wfp.progress_callback_user_data = this;
    }

    if (whisper_full(ctx_, wfp, pcm_f32_data.data(), pcm_f32_data.size()) != 0)
    {
        std::cerr << "WhisperInterface Error: whisper_full failed." << std::endl;
        return "[Error: Transcription failed]";
    }

    std::string full_transcript = "";
    const int n_segments = whisper_full_n_segments(ctx_);
    for (int i = 0; i < n_segments; ++i)
    {
        const char *segment_text = whisper_full_get_segment_text(ctx_, i);
        if (segment_text)
        {
            full_transcript += segment_text;
            if (!params.single_segment && i < n_segments - 1)
            {
                full_transcript += " ";
            }

            if (params.new_segment_callback && !wfp.new_segment_callback)
            {
                int64_t t0 = whisper_full_get_segment_t0(ctx_, i);
                int64_t t1 = whisper_full_get_segment_t1(ctx_, i);
                params.new_segment_callback(std::string(segment_text), t0 * 10, t1 * 10);
            }
        }
    }
    current_segment_callback_ = nullptr;
    current_progress_callback_ = nullptr;

    return full_transcript;
}