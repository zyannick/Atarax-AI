#pragma once
#include <string>
#include <vector>
#include <functional>

#include "whisper_model_params.hh"
#include "whisper_generation_params.hh"

struct whisper_context;
struct whisper_context_params;

class WhisperInterface
{
public:
    WhisperInterface();
    ~WhisperInterface();

    virtual bool load_model(const HegemonikonWhisperModelParams &params);

    virtual void unload_model();

    bool is_model_loaded() const;

    virtual std::string transcribe_pcm(const std::vector<float> &pcm_f32_data,
                               const HegemonikonWhisperGenerationParams &params);

    static void init_backend();
    static void free_backend();

private:
    whisper_context *ctx_ = nullptr;
    HegemonikonWhisperModelParams current_model_params_;

    static void static_new_segment_callback(struct whisper_context *ctx, struct whisper_state *state, int n_new, void *user_data);
    static void static_progress_callback(struct whisper_context *ctx, struct whisper_state *state, int progress, void *user_data);

    whisper_new_segment_callback_t current_segment_callback_;
    whisper_progress_callback_t current_progress_callback_;
};
