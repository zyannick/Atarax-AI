#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <thread>
#include <functional>

using whisper_new_segment_callback_t = std::function<void(const std::string &, int64_t, int64_t)>;

using whisper_progress_callback_t = std::function<void(int)>;

struct WhisperGenerationParams
{
    int32_t step_ms = 3000;
    int32_t length_ms = 10000;
    int32_t keep_ms = 200;
    int32_t capture_id = -1;
    float vad_thold = 0.6f;
    float freq_thold = 100.0f;

    bool translate = false;
    bool tinydiarize = false;
    bool no_fallback = false;
    bool no_context = true;
    int32_t max_tokens = 32;
    int32_t beam_size = -1;
    int32_t best_of = 2;
    int32_t audio_ctx = 0;

    float word_thold = 0.01f;
    float entropy_thold = 2.40f;
    float logprob_thold = -1.00f;
    float temperature = 0.00f;
    float temperature_inc = 0.20f;
    float no_speech_thold = 0.6f;

    bool print_special = false;
    bool no_timestamps = false;
    bool save_audio = false;
    std::string fname_out;

    WhisperGenerationParams() = default;
    WhisperGenerationParams(int32_t step_ms_, int32_t length_ms_, int32_t keep_ms_,
                            int32_t capture_id_, float vad_thold_, float freq_thold_,
                            bool translate_, bool tinydiarize_, bool no_fallback_,
                            bool no_context_, int32_t max_tokens_, int32_t beam_size_,
                            bool print_special_, bool no_timestamps_,
                            bool save_audio_, const std::string &fname_out_)
        : step_ms(step_ms_), length_ms(length_ms_), keep_ms(keep_ms_),
          capture_id(capture_id_), vad_thold(vad_thold_), freq_thold(freq_thold_),
          translate(translate_), tinydiarize(tinydiarize_), no_fallback(no_fallback_),
          no_context(no_context_), max_tokens(max_tokens_), beam_size(beam_size_),
          print_special(print_special_), no_timestamps(no_timestamps_),
          save_audio(save_audio_), fname_out(fname_out_) {}

    bool operator==(const WhisperGenerationParams &other) const
    {
        return step_ms == other.step_ms &&
               length_ms == other.length_ms &&
               keep_ms == other.keep_ms &&
               capture_id == other.capture_id &&
               vad_thold == other.vad_thold &&
               freq_thold == other.freq_thold &&
               translate == other.translate &&
               tinydiarize == other.tinydiarize &&
               no_fallback == other.no_fallback &&
               no_context == other.no_context &&
               max_tokens == other.max_tokens &&
               beam_size == other.beam_size &&
               print_special == other.print_special &&
               no_timestamps == other.no_timestamps &&
               save_audio == other.save_audio &&
               fname_out == other.fname_out;
    }

    bool operator!=(const WhisperGenerationParams &other) const
    {
        return !(*this == other);
    }

    std::size_t hash() const
    {
        return std::hash<int32_t>()(step_ms) ^
               std::hash<int32_t>()(length_ms) ^
               std::hash<int32_t>()(keep_ms) ^
               std::hash<int32_t>()(capture_id) ^
               std::hash<float>()(vad_thold) ^
               std::hash<float>()(freq_thold) ^
               std::hash<bool>()(translate) ^
               std::hash<bool>()(tinydiarize) ^
               std::hash<bool>()(no_fallback) ^
               std::hash<bool>()(no_context) ^
               std::hash<int32_t>()(max_tokens) ^
               std::hash<int32_t>()(beam_size) ^
               std::hash<bool>()(print_special) ^
               std::hash<bool>()(no_timestamps) ^
               std::hash<bool>()(save_audio) ^
               std::hash<std::string>()(fname_out);
    }

    std::string to_string() const
    {
        return "WhisperTranscriptionParams(step_ms=" + std::to_string(step_ms) +
               ", length_ms=" + std::to_string(length_ms) +
               ", keep_ms=" + std::to_string(keep_ms) +
               ", capture_id=" + std::to_string(capture_id) +
               ", vad_thold=" + std::to_string(vad_thold) +
               ", freq_thold=" + std::to_string(freq_thold) +
               ", translate=" + (translate ? "true" : "false") +
               ", tinydiarize=" + (tinydiarize ? "true" : "false") +
               ", no_fallback=" + (no_fallback ? "true" : "false") +
               ", no_context=" + (no_context ? "true" : "false") +
               ", max_tokens=" + std::to_string(max_tokens) +
               ", beam_size=" + std::to_string(beam_size) +
               ", print_special=" + (print_special ? "true" : "false") +
               ", no_timestamps=" + (no_timestamps ? "true" : "false") +
               ", save_audio=" + (save_audio ? "true" : "false") +
               ", fname_out='" + fname_out + "')";
    }

    WhisperGenerationParams &set_step_ms(int32_t step_ms_)
    {
        step_ms = step_ms_;
        return *this;
    }

    WhisperGenerationParams &set_length_ms(int32_t length_ms_)
    {
        length_ms = length_ms_;
        return *this;
    }

    WhisperGenerationParams &set_keep_ms(int32_t keep_ms_)
    {
        keep_ms = keep_ms_;
        return *this;
    }

    WhisperGenerationParams &set_capture_id(int32_t capture_id_)
    {
        capture_id = capture_id_;
        return *this;
    }

    WhisperGenerationParams &set_vad_thold(float vad_thold_)
    {
        vad_thold = vad_thold_;
        return *this;
    }

    WhisperGenerationParams &set_freq_thold(float freq_thold_)
    {
        freq_thold = freq_thold_;
        return *this;
    }

    WhisperGenerationParams &set_translate(bool translate_)
    {
        translate = translate_;
        return *this;
    }

    WhisperGenerationParams &set_tinydiarize(bool tinydiarize_)
    {
        tinydiarize = tinydiarize_;
        return *this;
    }

    WhisperGenerationParams &set_no_fallback(bool no_fallback_)
    {
        no_fallback = no_fallback_;
        return *this;
    }

    WhisperGenerationParams &set_no_context(bool no_context_)
    {
        no_context = no_context_;
        return *this;
    }

    WhisperGenerationParams &set_max_tokens(int32_t max_tokens_)
    {
        max_tokens = max_tokens_;
        return *this;
    }

    WhisperGenerationParams &set_beam_size(int32_t beam_size_)
    {
        beam_size = beam_size_;
        return *this;
    }

    WhisperGenerationParams &set_print_special(bool print_special_)
    {
        print_special = print_special_;
        return *this;
    }

    WhisperGenerationParams &set_no_timestamps(bool no_timestamps_)
    {
        no_timestamps = no_timestamps_;
        return *this;
    }

    WhisperGenerationParams &set_save_audio(bool save_audio_)
    {
        save_audio = save_audio_;
        return *this;
    }

    WhisperGenerationParams &set_fname_out(const std::string &fname_out_)
    {
        fname_out = fname_out_;
        return *this;
    }

    WhisperGenerationParams &set_temperature(float temperature_)
    {
        temperature = temperature_;
        return *this;
    }

    WhisperGenerationParams &set_temperature_inc(float temperature_inc_)
    {
        temperature_inc = temperature_inc_;
        return *this;
    }

    WhisperGenerationParams &set_word_thold(float word_thold_)
    {
        word_thold = word_thold_;
        return *this;
    }

    WhisperGenerationParams &set_entropy_thold(float entropy_thold_)
    {
        entropy_thold = entropy_thold_;
        return *this;
    }

    WhisperGenerationParams &set_logprob_thold(float logprob_thold_)
    {
        logprob_thold = logprob_thold_;
        return *this;
    }

    WhisperGenerationParams &set_no_speech_thold(float no_speech_thold_)
    {
        no_speech_thold = no_speech_thold_;
        return *this;
    }

    WhisperGenerationParams &set_audio_ctx(int32_t audio_ctx_)
    {
        audio_ctx = audio_ctx_;
        return *this;
    }

    WhisperGenerationParams &set_best_of(int32_t best_of_)
    {
        best_of = best_of_;
        return *this;
    }
};