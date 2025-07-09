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



    /**
     * @brief Constructs a WhisperGenerationParams object with the specified parameters.
     *
     * @param step_ms        Step size in milliseconds for processing.
     * @param length_ms      Total length in milliseconds for the generation window.
     * @param keep_ms        Duration in milliseconds to keep from the previous window.
     * @param capture_id     Identifier for the audio capture session.
     * @param vad_thold      Voice Activity Detection (VAD) threshold.
     * @param freq_thold     Frequency threshold for filtering.
     * @param translate      If true, enables translation of the recognized text.
     * @param tinydiarize    If true, enables speaker diarization.
     * @param no_fallback    If true, disables fallback mechanisms.
     * @param no_context     If true, disables use of previous context.
     * @param max_tokens     Maximum number of tokens to generate.
     * @param beam_size      Beam size for beam search decoding.
     * @param print_special  If true, prints special tokens in the output.
     * @param no_timestamps  If true, disables timestamp generation.
     * @param save_audio     If true, saves the processed audio to a file.
     * @param fname_out      Output filename for saving audio (if enabled).
     */
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

    /**
     * @brief Equality operator for WhisperGenerationParams.
     *
     * Compares all member variables of this instance with another
     * WhisperGenerationParams instance to determine if they are equal.
     *
     * @param other The WhisperGenerationParams instance to compare with.
     * @return true if all member variables are equal, false otherwise.
     */
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

    /**
     * @brief Inequality operator for WhisperGenerationParams.
     *
     * Compares this instance with another WhisperGenerationParams object for inequality.
     * Returns true if the two objects are not equal, as determined by the equality operator.
     *
     * @param other The WhisperGenerationParams instance to compare against.
     * @return true if the objects are not equal, false otherwise.
     */
    bool operator!=(const WhisperGenerationParams &other) const
    {
        return !(*this == other);
    }

    /**
     * @brief Computes a hash value for the current object.
     *
     * This function combines the hash values of all member variables to produce
     * a single hash value representing the state of the object. It uses the
     * standard library's std::hash for each member and combines them using
     * bitwise XOR operations.
     *
     * @return std::size_t The combined hash value of the object's members.
     */
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

    /**
     * @brief Converts the WhisperTranscriptionParams object to a human-readable string representation.
     *
     * The returned string contains the values of all member variables in a key-value format,
     * making it useful for debugging and logging purposes.
     *
     * @return std::string A string representation of the current parameter values.
     */
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

    /**
     * @brief Sets the step size in milliseconds for the Whisper generation process.
     *
     * @param step_ms_ The desired step size in milliseconds.
     * @return Reference to the current WhisperGenerationParams object for method chaining.
     */
    WhisperGenerationParams &set_step_ms(int32_t step_ms_)
    {
        step_ms = step_ms_;
        return *this;
    }

    /**
     * @brief Sets the desired length in milliseconds for the generation process.
     *
     * @param length_ms_ The length in milliseconds to set.
     * @return Reference to the current WhisperGenerationParams object for method chaining.
     */
    WhisperGenerationParams &set_length_ms(int32_t length_ms_)
    {
        length_ms = length_ms_;
        return *this;
    }

    /**
     * @brief Sets the number of milliseconds to keep in the generated whisper.
     *
     * This method updates the `keep_ms` parameter, which determines how many milliseconds
     * of audio or data should be retained during the whisper generation process.
     *
     * @param keep_ms_ The number of milliseconds to keep.
     * @return Reference to the current WhisperGenerationParams object for method chaining.
     */
    WhisperGenerationParams &set_keep_ms(int32_t keep_ms_)
    {
        keep_ms = keep_ms_;
        return *this;
    }

    /**
     * @brief Sets the capture ID for the WhisperGenerationParams object.
     *
     * This method assigns the provided capture ID to the internal member variable.
     *
     * @param capture_id_ The capture ID to set.
     * @return Reference to the current WhisperGenerationParams object for method chaining.
     */
    WhisperGenerationParams &set_capture_id(int32_t capture_id_)
    {
        capture_id = capture_id_;
        return *this;
    }

    /**
     * @brief Sets the voice activity detection (VAD) threshold parameter.
     *
     * This method updates the VAD threshold value used for detecting speech activity.
     * A higher threshold may result in less sensitivity to speech, while a lower threshold
     * increases sensitivity.
     *
     * @param vad_thold_ The new VAD threshold value to set.
     * @return Reference to the current WhisperGenerationParams object for method chaining.
     */
    WhisperGenerationParams &set_vad_thold(float vad_thold_)
    {
        vad_thold = vad_thold_;
        return *this;
    }

    /**
     * @brief Sets the frequency threshold parameter.
     *
     * This method updates the frequency threshold value used in the generation process.
     *
     * @param freq_thold_ The new frequency threshold value to set.
     * @return Reference to the current WhisperGenerationParams object for method chaining.
     */
    WhisperGenerationParams &set_freq_thold(float freq_thold_)
    {
        freq_thold = freq_thold_;
        return *this;
    }

    /**
     * @brief Sets the translation mode for Whisper generation.
     *
     * @param translate_ If true, enables translation of the generated output.
     * @return Reference to the current WhisperGenerationParams object for method chaining.
     */
    WhisperGenerationParams &set_translate(bool translate_)
    {
        translate = translate_;
        return *this;
    }

    /**
     * @brief Sets the TinyDiarize option for the WhisperGenerationParams.
     *
     * This method enables or disables the TinyDiarize feature, which may control
     * whether speaker diarization is performed using a lightweight approach.
     *
     * @param tinydiarize_ Boolean value to enable (true) or disable (false) TinyDiarize.
     * @return Reference to the current WhisperGenerationParams object for method chaining.
     */
    WhisperGenerationParams &set_tinydiarize(bool tinydiarize_)
    {
        tinydiarize = tinydiarize_;
        return *this;
    }

    /**
     * @brief Sets the no_fallback parameter for whisper generation.
     *
     * This method configures whether the fallback mechanism should be disabled during
     * the whisper generation process.
     *
     * @param no_fallback_ If true, disables fallback; if false, enables fallback.
     * @return Reference to the current WhisperGenerationParams object for method chaining.
     */
    WhisperGenerationParams &set_no_fallback(bool no_fallback_)
    {
        no_fallback = no_fallback_;
        return *this;
    }

    /**
     * @brief Sets whether to disable context during generation.
     *
     * @param no_context_ If true, context will be disabled; otherwise, context will be used.
     * @return Reference to the current WhisperGenerationParams object for method chaining.
     */
    WhisperGenerationParams &set_no_context(bool no_context_)
    {
        no_context = no_context_;
        return *this;
    }

    /**
     * @brief Sets the maximum number of tokens to generate.
     *
     * This method updates the maximum token limit for the generation process.
     *
     * @param max_tokens_ The new maximum number of tokens.
     * @return Reference to the current WhisperGenerationParams object for method chaining.
     */
    WhisperGenerationParams &set_max_tokens(int32_t max_tokens_)
    {
        max_tokens = max_tokens_;
        return *this;
    }

    /**
     * @brief Sets the beam size parameter for the Whisper generation.
     *
     * This method updates the beam size used during generation and returns a reference
     * to the current object to allow for method chaining.
     *
     * @param beam_size_ The desired beam size value.
     * @return Reference to the current WhisperGenerationParams object.
     */
    WhisperGenerationParams &set_beam_size(int32_t beam_size_)
    {
        beam_size = beam_size_;
        return *this;
    }

    /**
     * @brief Sets whether special tokens should be printed during generation.
     *
     * @param print_special_ Boolean flag indicating if special tokens should be printed.
     * @return Reference to the current WhisperGenerationParams object for method chaining.
     */
    WhisperGenerationParams &set_print_special(bool print_special_)
    {
        print_special = print_special_;
        return *this;
    }

    /**
     * @brief Sets whether to disable timestamps in the generation parameters.
     *
     * @param no_timestamps_ If true, disables timestamps; otherwise, enables them.
     * @return Reference to this WhisperGenerationParams object for method chaining.
     */
    WhisperGenerationParams &set_no_timestamps(bool no_timestamps_)
    {
        no_timestamps = no_timestamps_;
        return *this;
    }

    /**
     * @brief Sets the flag indicating whether to save audio output.
     *
     * This method updates the internal state to specify if the generated audio should be saved.
     *
     * @param save_audio_ Boolean value indicating whether to save audio (true) or not (false).
     * @return Reference to the current WhisperGenerationParams object for method chaining.
     */
    WhisperGenerationParams &set_save_audio(bool save_audio_)
    {
        save_audio = save_audio_;
        return *this;
    }

    /**
     * @brief Sets the output file name for the Whisper generation process.
     *
     * This method assigns the provided file name to the internal `fname_out` member,
     * which specifies where the output should be saved. It returns a reference to the
     * current object to allow for method chaining.
     *
     * @param fname_out_ The name of the output file.
     * @return Reference to the current WhisperGenerationParams object.
     */
    WhisperGenerationParams &set_fname_out(const std::string &fname_out_)
    {
        fname_out = fname_out_;
        return *this;
    }

    /**
     * @brief Sets the temperature parameter for generation.
     *
     * The temperature parameter controls the randomness of the generation process.
     * Higher values (e.g., 1.0) make the output more random, while lower values (e.g., 0.2)
     * make it more deterministic.
     *
     * @param temperature_ The new temperature value to set.
     * @return Reference to the current WhisperGenerationParams object for method chaining.
     */
    WhisperGenerationParams &set_temperature(float temperature_)
    {
        temperature = temperature_;
        return *this;
    }

    /**
     * @brief Sets the temperature increment parameter for whisper generation.
     *
     * This method updates the temperature increment value used during the generation process.
     * The temperature increment can influence the randomness or creativity of the generated output.
     *
     * @param temperature_inc_ The new temperature increment value to set.
     * @return Reference to the current WhisperGenerationParams object for method chaining.
     */
    WhisperGenerationParams &set_temperature_inc(float temperature_inc_)
    {
        temperature_inc = temperature_inc_;
        return *this;
    }

    /**
     * @brief Sets the word threshold parameter.
     *
     * This method updates the internal word threshold value used in the generation process.
     * The word threshold may be used to filter or control the output based on word-level confidence or probability.
     *
     * @param word_thold_ The new word threshold value to set.
     * @return Reference to the current WhisperGenerationParams object for method chaining.
     */
    WhisperGenerationParams &set_word_thold(float word_thold_)
    {
        word_thold = word_thold_;
        return *this;
    }

    /**
     * @brief Sets the entropy threshold parameter for whisper generation.
     *
     * This method updates the entropy threshold value used during the generation process.
     * A higher entropy threshold may allow for more diverse outputs, while a lower value
     * can make the generation more deterministic.
     *
     * @param entropy_thold_ The new entropy threshold value to set.
     * @return Reference to the current WhisperGenerationParams object for method chaining.
     */
    WhisperGenerationParams &set_entropy_thold(float entropy_thold_)
    {
        entropy_thold = entropy_thold_;
        return *this;
    }

    /**
     * @brief Sets the log probability threshold for generation.
     *
     * This method updates the internal logprob_thold value, which is used to determine
     * the minimum log probability required for a token to be considered during generation.
     *
     * @param logprob_thold_ The new log probability threshold value.
     * @return Reference to the current WhisperGenerationParams object for method chaining.
     */
    WhisperGenerationParams &set_logprob_thold(float logprob_thold_)
    {
        logprob_thold = logprob_thold_;
        return *this;
    }

    /**
     * @brief Sets the no speech threshold parameter.
     *
     * This method sets the threshold value used to determine the absence of speech.
     * If the detected speech probability is below this threshold, the system may
     * consider that there is no speech present in the input.
     *
     * @param no_speech_thold_ The new threshold value for no speech detection.
     * @return Reference to the current WhisperGenerationParams object for method chaining.
     */
    WhisperGenerationParams &set_no_speech_thold(float no_speech_thold_)
    {
        no_speech_thold = no_speech_thold_;
        return *this;
    }

    /**
     * @brief Sets the audio context parameter.
     *
     * This method assigns the provided audio context value to the internal
     * audio_ctx member variable. It enables method chaining by returning
     * a reference to the current WhisperGenerationParams instance.
     *
     * @param audio_ctx_ The audio context value to set.
     * @return Reference to the current WhisperGenerationParams object.
     */
    WhisperGenerationParams &set_audio_ctx(int32_t audio_ctx_)
    {
        audio_ctx = audio_ctx_;
        return *this;
    }

    /**
     * @brief Sets the number of best candidates to consider during generation.
     *
     * This method updates the 'best_of' parameter, which determines how many
     * candidate sequences are generated and considered before selecting the best one.
     *
     * @param best_of_ The number of best candidates to consider.
     * @return Reference to the current WhisperGenerationParams object for method chaining.
     */
    WhisperGenerationParams &set_best_of(int32_t best_of_)
    {
        best_of = best_of_;
        return *this;
    }
};