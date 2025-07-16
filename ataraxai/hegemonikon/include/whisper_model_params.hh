#pragma once
#include <string>
#include <vector>
#include <thread>
#include <algorithm>
#include <cstdint>

struct WhisperModelParams
{
    int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());
    bool use_gpu = true;
    bool flash_attn = false;
    int32_t audio_ctx = 0; 

    std::string model = "models/ggml-base.en.bin";
    std::string language = "en";

    WhisperModelParams() = default;
    /**
     * @brief Constructs a WhisperModelParams object with the specified parameters.
     *
     * @param model       Path or identifier of the Whisper model to use.
     * @param language    Language code (e.g., "en", "fr") for transcription or translation.
     * @param use_gpu     Whether to use GPU acceleration (default: true).
     * @param flash_attn  Enable Flash Attention optimization if supported (default: false).
     * @param audio_ctx   Audio context size or window (default: 0 for automatic).
     * @param n_threads   Number of threads to use for processing (default: minimum of 4 or hardware concurrency).
     */
    WhisperModelParams(const std::string &model, const std::string &language, bool use_gpu = true, bool flash_attn = false,
                       int32_t audio_ctx = 0, int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency()))
        : model(model), language(language), use_gpu(use_gpu), flash_attn(flash_attn), audio_ctx(audio_ctx), n_threads(n_threads) {}

    /**
     * @brief Equality operator for WhisperModelParams.
     *
     * Compares this instance with another WhisperModelParams object to determine if all
     * configuration parameters are equal. The comparison includes the number of threads,
     * GPU usage flag, flash attention flag, audio context size, model identifier, and language.
     *
     * @param other The WhisperModelParams instance to compare against.
     * @return true if all parameters are equal; false otherwise.
     */
    bool operator==(const WhisperModelParams &other) const
    {
        return n_threads == other.n_threads &&
               use_gpu == other.use_gpu &&
               flash_attn == other.flash_attn &&
               audio_ctx == other.audio_ctx &&
               model == other.model &&
               language == other.language;
    }

    /**
     * @brief Inequality operator for WhisperModelParams.
     *
     * Compares this instance with another WhisperModelParams object for inequality.
     * Returns true if the two objects are not equal, as determined by the equality operator.
     *
     * @param other The WhisperModelParams object to compare with.
     * @return true if the objects are not equal, false otherwise.
     */
    bool operator!=(const WhisperModelParams &other) const
    {
        return !(*this == other);
    }

    /**
     * @brief Computes a combined hash value for the object's parameters.
     *
     * This function generates a hash by combining the hash values of the member variables:
     * - model (std::string)
     * - language (std::string)
     * - use_gpu (bool)
     * - flash_attn (bool)
     * - audio_ctx (int32_t)
     * - n_threads (int32_t)
     *
     * The resulting hash can be used for storing objects in hash-based containers.
     *
     * @return std::size_t The combined hash value of the object's parameters.
     */
    std::size_t hash() const
    {
        return std::hash<std::string>()(model) ^
               std::hash<std::string>()(language) ^
               std::hash<bool>()(use_gpu) ^
               std::hash<bool>()(flash_attn) ^
               std::hash<int32_t>()(audio_ctx) ^
               std::hash<int32_t>()(n_threads);
    }

    /**
     * @brief Converts the WhisperModelParams object to a human-readable string representation.
     *
     * @return A string describing the current model parameters, including model name, language,
     *         GPU usage, flash attention status, audio context size, and number of threads.
     */
    std::string to_string() const
    {
        return "WhisperModelParams(model='" + model +
               "', language='" + language +
               "', use_gpu=" + (use_gpu ? "true" : "false") +
               ", flash_attn=" + (flash_attn ? "true" : "false") +
               ", audio_ctx=" + std::to_string(audio_ctx) +
               ", n_threads=" + std::to_string(n_threads) + ")";
    }

    /**
     * @brief Sets the model path.
     *
     * This method sets the model file path to the specified value.
     *
     * @param path The file system path to the model file.
     * @return Reference to the current WhisperModelParams object to allow method chaining.
     */
    WhisperModelParams &set_model_path(const std::string &path)
    {
        model = path;
        return *this;
    }

    /**
     * @brief Sets the language for the model.
     *
     * This method sets the language code to be used by the model.
     *
     * @param lang The language code (e.g., "en" for English).
     * @return Reference to the current WhisperModelParams object for method chaining.
     */
    WhisperModelParams &set_language(const std::string &lang)
    {
        language = lang;
        return *this;
    }

    /**
     * @brief Sets whether to use GPU for processing.
     *
     * This method enables or disables GPU usage based on the provided boolean value.
     *
     * @param gpu If true, enables GPU usage; if false, disables it.
     * @return Reference to the current WhisperModelParams object for method chaining.
     */
    WhisperModelParams &set_use_gpu(bool gpu)
    {
        use_gpu = gpu;
        return *this;
    }

    /**
     * @brief Sets whether to use flash attention.
     *
     * This method enables or disables flash attention based on the provided boolean value.
     *
     * @param flash If true, enables flash attention; if false, disables it.
     * @return Reference to the current WhisperModelParams object for method chaining.
     */
    WhisperModelParams &set_flash_attn(bool flash)
    {
        flash_attn = flash;
        return *this;
    }

    /**
     * @brief Sets the audio context size.
     *
     * This method sets the audio context size to be used by the model.
     *
     * @param ctx The desired audio context size.
     * @return Reference to the current WhisperModelParams object for method chaining.
     */
    WhisperModelParams &set_audio_ctx(int32_t ctx)
    {
        audio_ctx = ctx;
        return *this;
    }

    /**
     * @brief Sets the number of threads to be used.
     *
     * This method sets the number of threads for processing, which can help optimize performance.
     *
     * @param threads The desired number of threads.
     * @return Reference to the current WhisperModelParams object for method chaining.
     */
    WhisperModelParams &set_n_threads(int32_t threads)
    {
        n_threads = threads;
        return *this;
    }
};