#pragma once
#include <string>
#include <vector>
#include <thread>
#include <algorithm>
#include <cstdint>

struct WhisperModelParams
{
    // Hardware and Core Model Configuration
    int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());
    bool use_gpu = true;
    bool flash_attn = false;
    int32_t audio_ctx = 0; // Audio context size

    // Model and Language
    std::string model = "models/ggml-base.en.bin";
    std::string language = "en";

    WhisperModelParams() = default;
    WhisperModelParams(const std::string &model, const std::string &language, bool use_gpu = true, bool flash_attn = false,
                       int32_t audio_ctx = 0, int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency()))
        : model(model), language(language), use_gpu(use_gpu), flash_attn(flash_attn), audio_ctx(audio_ctx), n_threads(n_threads) {}

    bool operator==(const WhisperModelParams &other) const
    {
        return n_threads == other.n_threads &&
               use_gpu == other.use_gpu &&
               flash_attn == other.flash_attn &&
               audio_ctx == other.audio_ctx &&
               model == other.model &&
               language == other.language;
    }

    bool operator!=(const WhisperModelParams &other) const
    {
        return !(*this == other);
    }

    std::size_t hash() const
    {
        return std::hash<std::string>()(model) ^
               std::hash<std::string>()(language) ^
               std::hash<bool>()(use_gpu) ^
               std::hash<bool>()(flash_attn) ^
               std::hash<int32_t>()(audio_ctx) ^
               std::hash<int32_t>()(n_threads);
    }

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