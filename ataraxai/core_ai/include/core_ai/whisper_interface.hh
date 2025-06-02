#ifndef WHISPER_INTERFACE_HH
#define WHISPER_INTERFACE_HH

#include <string>
#include <vector>
#include <functional> // For std::function (callbacks)

// Forward declare whisper.cpp types to keep whisper.h out of this public header if possible.
struct whisper_context;
struct whisper_context_params; // You might wrap this if needed

// Define callback types for transcription results
// Callback for each new segment transcribed
// Parameters: segment_text, start_timestamp_ms, end_timestamp_ms
using whisper_new_segment_callback_t = std::function<void(const std::string &, int64_t, int64_t)>;
// Callback for overall progress
// Parameter: progress percentage (0-100)
using whisper_progress_callback_t = std::function<void(int)>;

// Parameters for loading a Whisper model
struct WhisperModelParams
{
    std::string model_path;
    bool use_gpu = true; // Attempt to use GPU if available and whisper.cpp is compiled with GPU support
    // int gpu_device = 0; // Optional: specify GPU device index
};

// Parameters for performing a transcription
struct WhisperTranscriptionParams
{
    int n_threads = 4;           // Number of CPU threads to use for computation
    std::string language = "en"; // Target language (e.g., "en", "es", "auto" for detection)
    bool translate = false;      // Translate from source language to English
    bool print_special = false;  // Print special tokens (e.g., <SOT>, <EOT>)
    bool print_progress = false; // Print progress to stderr (whisper.cpp internal)
    bool no_context = true;      // Disable context from previous audio (for isolated transcriptions)
    int max_len = 0;             // Max segment length in characters (0 for default)
    bool single_segment = false; // Force single segment output
    float temperature = 0.0f;    // Temperature for sampling (0.0 for greedy)
    // Add other whisper_full_params as needed (e.g., beam_size, word_timestamps, suppress_tokens)

    // Callbacks
    whisper_new_segment_callback_t new_segment_callback;
    whisper_progress_callback_t progress_callback;
};

class WhisperInterface
{
public:
    WhisperInterface();
    ~WhisperInterface();

    bool load_model(const WhisperModelParams &params);

    void unload_model();

    bool is_model_loaded() const;

    std::string transcribe_pcm(const std::vector<float> &pcm_f32_data,
                               const WhisperTranscriptionParams &params);

    static void init_backend();
    static void free_backend();

private:
    whisper_context *ctx_ = nullptr;
    WhisperModelParams current_model_params_;

    static void static_new_segment_callback(struct whisper_context *ctx, struct whisper_state *state, int n_new, void *user_data);
    static void static_progress_callback(struct whisper_context *ctx, struct whisper_state *state, int progress, void *user_data);

    whisper_new_segment_callback_t current_segment_callback_;
    whisper_progress_callback_t current_progress_callback_;
};

#endif