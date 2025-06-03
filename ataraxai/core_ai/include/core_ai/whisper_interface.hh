#pragma once
#include <string>
#include <vector>
#include <functional>

struct whisper_context;
struct whisper_context_params;

using whisper_new_segment_callback_t = std::function<void(const std::string &, int64_t, int64_t)>;

using whisper_progress_callback_t = std::function<void(int)>;

struct WhisperModelParams
{
    std::string model_path;
    bool use_gpu = true;

    WhisperModelParams() = default;
    WhisperModelParams(const std::string &path, bool gpu = true)
        : model_path(path), use_gpu(gpu) {}

    WhisperModelParams &set_model_path(const std::string &path)
    {
        model_path = path;
        return *this;
    }

    WhisperModelParams &set_use_gpu(bool gpu)
    {
        use_gpu = gpu;
        return *this;
    }

    bool operator==(const WhisperModelParams &other) const
    {
        return model_path == other.model_path && use_gpu == other.use_gpu;
    }
    bool operator!=(const WhisperModelParams &other) const
    {
        return !(*this == other);
    }
    std::size_t hash() const
    {
        return std::hash<std::string>()(model_path) ^ std::hash<bool>()(use_gpu);
    }
    std::string to_string() const
    {
        return "WhisperModelParams(model_path='" + model_path + "', use_gpu=" + (use_gpu ? "true" : "false") + ")";
    }
};

struct WhisperTranscriptionParams
{
    int n_threads = 4;
    std::string language = "en";
    bool translate = false;
    bool print_special = false;
    bool print_progress = false;
    bool no_context = true;
    int max_len = 0;
    bool single_segment = false;
    float temperature = 0.0f;

    whisper_new_segment_callback_t new_segment_callback;
    whisper_progress_callback_t progress_callback;

    WhisperTranscriptionParams() = default;

    WhisperTranscriptionParams(int threads, const std::string &lang)
        : n_threads(threads), language(lang), translate(false), print_special(false),
          print_progress(false), no_context(true), max_len(0), single_segment(false),
          temperature(0.0f) {}

    bool operator==(const WhisperTranscriptionParams &other) const
    {
        return n_threads == other.n_threads &&
               language == other.language &&
               translate == other.translate &&
               print_special == other.print_special &&
               print_progress == other.print_progress &&
               no_context == other.no_context &&
               max_len == other.max_len &&
               single_segment == other.single_segment &&
               temperature == other.temperature;
    }

    bool operator!=(const WhisperTranscriptionParams &other) const
    {
        return !(*this == other);
    }

    std::size_t hash() const
    {
        return std::hash<int>()(n_threads) ^
               std::hash<std::string>()(language) ^
               std::hash<bool>()(translate) ^
               std::hash<bool>()(print_special) ^
               std::hash<bool>()(print_progress) ^
               std::hash<bool>()(no_context) ^
               std::hash<int>()(max_len) ^
               std::hash<bool>()(single_segment) ^
               std::hash<float>()(temperature);
    }
    std::string to_string() const
    {
        return "WhisperTranscriptionParams(n_threads=" + std::to_string(n_threads) +
               ", language='" + language + "'" +
               ", translate=" + (translate ? "true" : "false") +
               ", print_special=" + (print_special ? "true" : "false") +
               ", print_progress=" + (print_progress ? "true" : "false") +
               ", no_context=" + (no_context ? "true" : "false") +
               ", max_len=" + std::to_string(max_len) +
               ", single_segment=" + (single_segment ? "true" : "false") +
               ", temperature=" + std::to_string(temperature) + ")";
    }

    WhisperTranscriptionParams &set_n_threads(int threads)
    {
        n_threads = threads;
        return *this;
    }

    WhisperTranscriptionParams &set_language(const std::string &lang)
    {
        language = lang;
        return *this;
    }

    WhisperTranscriptionParams &set_translate(bool translate)
    {
        this->translate = translate;
        return *this;
    }

    WhisperTranscriptionParams &set_print_special(bool print)
    {
        print_special = print;
        return *this;
    }

    WhisperTranscriptionParams &set_print_progress(bool print)
    {
        print_progress = print;
        return *this;
    }

    WhisperTranscriptionParams &set_no_context(bool no_ctx)
    {
        no_context = no_ctx;
        return *this;
    }

    WhisperTranscriptionParams &set_max_len(int len)
    {
        max_len = len;
        return *this;
    }

    WhisperTranscriptionParams &set_single_segment(bool single)
    {
        single_segment = single;
        return *this;
    }

    WhisperTranscriptionParams &set_temperature(float temp)
    {
        temperature = temp;
        return *this;
    }
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

