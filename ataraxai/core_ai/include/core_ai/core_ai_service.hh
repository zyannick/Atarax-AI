#pragma once 

#include <string>
#include <vector>   
#include <memory>   


#include "llama_interface.hh"  
#include "whisper_interface.hh" 

class CoreAIService {
public:
    CoreAIService();
    CoreAIService(const LlamaModelParams& llama_model_params_, const WhisperModelParams& whisper_model_params_);
    ~CoreAIService();


    bool initialize_llama_model(const LlamaModelParams& llama_model_params_);

    /**
     * @brief Unloads the currently loaded Llama model.
     */
    void unload_llama_model();

    /**
     * @brief Checks if a Llama model is currently loaded.
     * @return True if a model is loaded, false otherwise.
     */
    bool is_llama_model_loaded() const;

    /**
     * @brief Processes a text prompt using the loaded Llama model and returns the generated completion.
     * This is a blocking call.
     * @param prompt_text The input prompt.
     * @param llama_generation_params Parameters controlling text generation (max tokens, temp, etc.).
     * @return The generated text completion, or an error string.
     */
    std::string process_prompt(const std::string& prompt_text, const GenerationParams& llama_generation_params_);

    /**
     * @brief Processes a text prompt with streaming output via a callback.
     * @param prompt_text The input prompt.
     * @param llama_generation_params Parameters controlling text generation.
     * @param callback Function called for each new token (or chunk) generated.
     * The callback should return true to continue generation, false to stop.
     * @return True if generation completed (or was stopped by callback), false on error.
     */
    bool stream_prompt(const std::string& prompt_text,
                       const GenerationParams& llama_generation_params,
                       llama_token_callback callback);



    /**
     * @brief Initializes and loads the Whisper speech-to-text model.
     * @param whisper_model_params Parameters for loading the Whisper model (path, GPU usage).
     * @return True if successful, false otherwise.
     */
    bool initialize_whisper_model(const WhisperModelParams& whisper_model_params_);

    /**
     * @brief Unloads the currently loaded Whisper model.
     */
    void unload_whisper_model();

    /**
     * @brief Checks if a Whisper model is currently loaded.
     * @return True if a model is loaded, false otherwise.
     */
    bool is_whisper_model_loaded() const;

    /**
     * @brief Transcribes raw audio data (PCM float32, single channel, 16kHz).
     * @param pcm_f32_data Vector of float32 audio samples.
     * @param whisper_transcription_params Parameters controlling transcription (language, translate, etc.).
     * @return The transcribed text, or an error string.
     */
    std::string transcribe_audio_pcm(const std::vector<float>& pcm_f32_data,
                                     const WhisperTranscriptionParams& whisper_transcription_params);

    /**
     * @brief Convenience function to transcribe an audio file.
     * Internally, this will read the file, convert it to the required PCM f32 format,
     * and then call transcribe_audio_pcm.
     * @param audio_file_path Path to the audio file.
     * @param whisper_transcription_params Parameters controlling transcription.
     * @return The transcribed text, or an error string.
     */
    std::string transcribe_audio_file(const std::string& audio_file_path,
                                      const WhisperTranscriptionParams& whisper_transcription_params);


    /**
     * @brief Initializes global backends for llama.cpp and whisper.cpp.
     * Should be called once when the application starts, before loading any models.
     */
    static void initialize_global_backends();

    /**
     * @brief Frees global backends for llama.cpp and whisper.cpp.
     * Should be called once when the application exits.
     */
    static void free_global_backends();


private:
    std::unique_ptr<LlamaInterface> llama_interface_;
    std::unique_ptr<WhisperInterface> whisper_interface_;

    bool llama_model_loaded_ = false;
    bool whisper_model_loaded_ = false;

    LlamaModelParams llama_model_params;
    WhisperModelParams whisper_model_params;


    std::vector<float> convert_audio_file_to_pcm_f32(const std::string& audio_file_path);
};