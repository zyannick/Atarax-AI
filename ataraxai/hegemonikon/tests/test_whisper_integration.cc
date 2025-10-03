#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <iostream>
#include <vector>
#include <cmath>

#include "whisper_interface.hh"

const std::string REAL_WHISPER_MODEL_PATH = TEST_WHISPER_MODEL_PATH;

TEST_CASE("WhisperInterface can load and use a real GGUF model", "[integration][whisper]")
{

    if (!std::filesystem::exists(REAL_WHISPER_MODEL_PATH))
    {
        WARN("SKIPPING Whisper integration test: Model file not found at " << REAL_WHISPER_MODEL_PATH);
        return;
    }

    WhisperInterface whisper_service;
    HegemonikonWhisperModelParams params;
    params.model = REAL_WHISPER_MODEL_PATH;

    REQUIRE(whisper_service.load_model(params) == true);

    std::vector<float> dummy_pcm(16000);
    for (size_t i = 0; i < dummy_pcm.size(); ++i)
    {
        dummy_pcm[i] = 0.5f * sin(2.0f * 3.14159f * 440.0f * i / 16000.0f);
    }

    HegemonikonWhisperGenerationParams gen_params;
    std::string result = whisper_service.transcribe_pcm(dummy_pcm, gen_params);

    REQUIRE(!result.empty());
    std::cout << "Whisper integration test response: " << result << std::endl;
}