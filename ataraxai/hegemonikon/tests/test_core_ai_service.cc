
#include <catch2/catch_test_macros.hpp>
#include <string>
#include <vector>
#include <memory>

#include "hegemonikon/core_ai_service.hh"
#include "hegemonikon/llama_interface.hh"      
#include "hegemonikon/whisper_interface.hh"


class MockLlamaInterface : public LlamaInterface {
public:
    mutable bool load_model_called = false;
    mutable bool unload_model_called = false;
    mutable bool generate_completion_called = false;
    mutable bool generate_streaming_called = false;
    
    bool should_fail_load = false;

    bool load_model(const LlamaModelParams&) override { 
        load_model_called = true;
        return !should_fail_load;
    }

    void unload_model() override {
        unload_model_called = true;
    }

    std::string generate_completion(const std::string& prompt, const GenerationParams&) override {
        generate_completion_called = true;
        return "mocked completion: " + prompt;
    }

    bool generate_completion_streaming(const std::string&, const GenerationParams&, llama_token_callback cb) override {
        generate_streaming_called = true;
        if (cb) {
            cb("streamed");
            cb(" ");
            cb("response");
        }
        return true;
    }
};

class MockWhisperInterface : public WhisperInterface {
public:
    mutable bool load_model_called = false;

    bool load_model(const WhisperModelParams&) override { 
        load_model_called = true;
        return true; 
    }
    void unload_model() override {}
    std::string transcribe_pcm(const std::vector<float>&, const WhisperGenerationParams&) override {
        return "mocked transcription";
    }
};



TEST_CASE("CoreAIService with Mock Dependencies", "[service][unit]") {

    auto mock_llama = std::make_unique<MockLlamaInterface>();
    auto mock_whisper = std::make_unique<MockWhisperInterface>();

    MockLlamaInterface* llama_ptr = mock_llama.get();
    MockWhisperInterface* whisper_ptr = mock_whisper.get();

    CoreAIService service(std::move(mock_llama), std::move(mock_whisper));


    SECTION("Initialization and State Management") {
        REQUIRE_FALSE(service.is_llama_model_loaded());
        REQUIRE_FALSE(llama_ptr->load_model_called);

        LlamaModelParams llama_params;
        REQUIRE(service.initialize_llama_model(llama_params) == true);

        REQUIRE(service.is_llama_model_loaded() == true);
        REQUIRE(llama_ptr->load_model_called == true);
        
        service.unload_llama_model();
        
        REQUIRE(service.is_llama_model_loaded() == false);
        REQUIRE(llama_ptr->unload_model_called == true);
    }
    
    SECTION("Handles Llama model load failure correctly") {
        llama_ptr->should_fail_load = true;

        LlamaModelParams llama_params;
        
        REQUIRE(service.initialize_llama_model(llama_params) == false);
        REQUIRE(service.is_llama_model_loaded() == false);
    }

    SECTION("Text generation works correctly when model is loaded") {
        service.initialize_llama_model({}); 
        REQUIRE(service.is_llama_model_loaded());

        GenerationParams gen_params;
        std::string result = service.process_prompt("hello", gen_params);

        REQUIRE(llama_ptr->generate_completion_called == true);
        REQUIRE(result == "mocked completion: hello");
    }

    SECTION("Streaming text generation works correctly when model is loaded") {
        service.initialize_llama_model({});
        std::string accumulated_response;

        bool success = service.stream_prompt("stream test", {}, [&](const std::string& token) {
            accumulated_response += token;
            return true; // Continue streaming
        });

        REQUIRE(success == true);
        REQUIRE(llama_ptr->generate_streaming_called == true);
        REQUIRE(accumulated_response == "streamed response");
    }
}

TEST_CASE("CoreAIService with Null Dependencies (Uninitialized State)", "[service][edge_case]") {
    
    CoreAIService service;

    SECTION("Returns correct errors when no models are loaded") {
        GenerationParams gen_params;
        REQUIRE(service.process_prompt("test", gen_params) == "[Error: Llama model not loaded]");
        
        std::string error_message;
        bool success = service.stream_prompt("test", gen_params, [&](const std::string& token){
            error_message = token;
            return true;
        });
        REQUIRE(success == false);
        REQUIRE(error_message == "[Error: Llama model not loaded]");

        WhisperGenerationParams whisper_params;
        REQUIRE(service.transcribe_audio_pcm({}, whisper_params) == "[Error: Whisper model not loaded]");
    }
    
    SECTION("Unloading models when none are loaded does not crash") {
        REQUIRE_NOTHROW(service.unload_llama_model());
        REQUIRE_NOTHROW(service.unload_whisper_model());
    }
}