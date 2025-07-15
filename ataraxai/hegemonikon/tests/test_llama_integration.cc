#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <iostream>

#include "hegemonikon/llama_interface.hh"

const std::string REAL_LLAMA_MODEL_PATH = TEST_LLAMA_MODEL_PATH;

TEST_CASE("LlamaInterface can load and use a real GGUF model", "[integration][llama]") {
    

    if (!std::filesystem::exists(REAL_LLAMA_MODEL_PATH)) {
        WARN("SKIPPING Llama integration test: Model file not found at " << REAL_LLAMA_MODEL_PATH);
        return;
    }

    LlamaInterface llama_service;
    LlamaModelParams params;
    params.model_path = REAL_LLAMA_MODEL_PATH;

    REQUIRE(llama_service.load_model(params) == true);
    REQUIRE(llama_service.is_model_loaded() == true);

    llama_service.unload_model();
    REQUIRE(llama_service.is_model_loaded() == false);
}
