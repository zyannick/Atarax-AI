cmake_minimum_required(VERSION 3.16)
project(AtaraxIA LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ATARAXAI_USE_CUDA "Enable CUDA support for AtaraxAI" OFF)
add_compile_options(/utf-8)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(FetchContent)


set(LLAMA_BUILD_TESTS OFF CACHE BOOL "")
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "")
set(GGML_CUDA ${ATARAXAI_USE_CUDA} CACHE BOOL "Propagate CUDA support to llama.cpp")

FetchContent_Declare(
  llama_cpp
  GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
  GIT_TAG b6935
)

FetchContent_MakeAvailable(llama_cpp)

set(WHISPER_BUILD_TESTS OFF CACHE BOOL "")
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "")
set(WHISPER_CUDA OFF CACHE BOOL "" FORCE)
set(WHISPER_COREML OFF CACHE BOOL "" FORCE)
set(WHISPER_CLBLAST OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "")
set(GGML_CUDA ${ATARAXAI_USE_CUDA} CACHE BOOL "Propagate CUDA support to whisper.cpp")

FetchContent_Declare(
  whisper_cpp
  GIT_REPOSITORY https://github.com/ggml-org/whisper.cpp.git
  GIT_TAG v1.8.2
)
FetchContent_MakeAvailable(whisper_cpp)


add_subdirectory(ataraxai/hegemonikon)