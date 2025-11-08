cmake_minimum_required(VERSION 3.18)
project(HegemonikonModule LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ATARAXAI_USE_CUDA "Enable CUDA support for AtaraxAI" OFF)

include(FetchContent)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.12.0 
)
FetchContent_MakeAvailable(pybind11)

find_package(Python REQUIRED COMPONENTS Interpreter Development)
if(NOT Python_FOUND)
    message(FATAL_ERROR "Could not find a valid Python installation.")
else()
    message(STATUS "Found Python Interpreter: ${Python_EXECUTABLE}")
    message(STATUS "Found Python Includes:   ${Python_INCLUDE_DIRS}")
    message(STATUS "Found Python Libraries:  ${Python_LIBRARIES}")
endif()

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP: ${OpenMP_CXX_VERSION}")
else()
    message(FATAL_ERROR "OpenMP not found")
endif()

add_library(hegemonikon STATIC
    src/core_ai_service.cc
    src/llama_interface.cc
    src/whisper_interface.cc
    src/argon2/argon2-core.cpp
    src/argon2/argon2-opt-core.cpp
    src/argon2/argon2-ref-core.cpp
    src/argon2/argon2.cpp
    src/argon2/kat.cpp
    src/argon2/blake2b.c
)

set_target_properties(hegemonikon PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(hegemonikon PRIVATE
    llama     
    whisper    
    OpenMP::OpenMP_CXX 
)

if(WIN32)
    target_link_libraries(hegemonikon PRIVATE Psapi.lib)
else()
    target_link_libraries(hegemonikon PRIVATE pthread dl m)
endif()

target_include_directories(hegemonikon PUBLIC 
    "${CMAKE_CURRENT_SOURCE_DIR}/include"      
    "${CMAKE_CURRENT_SOURCE_DIR}/include/argon2"
    "${llama_cpp_SOURCE_DIR}/include"
    "${llama_cpp_SOURCE_DIR}/ggml/include"
    "${whisper_cpp_SOURCE_DIR}/include"
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"        
)

pybind11_add_module(hegemonikon_py SHARED
    src/bindings.cc
)
target_link_libraries(hegemonikon_py PRIVATE
    hegemonikon     
    pybind11::module
)

if(WIN32)
    target_link_libraries(hegemonikon_py PRIVATE Psapi.lib)
else()
    target_link_libraries(hegemonikon_py PRIVATE pthread dl m)
endif()

include(CTest)

if (BUILD_TESTING)
    message(STATUS "CTest is enabled. Building unit tests...")

    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.6.0 
    )
    FetchContent_MakeAvailable(Catch2)

    add_executable(hegemonikon_tests
        tests/test_core_ai_service.cc
        tests/test_llama_integration.cc
        tests/test_whisper_integration.cc
    )
    
    if(NOT WIN32)
        target_sources(hegemonikon_tests PRIVATE tests/test_memory_locker.cc)
    endif()

    set(DOWNLOADED_LLAMA_MODEL_PATH "${CMAKE_SOURCE_DIR}/test_assets/tinyllama.gguf")
    set(DOWNLOADED_WHISPER_MODEL_PATH "${CMAKE_SOURCE_DIR}/test_assets/ggml-tiny.en.bin")

    target_compile_definitions(hegemonikon_tests PRIVATE 
        "TEST_LLAMA_MODEL_PATH=\"${DOWNLOADED_LLAMA_MODEL_PATH}\""
        "TEST_WHISPER_MODEL_PATH=\"${DOWNLOADED_WHISPER_MODEL_PATH}\""
    )

    target_include_directories(hegemonikon_tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${llama_cpp_SOURCE_DIR}/include"
        "${llama_cpp_SOURCE_DIR}/ggml/include"
        "${whisper_cpp_SOURCE_DIR}/include"
    )

    target_link_libraries(hegemonikon_tests PRIVATE 
        hegemonikon
        Catch2::Catch2
        Catch2::Catch2WithMain
    )
    
    if(NOT WIN32)
        target_link_libraries(hegemonikon_tests PRIVATE pthread dl m)
    endif()

    add_test(NAME HegemonikonTests COMMAND hegemonikon_tests)
endif()

if(ATARAXAI_USE_CUDA)
    message(STATUS "CUDA support is ENABLED for AtaraxAI.")
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "Found CUDA Toolkit Version: ${CUDAToolkit_VERSION}")
    
    target_link_libraries(hegemonikon_py PRIVATE
        CUDA::cudart
        CUDA::cuda_driver
        CUDA::cublas
        CUDA::cublasLt
    )
else()
    message(STATUS "CUDA support is DISABLED for AtaraxAI.")
endif()

target_include_directories(hegemonikon_py PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

install(TARGETS hegemonikon_py
        LIBRARY DESTINATION ataraxai
        RUNTIME DESTINATION ataraxai)