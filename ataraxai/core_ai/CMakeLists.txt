cmake_minimum_required(VERSION 3.18)
project(AtaraxAICoreAIModule LANGUAGES CXX C)

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


# find_package(Boost REQUIRED COMPONENTS process filesystem system)
# message(STATUS "Boost version: ${Boost_VERSION_STRING}")

get_filename_component(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../" ABSOLUTE)
set(THIRD_PARTY_INSTALL_DIR "${PROJECT_ROOT}/build/third_party_install")

set(LLAMA_INSTALL_PREFIX "${THIRD_PARTY_INSTALL_DIR}/llama")


find_library(GGML_BASE_LIBRARY_FILE 
              NAMES ggml-base      
              HINTS "${LLAMA_INSTALL_PREFIX}" 
              PATH_SUFFIXES lib lib64      
              NO_DEFAULT_PATH              
)

find_library(GGML_CPU_LIBRARY_FILE 
              NAMES ggml-cpu      
              HINTS "${LLAMA_INSTALL_PREFIX}" 
              PATH_SUFFIXES lib lib64      
              NO_DEFAULT_PATH              
)

find_library(GGML_GPU_LIBRARY_FILE 
              NAMES ggml-cuda      
              HINTS "${LLAMA_INSTALL_PREFIX}" 
              PATH_SUFFIXES lib lib64      
              NO_DEFAULT_PATH              
)

find_library(GGML_LIBRARY_FILE 
              NAMES ggml      
              HINTS "${LLAMA_INSTALL_PREFIX}" 
              PATH_SUFFIXES lib lib64      
              NO_DEFAULT_PATH              
)

find_library(LLAMA_LIBRARY_FILE 
              NAMES llama      
              HINTS "${LLAMA_INSTALL_PREFIX}" 
              PATH_SUFFIXES lib lib64      
              NO_DEFAULT_PATH              
)



if(NOT GGML_BASE_LIBRARY_FILE OR NOT GGML_CPU_LIBRARY_FILE OR NOT GGML_LIBRARY_FILE OR NOT LLAMA_LIBRARY_FILE)
    message(FATAL_ERROR "One or more required libraries not found under ${LLAMA_INSTALL_PREFIX}/lib64/")
else()
    message(STATUS "Found libraries:")
    message(STATUS "  libggml-base: ${GGML_BASE_LIBRARY_FILE}")
    message(STATUS "  libggml-cpu: ${GGML_CPU_LIBRARY_FILE}")
    message(STATUS "  libggml: ${GGML_LIBRARY_FILE}")
    message(STATUS "  libllama: ${LLAMA_LIBRARY_FILE}")
    # message(STATUS "  libggml-gpu: ${GGML_GPU_LIBRARY_FILE}")
endif()


set(WHISPER_INSTALL_PREFIX "${THIRD_PARTY_INSTALL_DIR}/whisper")
find_library(WHISPER_LIBRARY_FILE
              NAMES whisper
              HINTS "${WHISPER_INSTALL_PREFIX}"
              PATH_SUFFIXES lib lib64
              NO_DEFAULT_PATH
)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP: ${OpenMP_CXX_VERSION}")
else()
    message(FATAL_ERROR "OpenMP not found")
endif()

if(NOT WHISPER_LIBRARY_FILE)
    message(FATAL_ERROR "libwhisper.a not found under ${WHISPER_INSTALL_PREFIX} in either lib/ or lib64/. Did setup_third_party.sh run correctly and install it?")
else()
    message(STATUS "Found libwhisper at: ${WHISPER_LIBRARY_FILE}")
endif()

set(LLAMA_INCLUDE_DIR   "${LLAMA_INSTALL_PREFIX}/include")
set(WHISPER_INCLUDE_DIR "${WHISPER_INSTALL_PREFIX}/include")

add_library(core_ai_logic STATIC
    src/core_ai_service.cc
    src/llama_interface.cc
    src/whisper_interface.cc
)

set_target_properties(core_ai_logic PROPERTIES POSITION_INDEPENDENT_CODE ON)


target_link_libraries(core_ai_logic PRIVATE
    "${LLAMA_LIBRARY_FILE}"
    "${GGML_LIBRARY_FILE}"
    "${GGML_BASE_LIBRARY_FILE}"
    "${GGML_CPU_LIBRARY_FILE}"
    "${WHISPER_LIBRARY_FILE}" 
    OpenMP::OpenMP_CXX 
     pthread
    dl
    m
)

target_include_directories(core_ai_logic PUBLIC 
    "${CMAKE_CURRENT_SOURCE_DIR}/include"      
    ${LLAMA_INCLUDE_DIR}            
    "${LLAMA_INCLUDE_DIR}/ggml/include"
    ${Boost_INCLUDE_DIRS}
    ${WHISPER_INCLUDE_DIR}
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"        
)


pybind11_add_module(core_ai_py SHARED
    src/bindings.cc
)
target_link_libraries(core_ai_py PRIVATE
    core_ai_logic     
    pybind11::module   
    pthread
    dl
    m
)


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

    add_executable(core_ai_tests
        tests/test_core_ai_service.cc
        tests/test_llama_integration.cc
        tests/test_whisper_integration.cc
    )

    set(DOWNLOADED_LLAMA_MODEL_PATH "${CMAKE_SOURCE_DIR}/test_assets/tinyllama.gguf")
    set(DOWNLOADED_WHISPER_MODEL_PATH "${CMAKE_SOURCE_DIR}/test_assets/ggml-tiny.en.bin")

    target_compile_definitions(core_ai_tests PRIVATE 
        "TEST_LLAMA_MODEL_PATH=\"${DOWNLOADED_LLAMA_MODEL_PATH}\""
        "TEST_WHISPER_MODEL_PATH=\"${DOWNLOADED_WHISPER_MODEL_PATH}\""
    )


    target_include_directories(core_ai_tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        ${LLAMA_INCLUDE_DIR}
        "${LLAMA_INCLUDE_DIR}/ggml/include"
        ${WHISPER_INCLUDE_DIR}
    )

    target_link_libraries(core_ai_tests PRIVATE 
        core_ai_logic
        Catch2::Catch2
        Catch2::Catch2WithMain
        pthread
        dl
        m
    )

    add_test(NAME CoreAILogicTests COMMAND core_ai_tests)

endif()

if(ATARAXAI_USE_CUDA)
    message(STATUS "CUDA support is ENABLED for AtaraxAI.")
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "Found CUDA Toolkit Version: ${CUDAToolkit_VERSION}")

    target_link_libraries(core_ai_py PRIVATE
        "${GGML_GPU_LIBRARY_FILE}"
        CUDA::cudart
        CUDA::cuda_driver
        CUDA::cublas
        CUDA::cublasLt
    )
else()
    message(STATUS "CUDA support is DISABLED for AtaraxAI.")
endif()

target_include_directories(core_ai_py PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")


install(TARGETS core_ai_py
        LIBRARY DESTINATION ataraxai
        RUNTIME DESTINATION ataraxai)
