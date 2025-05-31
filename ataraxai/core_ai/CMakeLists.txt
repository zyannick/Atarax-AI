cmake_minimum_required(VERSION 3.15)
project(AtaraxAICoreAIModule LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.12.0 
)
FetchContent_MakeAvailable(pybind11)

find_package(Boost REQUIRED COMPONENTS process filesystem system)
message(STATUS "Boost version: ${Boost_VERSION_STRING}")

get_filename_component(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../" ABSOLUTE)
set(THIRD_PARTY_INSTALL_DIR "${PROJECT_ROOT}/build/third_party_install")

set(LLAMA_INSTALL_PREFIX "${THIRD_PARTY_INSTALL_DIR}/llama")
find_library(LLAMA_LIBRARY_FILE 
             NAMES llama       
             HINTS "${LLAMA_INSTALL_PREFIX}" 
             PATH_SUFFIXES lib lib64       
             NO_DEFAULT_PATH                
)

if(NOT LLAMA_LIBRARY_FILE)
    message(FATAL_ERROR "libllama.a not found under ${LLAMA_INSTALL_PREFIX} in either lib/ or lib64/. Did setup_third_party.sh run correctly and install it?")
else()
    message(STATUS "Found libllama at: ${LLAMA_LIBRARY_FILE}")
endif()

set(WHISPER_INSTALL_PREFIX "${THIRD_PARTY_INSTALL_DIR}/whisper")
find_library(WHISPER_LIBRARY_FILE
             NAMES whisper
             HINTS "${WHISPER_INSTALL_PREFIX}"
             PATH_SUFFIXES lib lib64
             NO_DEFAULT_PATH
)

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
)
target_link_libraries(core_ai_logic PRIVATE
    "${LLAMA_LIBRARY_FILE}"   
    "${WHISPER_LIBRARY_FILE}" 
    Boost::process Boost::filesystem Boost::system
)
target_include_directories(core_ai_logic PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    ${Boost_INCLUDE_DIRS}
    ${LLAMA_INCLUDE_DIR}
    "${LLAMA_INCLUDE_DIR}/ggml/include" 
    ${WHISPER_INCLUDE_DIR}
)


pybind11_add_module(core_ai_py SHARED
    src/bindings.cc
)
target_link_libraries(core_ai_py PRIVATE
    core_ai_logic     
    pybind11::module   
)

target_include_directories(core_ai_py PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")


install(TARGETS core_ai_py
        LIBRARY DESTINATION ataraxai
        RUNTIME DESTINATION ataraxai)
