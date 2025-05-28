file(GLOB_RECURSE ATARAXIA_SOURCES
    *.cc
    core_ai/*.cc
    prompt_engine/*.cc
    app_logic/*.cc
    common/*.cc
)

add_executable(ataraxia ${ATARAXIA_SOURCES})


target_include_directories(ataraxia PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/core_ai
    ${CMAKE_CURRENT_SOURCE_DIR}/prompt_engine
    ${CMAKE_CURRENT_SOURCE_DIR}/app_logic
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/whisper.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/ggml/include
)


target_link_directories(ataraxia PRIVATE
    ${CMAKE_BINARY_DIR}/third_party/llama.cpp
)

target_link_directories(ataraxia PRIVATE
    ${CMAKE_BINARY_DIR}/third_party/whisper.cpp
)




set_target_properties(ataraxia PROPERTIES
    INSTALL_RPATH "$ORIGIN/../third_party/llama.cpp"
)


install(TARGETS ataraxia RUNTIME DESTINATION bin)
