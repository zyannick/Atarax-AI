file(GLOB_RECURSE ATARAXIA_SOURCES
    *.cc
    core_ai/*.cc
    prompt_engine/*.cc
    app_logic/*.cc
    common/*.cc
)

list(FILTER ATARAXIA_SOURCES EXCLUDE REGEX ".*/third_party/.*")

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

find_package(Boost  REQUIRED
    COMPONENTS
        process
        filesystem
        system
)

if(Boost_FOUND)
    message(STATUS "Boost version: ${Boost_VERSION_STRING}")
    message(STATUS "Boost include directories: ${Boost_INCLUDE_DIRS}")
    message(STATUS "Boost libraries: ${Boost_LIBRARIES}") 

    target_include_directories(ataraxia PRIVATE ${Boost_INCLUDE_DIRS})


    target_include_directories(ataraxia  PRIVATE ${Boost_INCLUDE_DIRS})

    target_link_libraries(ataraxia PRIVATE
        Boost::process
        Boost::filesystem
        Boost::system
    )

else()
    message(FATAL_ERROR "Boost not found. Please install Boost or set BOOST_ROOT.")
endif()



set_target_properties(ataraxia PROPERTIES
    INSTALL_RPATH "$ORIGIN/../third_party/llama.cpp"
)


install(TARGETS ataraxia RUNTIME DESTINATION bin)
