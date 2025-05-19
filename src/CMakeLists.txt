cmake_minimum_required(VERSION 3.10)
project(core_ai)

set(CMAKE_CXX_STANDARD 17)

include_directories(${CMAKE_SOURCE_DIR})

add_executable(core_ai main.cc)
