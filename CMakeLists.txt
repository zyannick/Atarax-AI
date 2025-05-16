cmake_minimum_required(VERSION 3.16)

project(Atarax-AI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(Atarax-AI main.cpp)

include(GNUInstallDirs)
install(TARGETS Atarax-AI
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
