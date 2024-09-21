cmake_minimum_required(VERSION 3.24)

project(amrexpr DESCRIPTION "AMReX's Math Expression Parser" VERSION 1.0
        LANGUAGES CXX)

if (CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
    message(FATAL_ERROR "\nIn-source builds are not allowed.\n")
endif()

set(CMAKE_CXX_STANDARD 17)

add_library(amrexpr)

add_subdirectory(Src)

configure_file(${PROJECT_SOURCE_DIR}/Tools/CMake/amrexpr_Config.H.in
               ${CMAKE_BINARY_DIR}/amrexpr_Config.H)

include_directories(${PROJECT_SOURCE_DIR}/Src ${CMAKE_BINARY_DIR})

install(TARGETS amrexpr)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/Src/ DESTINATION include
        FILES_MATCHING PATTERN "*.hpp" PATTERN "*.H"
)

install(FILES ${CMAKE_BINARY_DIR}/amrexpr_Config.H DESTINATION include)

# tests
# enable_testing
# add_subdirectory(Tests)

# cofig:
# AMREXPR_USE_SYCL, HIP, CUDA, GPU
# NDEBUG
