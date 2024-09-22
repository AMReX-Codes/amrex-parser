cmake_minimum_required(VERSION 3.24)

option(ENABLE_CUDA "Enable CUDA support" OFF)
option(ENABLE_HIP "Enable HIP support" OFF)
option(ENABLE_SYCL "Enable SYCL support" OFF)

if (ENABLE_CUDA)
    set(PROJECT_LANGUAGES CXX CUDA)
elseif (ENABLE_HIP)
    set(PROJECT_LANGUAGES CXX HIP)
else()
    set(PROJECT_LANGUAGES CXX)
endif()

project(amrexpr_test LANGUAGES ${PROJECT_LANGUAGES})

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_CONFIGURATION_TYPES "Release;Debug;MinSizeRel;RelWithDebInfo")
    set(CMAKE_BUILD_TYPE Release
        CACHE STRING
        "Choose the build type, e.g. Release, Debug, or RelWithDebInfo." FORCE)
endif()

if (ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

if (ENABLE_HIP)
    find_package(hip REQUIRED)
endif()

if (ENABLE_SYCL)
    set(CMAKE_CXX_COMPILER icpx)
endif()

find_package(amrexpr REQUIRED)

add_executable(parser_test main.cpp)

if (ENABLE_CUDA)
    set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
    if (NOT DEFINED CUDA_ARCHITECTURES)
        message(STATUS "No CUDA architectures specified. Native will be used.")
        set_target_properties(parser_test PROPERTIES CUDA_ARCHITECTURES native)
    endif()
    set_target_properties(parser_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

target_link_libraries(parser_test PRIVATE amrexpr::amrexpr)
