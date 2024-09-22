cmake_minimum_required(VERSION 3.24)

project(amrexpr DESCRIPTION "AMReX's Math Expression Parser" VERSION 1.0
        LANGUAGES CXX)

if (CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
    message(FATAL_ERROR "\nIn-source builds are not allowed.\n")
endif()

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_CONFIGURATION_TYPES "Release;Debug;MinSizeRel;RelWithDebInfo")
    set(CMAKE_BUILD_TYPE Release
        CACHE STRING
        "Choose the build type, e.g. Release, Debug, or RelWithDebInfo." FORCE)
endif()

option(ENABLE_CUDA "Enable CUDA support" OFF)
if (ENABLE_CUDA)
    set(AMREXPR_USE_GPU 1)
    set(AMREXPR_USE_CUDA 1)
    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)
endif()

option(ENABLE_HIP "Enable HIP support" OFF)
if (ENABLE_HIP)
    set(AMREXPR_USE_GPU 1)
    set(AMREXPR_USE_HIP 1)
    enable_language(HIP)
endif()

option(ENABLE_SYCL "Enable SYCL support" OFF)
if (ENABLE_SYCL)
    set(AMREXPR_USE_GPU 1)
    set(AMREXPR_USE_SYCL 1)
    set(CMAKE_CXX_COMPILER icpx)
endif()

add_library(amrexpr)
add_subdirectory(Src)

target_compile_features(amrexpr PUBLIC cxx_std_17)

configure_file(${PROJECT_SOURCE_DIR}/Tools/CMake/amrexpr_Config.H.in
               ${CMAKE_BINARY_DIR}/amrexpr_Config.H)

target_include_directories(amrexpr PUBLIC
                           $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/Src>
                           $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
                           $<INSTALL_INTERFACE:include>)

get_target_property(amrexpr_sources amrexpr SOURCES)
list(FILTER amrexpr_sources INCLUDE REGEX "\\.cpp$")

if (ENABLE_CUDA)
    set_source_files_properties(${amrexpr_sources} PROPERTIES LANGUAGE CUDA)
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        message(STATUS "No CUDA architectures specified. Native will be used.")
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    set_target_properties(amrexpr PROPERTIES
                          CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
                          CUDA_SEPARABLE_COMPILATION ON)
    target_compile_features(amrexpr PUBLIC cuda_std_17)
    target_compile_options(amrexpr PUBLIC
                           $<$<COMPILE_LANGUAGE:CUDA>:-m64>
                           $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
                           $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
                           $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
    target_link_libraries(amrexpr PUBLIC CUDA::cudart)
endif()

if (ENABLE_HIP)
    set_source_files_properties(${amrexpr_sources} PROPERTIES LANGUAGE HIP)
    if (NOT DEFINED CMAKE_HIP_ARCHITECTURES)
        message(STATUS "No HIP architectures specified. 'gfx90a' will be used.")
        set(CMAKE_HIP_ARCHITECTURES gfx90a)
    endif()
    set_target_properties(amrexpr PROPERTIES HIP_ARCHITECTURES ${CMAKE_HIP_ARCHITECTURES})
    target_compile_features(amrexpr PUBLIC hip_std_17)
endif()

if (ENABLE_SYCL)
    target_compile_options(amrexpr PUBLIC -fsycl)
    target_link_options(amrexpr PUBLIC -fsycl -fsycl-device-lib=libc,libm-fp32,libm-fp64)
endif()

install(TARGETS amrexpr EXPORT amrexprTargets)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/Src/ DESTINATION include
        FILES_MATCHING PATTERN "*.hpp" PATTERN "*.H")

install(FILES ${CMAKE_BINARY_DIR}/amrexpr_Config.H DESTINATION include)

set(amrexpr_FIND_DEPENDENCIES "")
if (ENABLE_CUDA)
    set(amrexpr_FIND_DEPENDENCIES "${amrexpr_FIND_DEPENDENCIES}\nfind_dependency(CUDAToolkit REQUIRED)")
endif()

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/amrexprConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

set(PACKAGE_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
set(PACKAGE_INCLUDE_INSTALL_DIR "include")

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/Tools/CMake/amrexprConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/amrexprConfig.cmake"
    INSTALL_DESTINATION lib/cmake/amrexpr
    PATH_VARS PACKAGE_INCLUDE_INSTALL_DIR PACKAGE_INSTALL_PREFIX
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/amrexprConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/amrexprConfigVersion.cmake"
    DESTINATION lib/cmake/amrexpr
)

install(EXPORT amrexprTargets
        FILE amrexprTargets.cmake
        NAMESPACE amrexpr::
        DESTINATION lib/cmake/amrexpr
)

option(ENABLE_TESTS "Enable Tests" OFF)
if (ENABLE_TESTS)
    enable_testing()
    add_subdirectory(Tests)
endif()
