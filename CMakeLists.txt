cmake_minimum_required(VERSION 3.24)

project(amrexpr DESCRIPTION "AMReX's Math Expression Parser" VERSION 1.0
        LANGUAGES CXX)

if (CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
    message(FATAL_ERROR "\nIn-source builds are not allowed.\n")
endif()

set(CMAKE_CXX_STANDARD 17)

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
    set(CMAKE_CUDA_STANDARD 17)
endif()

option(ENABLE_HIP "Enable HIP support" OFF)
if (ENABLE_HIP)
    set(AMREXPR_USE_GPU 1)
    set(AMREXPR_USE_HIP 1)
    find_package(hip REQUIRED)
    enable_language(HIP)
    set(CMAKE_HIP_STANDARD 17)
endif()

option(ENABLE_SYCL "Enable SYCL support" OFF)
if (ENABLE_SYCL)
    set(AMREXPR_USE_GPU 1)
    set(AMREXPR_USE_SYCL 1)
    set(CMAKE_CXX_COMPILER icpx)
endif()

add_library(amrexpr)
add_subdirectory(Src)

configure_file(${PROJECT_SOURCE_DIR}/Tools/CMake/amrexpr_Config.H.in
               ${CMAKE_BINARY_DIR}/amrexpr_Config.H)

target_include_directories(amrexpr PUBLIC
                           $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/Src>
                           $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
                           $<INSTALL_INTERFACE:include>)

if (ENABLE_CUDA)
    get_target_property(_sources amrexpr SOURCES)
    list(FILTER _sources INCLUDE REGEX "\\.cpp$")
    set_source_files_properties(${_sources} PROPERTIES LANGUAGE CUDA)
    if (NOT DEFINED CUDA_ARCHITECTURES)
        message(STATUS "No CUDA architectures specified. Native will be used.")
        set_target_properties(amrexpr PROPERTIES CUDA_ARCHITECTURES native)
    endif()
    set_target_properties(amrexpr PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(amrexpr PRIVATE
                           $<$<COMPILE_LANGUAGE:CUDA>:-m64>
                           $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
                           $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
                           $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
    target_link_libraries(amrexpr PRIVATE CUDA::cudart)
endif()

if (ENABLE_SYCL)
    target_compile_options(amrexpr PRIVATE -fsycl)
endif()

install(TARGETS amrexpr EXPORT amrexprTargets)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/Src/ DESTINATION include
        FILES_MATCHING PATTERN "*.hpp" PATTERN "*.H")

install(FILES ${CMAKE_BINARY_DIR}/amrexpr_Config.H DESTINATION include)

set(PACKAGE_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
set(PACKAGE_INCLUDE_INSTALL_DIR "include")

set(amrexpr_FIND_DEPENDENCIES "")
if (ENABLE_CUDA)
    set(amrexpr_FIND_DEPENDENCIES "${amrexpr_FIND_DEPENDENCIES}\nfind_dependency(CUDAToolkit REQUIRED)")
endif()
if (ENABLE_HIP)
    set(amrexpr_FIND_DEPENDENCIES "${amrexpr_FIND_DEPENDENCIES}\nfind_dependency(hip REQUIRED)")
endif()
if (ENABLE_SYCL)
    # Adjust SYCL dependency find commands if necessary
endif()

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/amrexprConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

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

# tests
# enable_testing
# add_subdirectory(Tests)

# cofig:
# AMREXPR_USE_SYCL, HIP, CUDA, GPU
# NDEBUG
