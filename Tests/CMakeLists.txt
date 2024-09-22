
if (ENABLE_CUDA OR ENABLE_HIP OR ENABLE_SYCL)
    add_subdirectory(GPU)
else()
    add_subdirectory(Parser)
    add_subdirectory(Parser2)
endif()
