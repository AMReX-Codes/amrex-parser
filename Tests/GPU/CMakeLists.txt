add_executable(test_gpu main.cpp)

target_link_libraries(test_gpu PRIVATE amrexpr)

target_include_directories(test_gpu PRIVATE
    $<TARGET_PROPERTY:amrexpr,INTERFACE_INCLUDE_DIRECTORIES>
)

target_compile_features(test_gpu PRIVATE cxx_std_17)

if (ENABLE_CUDA)
    set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
    set_target_properties(test_gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_compile_options(test_gpu PRIVATE $<TARGET_PROPERTY:amrexpr>)
endif()

if (ENABLE_HIP)
    set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
    target_compile_options(test_gpu PRIVATE $<TARGET_PROPERTY:amrexpr>)
endif()

add_test(NAME test_gpu COMMAND test_gpu)
