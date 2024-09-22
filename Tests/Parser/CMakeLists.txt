add_executable(test_parser main.cpp)

target_link_libraries(test_parser PRIVATE amrexpr)

target_include_directories(test_parser PRIVATE
    $<TARGET_PROPERTY:amrexpr,INTERFACE_INCLUDE_DIRECTORIES>
)

target_compile_features(test_parser PRIVATE cxx_std_17)

add_test(NAME test_parser COMMAND test_parser)
