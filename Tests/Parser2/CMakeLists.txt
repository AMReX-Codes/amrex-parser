add_executable(test_parser2 main.cpp fn.cpp)

set(TEST_PARSER2_WORK_DIRECTORY $<TARGET_FILE_DIR:test_parser2>)
add_custom_command(TARGET test_parser2 PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/fn.cpp
    ${TEST_PARSER2_WORK_DIRECTORY}/fn.cpp
)

target_link_libraries(test_parser2 PRIVATE amrexpr)

target_include_directories(test_parser2 PRIVATE
    $<TARGET_PROPERTY:amrexpr,INTERFACE_INCLUDE_DIRECTORIES>
)

target_compile_features(test_parser2 PRIVATE cxx_std_17)

add_test(NAME test_parser2 COMMAND test_parser2)
