################################
# Unit Tests
################################
# Add test cpp file
add_executable(test_coefficient_base test_coefficient_base.cc )
# Link test executable against gtest & gtest_main
target_link_libraries(
    test_coefficient_base PUBLIC
    coefficient_base
    gtest
    gtest_main
)

add_test(NAME test_coefficient_base COMMAND test_coefficient_base )