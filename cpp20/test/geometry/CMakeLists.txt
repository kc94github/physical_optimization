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

add_executable(test_polynomial1d test_polynomial1d.cc )
target_link_libraries(
    test_polynomial1d PUBLIC
    coefficient_base
    polynomial1d
    gtest
    gtest_main
)

add_test(NAME test_polynomial1d COMMAND test_polynomial1d )

add_executable(test_polynomial2d test_polynomial2d.cc )
target_link_libraries(
    test_polynomial2d PUBLIC
    polynomial2d
    polynomial1d
    gtest
    gtest_main
)

add_test(NAME test_polynomial2d COMMAND test_polynomial2d )

add_executable(test_spline1d test_spline1d.cc )
target_link_libraries(
    test_spline1d PUBLIC
    polynomial1d
    spline1d
    gtest
    gtest_main
)

add_test(NAME test_spline1d COMMAND test_spline1d )