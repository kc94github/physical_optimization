################################
# Unit Tests
################################
# Add test cpp file
add_executable(test_solver_impl test_solver_impl.cc )
# Link test executable against gtest & gtest_main
target_link_libraries(
    test_solver_impl PUBLIC
    solver_impl
    gtest
    gtest_main
)

add_test(NAME test_solver_impl COMMAND test_solver_impl)