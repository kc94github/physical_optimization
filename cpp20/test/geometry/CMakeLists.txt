# find_package(GOOGLETEST REQUIRED)
# project(geometry_test)

################################
# GTest
################################

# include_directories(${GTEST_INCLUDE_DIRS})


# include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

################################
# Unit Tests
################################
# Add test cpp file
add_executable( test_coefficient_base test_coefficient_base.cc )
# Link test executable against gtest & gtest_main
target_link_libraries(
    test_coefficient_base PUBLIC
    coefficient_base
    gtest
    gtest_main
)

# enable_testing()
add_test(NAME test_coefficient_base COMMAND test_coefficient_base )