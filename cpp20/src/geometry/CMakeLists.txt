add_library(coefficient_base coefficient_base.h coefficient_base.cc)

target_include_directories(coefficient_base PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(coefficient_base PUBLIC OsqpEigen::OsqpEigen abstract)

add_library(polynomial1d polynomial1d.h polynomial1d.cc)

target_include_directories(polynomial1d PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(polynomial1d PUBLIC coefficient_base)

add_library(polynomial2d polynomial2d.h polynomial2d.cc)

target_include_directories(polynomial2d PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(polynomial2d PUBLIC abstract polynomial1d)