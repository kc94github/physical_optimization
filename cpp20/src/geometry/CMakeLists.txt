add_library(coefficient_base coefficient_base.h coefficient_base.cc)

target_include_directories(coefficient_base PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(coefficient_base PUBLIC OsqpEigen::OsqpEigen abstract)

add_library(polynomial1d polynomial1d.h polynomial1d.cc)

target_include_directories(polynomial1d PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(polynomial1d PUBLIC coefficient_base)

add_library(polynomial2d polynomial2d.h polynomial2d.cc)

target_include_directories(polynomial2d PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(polynomial2d PUBLIC abstract polynomial1d)

add_library(spline1d spline1d.h spline1d.cc)

target_include_directories(spline1d PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(spline1d PUBLIC abstract polynomial1d)

add_library(spline2d spline2d.h spline2d.cc)

target_include_directories(spline2d PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(spline2d PUBLIC abstract polynomial1d spline1d)