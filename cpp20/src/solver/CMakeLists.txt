add_executable(qp_test qp_test.cc)

target_include_directories(qp_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(qp_test OsqpEigen::OsqpEigen)

add_library(solver_impl solver_impl.h solver_impl.cc)

target_include_directories(solver_impl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(solver_impl OsqpEigen::OsqpEigen)