add_library(solver_impl solver_impl.h solver_impl.cc)

target_include_directories(solver_impl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(solver_impl PUBLIC OsqpEigen::OsqpEigen)

add_library(spline_nd_solver spline_nd_solver.h spline_nd_solver.cc)

target_include_directories(spline_nd_solver PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(spline_nd_solver PUBLIC abstract solver_impl spline1d coefficient_base)

add_library(path_solver path_solver.h path_solver.cc)

target_include_directories(path_solver PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(path_solver PUBLIC abstract spline_nd_solver)

add_library(distance_time_solver distance_time_solver.h distance_time_solver.cc)

target_include_directories(distance_time_solver PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(distance_time_solver PUBLIC abstract spline_nd_solver)