cmake_minimum_required(VERSION 3.14)

# set the project name
project(DrivePathSolver)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

enable_testing()

set(OsqpEigen_DIR "/root/anaconda/lib/cmake/OsqpEigen")
find_package(OsqpEigen REQUIRED)

add_subdirectory(googletest)

add_subdirectory(src/geometry)
add_subdirectory(src/solver)
add_subdirectory(src/utils)

add_subdirectory(test/geometry)
add_subdirectory(test/solver)

message(STATUS "CMAKE_PREFIX_PATH: ${OsqpEigen_DIR}")

# You can print multiple variables and messages
message(STATUS "Debugging information:")
message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
message(STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")


# add the executable
#add_executable()