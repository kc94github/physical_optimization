cmake_minimum_required(VERSION 3.14)

# set the project name
project(DrivePathSolver)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

enable_testing()

find_package(OsqpEigen REQUIRED)

add_subdirectory(googletest)

add_subdirectory(src/geometry)

add_subdirectory(src/solver)
add_subdirectory(test/geometry)
# add_subdirectory(test/solver)


# add the executable
#add_executable()