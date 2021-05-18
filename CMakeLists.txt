cmake_minimum_required(VERSION 3.19)
project(project)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_STANDARD 20)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(project main.cpp)
target_link_libraries (project Eigen3::Eigen)
