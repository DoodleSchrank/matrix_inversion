cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_COMPILER "/home/yannik/programs/spack/opt/spack/linux-archrolling-skylake/gcc-11.1.0/gcc-11.2.0-sacutezrgifekoo3z5tmzaps6twen6jm/bin/g++")
set(CMAKE_CUDA_COMPILER "/usr/bin/nvc++")
project(MatrixInversion LANGUAGES CXX CUDA)


include(CheckLanguage)
check_language(CUDA)
find_package(CUDA)
message(${CUDA_TOOLKIT_INCLUDE})
include_directories(${CUDA_TOOLKIT_INCLUDE})
find_package(OpenMP)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "-O2")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_subdirectory(gcc)
add_subdirectory(nvcc)
set(CMAKE_CUDA_FLAGS "-O2 -Wall -Wextra -acc -fopenmp")


target_link_libraries (MatrixInversionGCC Eigen3::Eigen)
