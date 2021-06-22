cmake_minimum_required(VERSION 3.19)

set(CMAKE_CXX_COMPILER "/home/yannik/programs/spack/opt/spack/linux-arch-skylake/gcc-10.2.0/gcc-10.2.0-plfrsmdlee3htzk7cecy2lnvkmhwcfxt/bin/g++")
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
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2 -Wall -Wextra -acc")


target_link_libraries (MatrixInversionGCC Eigen3::Eigen)
