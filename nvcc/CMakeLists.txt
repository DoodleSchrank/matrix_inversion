cmake_minimum_required(VERSION 3.20)

project

enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES "61")
add_executable(MatrixInversionNVCC inversion.cu)