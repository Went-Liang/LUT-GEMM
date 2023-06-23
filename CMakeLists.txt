cmake_minimum_required(VERSION 3.24)
project(lut_gemm CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
find_package(CUDA)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

add_compile_options(-lineinfo)

include_directories(${EIGEN3_INCLUDE_DIRS})

add_executable(lut_gemm main.cu)

target_link_libraries(lut_gemm OpenMP::OpenMP_CXX ${CUDA_cublas_LIBRARY})

set_target_properties(lut_gemm PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

