cmake_minimum_required(VERSION 3.30)
project(CUDAA CUDA)

set(CMAKE_CUDA_STANDARD 26)

add_executable(CUDAA main.cu
        raii_pod_array_buffer.cu
        gpu_vector.cu)

set_target_properties(CUDAA PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)
