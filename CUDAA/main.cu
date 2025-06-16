#include <vector>
#include <cstdio>
#include <type_traits>
#include <iostream>
#include "raii_pod_buffer.cu"
#include "raii_pod_array_buffer.cu"
#include "gpu_vector.cu"

std::vector<int> vec_1{1, 2, 3, 4, 5, 6};
std::vector<int> vec_2{2, 4, 6, 8, 10, 12};

__global__ void test_func(char a, char b, char* const c) {
    printf("From device: %c\n", a);
    *c = a;
}

__global__ void test_kernel(int a[], int size) {
    for (int i{0}; i < size; ++i) {
        printf("From device: %d\n", a[i]);
    }
}

void function() {
    RAIIPODBuffer<char> c{};
    test_func<<<1,1>>>('a', 'b', c.get_device_ptr());
    cudaDeviceSynchronize();
    char result = c.view_data();
    std::cout << "From host: " << result << std::endl;
}

void function_array() {
    std::vector<int> vec{1, 2, 3, 4, 5};
    RAIIPODArrayBuffer<int> buffer{vec, 5};
    test_kernel<<<1,1>>>(buffer.get_device_ptr(), 5);
    cudaDeviceSynchronize();
    auto data = buffer.view_data();
    for (const auto& val : data) {
        std::cout << "From host:" << val << std::endl;
    }
}

void function_parallel_aware_vector() {
    ParallelAwareVector<int, (9 + 2 - 1) / 2, 2> vec_1{1, 2, 3, 4, 5, 6, 7, 8, 9};
    ParallelAwareVector<int, (9 + 2 - 1) / 2, 2> vec_2{9, 8, 7, 6, 5, 4, 3, 2, 1};

    vec_1 += vec_2;

    for (const auto& val : vec_1.get_vector()) {
        std::cout << "From host:" << val << std::endl;
    }
}

int main() {
    function();
    std::cout << std::endl;
    function_array();
    std::cout << std::endl;
    function_parallel_aware_vector();
}