#include <vector>
#include <cstdio>
#include <type_traits>
#include <iostream>
#include "raii_pod_buffer.cu"
#include "raii_pod_array_buffer.cu"

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
    RAIIPODArrayBuffer<int, 5> buffer{vec};
    test_kernel<<<1,1>>>(buffer.get_device_ptr(), 5);
}

int main() {
    function();
    function_array();
}