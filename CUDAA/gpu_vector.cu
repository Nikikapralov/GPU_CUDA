#pragma once

#include <vector>
#include <span>
#include <utility>
#include <iostream>
#include <stdexcept>
#include "raii_pod_array_buffer.cu"

template<typename T> requires std::is_trivially_copyable_v<T> && std::is_trivially_default_constructible_v<T>
__global__ void gpu_addition(T vec_1[], T vec_2[], const unsigned int size) {
    const unsigned int total_threads = gridDim.x * blockDim.x; // How many blocks x how many threads.
    unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // Block * how many threads + curr thread.
    while (thread_id < size) {
        vec_1[thread_id] = vec_1[thread_id] + vec_2[thread_id];
        thread_id += total_threads;
    }
}

template <typename T, int Blocks, int Threads> requires std::is_trivially_copyable_v<T> && std::is_trivially_default_constructible_v<T>
class ParallelAwareVector {
public:
    template <typename... Args> // Take any amount of types (multiple allowed)
    explicit ParallelAwareVector(Args&&... args) : vector_data(std::vector<T>()) { // Take any amount of args.
        // Perfect forward (lvalue as lvalue, rvalue as rvalue) to emplace_back;
        (this->vector_data.emplace_back(std::forward<Args>(args)), ...); // Fold operator.
    }
    ParallelAwareVector(const ParallelAwareVector& other) = default;
    ParallelAwareVector& operator=(const ParallelAwareVector& other) = default;
    ParallelAwareVector(ParallelAwareVector&& other) noexcept = default;
    ParallelAwareVector& operator=(ParallelAwareVector&& other) noexcept = default;
    ~ParallelAwareVector() = default;

   [[nodiscard]] std::span<const T> get_vector() const noexcept {
        return std::span<const T>(this->vector_data);
    }

    ParallelAwareVector<T, Blocks, Threads>& operator+=(const ParallelAwareVector<T, Blocks, Threads>& rhs) {
       // Const T2& (reference) can bind to both L and R values. T2& - only L, T2&& - only R. No copies are made.
       // Copied on copy or move, but can't do it to a const. If I need to do it, I need T2&& or T2& and perfect
       // forwarding with std::forward. Currently not needed.
       std::cout << "Attempting to add vectors." << std::endl;

       if (this->vector_data.size() != rhs.get_vector().size()) {
           std::cout << "Vector sizes are not equal! Abort." << std::endl;
           throw std::runtime_error("Error");
       }

           if (this->set_device()) {
               //this->raw(rhs);
               this->RAII(rhs);
               return *this;
           }
       // Calculate with CPU -> Currently not implemented.
       std::cout << "CPU path chosen." << std::endl;
       return *this;
       }

private:
    std::vector<T> vector_data;

    bool set_device() {
        std::cout << "Looking for GPU." << std::endl;
        int devices{0};
        cudaGetDeviceCount(&devices);
        if (devices > 0) {
            cudaSetDevice(0); // Use first CUDA capable device, simple solution.
            return true;
        }
        return false;
    }

    void raw(const auto &rhs) {
        // Calculate with CUDA.
        std::cout << "GPU path chosen." << std::endl;
        int* vec_1{nullptr};
        int* vec_2{nullptr};
        //memccpy(vec_1, this->vector_data.data(), this->vector_data.size() * sizeof(T));
        //memccpy(vec_2, rhs.get_vector().data(), rhs.get_vector().size() * sizeof(T));

        // Allocate memory.
        cudaMalloc((void**)&vec_1, this->vector_data.size() * sizeof(T));
        cudaMalloc((void**)&vec_2, rhs.get_vector().size() * sizeof(T));

        // Copy data to the GPU.
        cudaMemcpy(vec_1, this->vector_data.data(), this->vector_data.size() * sizeof(T),
            cudaMemcpyHostToDevice);
        cudaMemcpy(vec_2, rhs.get_vector().data(), rhs.get_vector().size() * sizeof(T),
            cudaMemcpyHostToDevice);

        // Call kernel.

        gpu_addition<<<Blocks, Threads>>>(vec_1, vec_2, this->vector_data.size());
        cudaDeviceSynchronize();

        // Copy data from GPU to CPU.
        cudaMemcpy(this->vector_data.data(), vec_1, this->vector_data.size() * sizeof(T),
            cudaMemcpyDeviceToHost);

        cudaFree(vec_1);
        cudaFree(vec_2);
    }

    void RAII(const ParallelAwareVector<T, Blocks, Threads>& rhs) {
        const unsigned int vec_size = this->get_vector().size();
        RAIIPODArrayBuffer<T> vec_1{this->vector_data, vec_size};
        RAIIPODArrayBuffer<T> vec_2{rhs.get_vector(), vec_size};

        gpu_addition<<<Blocks, Threads>>>(vec_1.get_device_ptr(), vec_2.get_device_ptr(), vec_size);
        cudaDeviceSynchronize();

        std::cout << "Finished addition" << std::endl;

        memcpy(this->vector_data.data(), vec_1.view_data().data(), vec_1.view_data().size() * sizeof(T));
    }

};
