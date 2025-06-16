#pragma once

#include <iostream>
#include <span>
#include<vector>
#include <stdexcept>


template<typename T> requires std::is_trivially_copyable_v<T> && std::is_trivially_default_constructible_v<T>
class RAIIPODArrayBuffer {

    public:
        template <std::ranges::input_range Range>
        requires std::convertible_to<std::ranges::range_value_t<Range>, T>
        explicit RAIIPODArrayBuffer(const Range& data, const unsigned int size) : buffer_ptr(nullptr), size(size) {
            //Copy input data
            this->copy_input_data(data);
            //Allocate memory.
            cudaMalloc((void**)&buffer_ptr, this->size * sizeof(T));
            std::cout << "Allocated memory for array with size:" << this->size << std::endl;
            //Copy data to GPU.
            this->upload_data(data);
        }

        RAIIPODArrayBuffer(const RAIIPODArrayBuffer& other) = delete;
        RAIIPODArrayBuffer& operator=(RAIIPODArrayBuffer& other) = delete;
        RAIIPODArrayBuffer(RAIIPODArrayBuffer&& other) noexcept : buffer_ptr(other.buffer_ptr), size(other.size) {
            other.buffer_ptr = nullptr;
            std::memcpy(data.data(), other.data.data(), this->size * sizeof(T));
        }

        RAIIPODArrayBuffer& operator=(RAIIPODArrayBuffer&& other) noexcept {
                if (this != other) {
                    if (this->buffer_ptr) {
                        cudaFree(this->buffer_ptr);
                    }
                    this->buffer_ptr = other.buffer_ptr;
                    other.buffer_ptr = nullptr;
                    std::memcpy(data.data(), other.data.data(), sizeof(T) * this->size);
                }

                return *this;
            }

        ~RAIIPODArrayBuffer() {
            if (this->buffer_ptr) {
                cudaFree(this->buffer_ptr);
                this->buffer_ptr = nullptr;
                std::cout << "Deallocating memory" << std::endl;
            }
        }

        [[nodiscard]] std::span<const T> view_data() noexcept {
            this->data.resize(this->size);
            cudaMemcpy(this->data.data(), this->buffer_ptr, this->size * sizeof(T), cudaMemcpyDeviceToHost);
            return std::span<const T>(this->data);
            }
        [[nodiscard]] std::span<const T> view_input_data() noexcept {
            return std::span<const T>(this->input_data);
        }

        [[nodiscard]] T* get_device_ptr() const noexcept {
            return this->buffer_ptr;
        }

    private:
        T* buffer_ptr;
        std::vector<T> input_data{};
        const unsigned int size;
        std::vector<T> data{};
        template <std::ranges::input_range Range>
        requires std::convertible_to<std::ranges::range_value_t<Range>, T>
        void copy_input_data(const Range& data) {
            // Copy to input data.
            for (const auto& val : data) {
                input_data.push_back(val);
            }
        }

        template <std::ranges::input_range Range>
        requires std::convertible_to<std::ranges::range_value_t<Range>, T>
        void upload_data(const Range &data) {
                // Fill the host buffer that will be sent to the GPU.
                std::vector<T> host_buffer{};
                for (const auto& val : data) {
                    host_buffer.emplace_back(static_cast<T>(val));
                }
                cudaMemcpy(buffer_ptr, host_buffer.data(), this->size * sizeof(T), cudaMemcpyHostToDevice);
                std::cout << "Uploaded data to the GPU" << std::endl;
            }
};