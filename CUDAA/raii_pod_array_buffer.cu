#include <iostream>
#include <span>
#include<vector>
#include <stdexcept>


template<typename T, int N> requires std::is_trivially_copyable_v<T> && std::is_trivially_default_constructible_v<T>
class RAIIPODArrayBuffer {

    public:
        template <std::ranges::input_range Range>
        requires std::convertible_to<std::ranges::range_value_t<Range>, T>
        explicit RAIIPODArrayBuffer(const Range& data) : buffer_ptr(nullptr), size(std::ranges::distance(data)) {
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
        RAIIPODArrayBuffer(RAIIPODArrayBuffer&& other) noexcept : buffer_ptr(other.buffer_ptr) {
            other.buffer_ptr = nullptr;
            std::memcpy(data_array, other.data_array, sizeof(T) * N);
        }

        RAIIPODArrayBuffer& operator=(RAIIPODArrayBuffer&& other) noexcept {
                if (this != other) {
                    if (this->buffer_ptr) {
                        cudaFree(this->buffer_ptr);
                    }
                    this->buffer_ptr = other.buffer_ptr;
                    other.buffer_ptr = nullptr;
                    std::memcpy(data_array, other.data_array, sizeof(T) * N);
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

        [[nodiscard]] std::span<const T, N> view_data() noexcept {
                cudaMemcpy(this->data_array, this->buffer_ptr, N * sizeof(T));
                return std::span<const T, N>(this->data_array);
            }
        [[nodiscard]] std::span<const T,N> view_input_data() noexcept {
            return std::span<const T>(this->input_data);
        }

        [[nodiscard]] T* get_device_ptr() const noexcept {
            return this->buffer_ptr;
        }

    private:
        T* buffer_ptr;
        std::vector<T> input_data{};
        T data_array[N]{};
        int size{0};

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
                // Upload data from a range supported container to the GPU.
                // Check if the container is not too big.
                if (this->size > N) {
                    throw(std::runtime_error("Size of data is bigger than buffer size."));
                }
                // Fill the host buffer that will be sent to the GPU.
                T host_buffer[N]{};
                int idx{0};
                for (const auto& val : data) {
                    host_buffer[idx] = static_cast<T>(val);
                    ++idx;
                }
                cudaMemcpy(buffer_ptr, host_buffer, this->size * sizeof(T), cudaMemcpyHostToDevice);
                std::cout << "Uploaded data to the GPU" << std::endl;
            }
};