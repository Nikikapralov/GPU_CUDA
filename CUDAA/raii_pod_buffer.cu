#include <iostream>

template <typename T> requires std::is_trivially_copyable_v<T> && std::is_trivially_default_constructible_v<T>
class RAIIPODBuffer {

    public:
        explicit RAIIPODBuffer () : buffer_ptr(nullptr) {
            cudaMalloc((void**)&buffer_ptr, sizeof(T));
            std::cout << "Allocated memory" << std::endl;
        };

        ~RAIIPODBuffer() {
            if (this->buffer_ptr) {
                cudaFree(this->buffer_ptr); // Free the memory.
                this->buffer_ptr = nullptr; // Nullify the to show memory is freed.
                std::cout << "Deallocated memory" << std::endl;
                }
            };

        RAIIPODBuffer(const RAIIPODBuffer& other) = delete;
        RAIIPODBuffer& operator=(const RAIIPODBuffer& other) = delete;
        RAIIPODBuffer(RAIIPODBuffer&& other) noexcept : buffer_ptr(other.buffer_ptr),
        data(std::move(other.data)){
            // No exceptions will be thrown so std knows to use move semantics.
            // Otherwise it will fallback to copy -> slow.
            // No need to check if == other since the move constructor will always create a new object.
            other.buffer_ptr = nullptr;
        };
        RAIIPODBuffer& operator=(RAIIPODBuffer&& other) noexcept {
            // Move assignment assigns an already existing object, the data of another one.
            // As such, we need to free the CUDA memory of already existing one and copy the
            // new pointer to the old data.
            if (this != &other) {
                if (this->buffer_ptr) {
                    cudaFree(this->buffer_ptr);
                }
                this->buffer_ptr = other.buffer_ptr;
                other.buffer_ptr = nullptr;
                this->data = std::move(other.data);
            }
            return *this;
        }

        [[nodiscard]] T* get_device_ptr() const noexcept{
            return this->buffer_ptr;
        }

        [[nodiscard]] const T& view_data() {
            cudaError_t err = cudaMemcpy(&this->data, this->buffer_ptr, sizeof(T), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                // handle error
                std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
            }
            return this->data;
        }

    private:
        T* buffer_ptr;
        T data{};
};
