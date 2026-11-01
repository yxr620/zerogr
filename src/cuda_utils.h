#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace zerogr {

inline void Check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

inline void Check(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("cuBLAS error " + std::to_string(int(status)));
}

inline size_t Product(size_t a, size_t b) {
  if (a && b > std::numeric_limits<size_t>::max() / a)
    throw std::length_error("allocation size overflow");
  return a * b;
}

template <class T> class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t count) { Allocate(count); }
  ~DeviceBuffer() { if (data_) cudaFree(data_); }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  void Allocate(size_t count) {
    if (data_) throw std::logic_error("buffer already allocated");
    Check(cudaMalloc(reinterpret_cast<void **>(&data_), Product(count, sizeof(T))));
    count_ = count;
  }
  void Upload(const std::vector<T> &host) {
    if (host.size() != count_) throw std::logic_error("upload size mismatch");
    Check(cudaMemcpy(data_, host.data(), Product(count_, sizeof(T)), cudaMemcpyHostToDevice));
  }
  std::vector<T> Download() const {
    std::vector<T> host(count_);
    Check(cudaMemcpy(host.data(), data_, Product(count_, sizeof(T)), cudaMemcpyDeviceToHost));
    return host;
  }
  void Clear() { Check(cudaMemsetAsync(data_, 0, Product(count_, sizeof(T)))); }
  T *get() const { return data_; }

 private:
  T *data_ = nullptr;
  size_t count_ = 0;
};

class Event {
 public:
  Event() { Check(cudaEventCreate(&event_)); }
  ~Event() { cudaEventDestroy(event_); }
  Event(const Event &) = delete;
  Event &operator=(const Event &) = delete;
  void Record() { Check(cudaEventRecord(event_)); }
  void Wait() const { Check(cudaEventSynchronize(event_)); }
  float MicrosecondsSince(const Event &begin) const {
    float ms = 0;
    Check(cudaEventElapsedTime(&ms, begin.event_, event_));
    return ms * 1000.0f;
  }

 private:
  cudaEvent_t event_ = nullptr;
};

class Blas {
 public:
  Blas() {
    Check(cublasCreate(&handle_));
    const cublasStatus_t status = cublasSetMathMode(handle_, CUBLAS_PEDANTIC_MATH);
    if (status != CUBLAS_STATUS_SUCCESS) {
      cublasDestroy(handle_);
      Check(status);
    }
  }
  ~Blas() { cublasDestroy(handle_); }
  Blas(const Blas &) = delete;
  Blas &operator=(const Blas &) = delete;
  cublasHandle_t get() const { return handle_; }

 private:
  cublasHandle_t handle_ = nullptr;
};

}  // namespace zerogr
