#pragma once

#include "common.h"
#include "cuda_wrapper.h"

#include <cassert>
#include <utility>

namespace cuda {

template <typename T> struct raw_ptr {
private:
  T *ptr_       = nullptr;
  size_t count_ = 0;

public:
  CU_HD raw_ptr(T *ptr = nullptr, size_t count = 0) noexcept : ptr_(ptr), count_(count) {}
  CU_HD ~raw_ptr() = default;

  CU_HD void set(T *ptr, size_t count = 0) {
    ptr_   = ptr;
    count_ = count;
  }

  void allocateManaged(size_t count = 1, unsigned int flags = cudaMemAttachGlobal) {
    CUDA_CALL(cudaMallocManaged(reinterpret_cast<void **>(&ptr_), count * sizeof(T), flags));
    CUDA_CALL(cudaDeviceSynchronize());
    count_ = count;
    spdlog::debug("Allocated RAW CUDA memory of size {} at {}", count * sizeof(T), fmt::ptr(ptr_));
  }

  CU_HD explicit operator bool() const { return ptr_; }

  CU_HD T *operator->() const noexcept { return ptr_; }
  CU_HD T *get() const noexcept { return ptr_; }

  CU_HD T *begin() const { return &ptr_[0]; }
  CU_HD T *end() const { return &ptr_[count_]; }
  CU_HD T &operator[](size_t index) const {
    // assert(index < count_);
    return ptr_[index];
  }
  CU_HD T &operator*() const { return *ptr_; }

  CU_HD size_t size() const { return count_; }
  CU_HD size_t sizeBytes() const { return count_ * sizeof(T); }
};

template <typename T> class owning_ptr {
private:
  T *ptr_       = nullptr;
  size_t count_ = 0;

public:
  owning_ptr(T *ptr = nullptr, size_t count = 0) noexcept : ptr_(ptr), count_(count) {}
  ~owning_ptr() { clear(); }
  owning_ptr(const owning_ptr &) = delete;
  owning_ptr &operator=(const owning_ptr &) = delete;

  owning_ptr(owning_ptr &&ob) noexcept : ptr_(nullptr), count_(0) { swap(ob); }

  owning_ptr &operator=(owning_ptr &&ob) noexcept {
    swap(ob);
    return *this;
  }

  void allocateManaged(size_t count = 1, unsigned int flags = cudaMemAttachGlobal) {
    clear();
    CUDA_CALL(cudaMallocManaged(reinterpret_cast<void **>(&ptr_), count * sizeof(T), flags));
    CUDA_CALL(cudaDeviceSynchronize());
    count_ = count;
    spdlog::debug("Allocated CUDA memory of size {} at {}", count * sizeof(T), fmt::ptr(ptr_));
  }

  CU_HD explicit operator bool() const { return ptr_; }
  CU_HD operator raw_ptr<T>() const { return raw_ptr<T>(ptr_, count_); }
  CU_HD bool empty() const { return count_ == 0; }

  CU_HD T *begin() const { return &ptr_[0]; }
  CU_HD T *end() const { return &ptr_[count_]; }

  CU_HD T &operator[](size_t index) const {
    // assert(index < count_);
    return ptr_[index];
  }

  CU_HD T &operator*() const { return *ptr_; }

  CU_HD T *operator->() const noexcept { return ptr_; }

  CU_HD T *get() const noexcept { return ptr_; }

  T *release() noexcept {
    T *ans = ptr_;
    ptr_   = nullptr;
    count_ = 0;
    return ans;
  }

  void clear() {
    if (ptr_ && count_ > 0) {
      spdlog::debug("Freeing CUDA memory at {}, {}, {}", fmt::ptr(ptr_), count_, sizeBytes());
      CUDA_CALL(cudaDeviceSynchronize());
      CUDA_CALL(cudaFree(static_cast<void *>(ptr_)));
    }
    ptr_   = nullptr;
    count_ = 0;
  }

  void swap(owning_ptr &ob) noexcept {
    using std::swap;
    swap(ptr_, ob.ptr_);
    swap(count_, ob.count_);
  }

  CU_HD size_t size() const { return count_; }
  CU_HD size_t sizeBytes() const { return count_ * sizeof(T); }

  void loadIntoGPU(cudaStream_t stream) { CUDA_CALL(cudaStreamAttachMemAsync(stream, ptr_)); }
};

} // namespace cuda