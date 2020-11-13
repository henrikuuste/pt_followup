#pragma once

#ifdef _MSC_VER
#pragma warning(push, 0)
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

namespace cuda {

#define CU_HD __host__ __device__
#define CU_D __device__
#define CU_H __host__

#ifdef __DRIVER_TYPES_H__
static const char *errorName(cudaError_t error) { return cudaGetErrorName(error); }
#endif

#ifdef CUDA_DRIVER_API
// CUDA Driver API errors
static const char *errorName(CUresult error) {
  static char unknown[] = "<unknown>";
  const char *ret       = NULL;
  cuGetErrorName(error, &ret);
  return ret ? ret : unknown;
}
#endif

#ifdef __DRIVER_TYPES_H__
#ifndef DEVICE_RESET
#define DEVICE_RESET cudaDeviceReset();
#endif
#else
#ifndef DEVICE_RESET
#define DEVICE_RESET
#endif
#endif

template <typename T>
void check(T result, char const *const func, const char *const file, int const line) {
  if (result) {
    spdlog::error("CUDA error at {}:{} code={}({}) \"{}\"", file, line,
                  static_cast<unsigned int>(result), errorName(result), func);
    DEVICE_RESET
    // Make sure we call CUDA Device Reset before exiting
    throw std::runtime_error(errorName(result));
  }
}

#ifdef __DRIVER_TYPES_H__
// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define CUDA_CALL(val) cuda::check((val), #val, __FILE__, __LINE__)
#endif

// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor) {
  // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
  typedef struct {
    int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {
      {0x30, 192}, {0x32, 192}, {0x35, 192}, {0x37, 192}, {0x50, 128}, {0x52, 128}, {0x53, 128},
      {0x60, 64},  {0x61, 128}, {0x62, 128}, {0x70, 64},  {0x72, 64},  {0x75, 64},  {-1, -1}};

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  // If we don't find the values, we default use the previous one to run properly
  spdlog::info("MapSMtoCores for SM {}.{} is undefined. Default to use {} Cores/SM", major, minor,
               nGpuArchCoresPerSM[index - 1].Cores);
  return nGpuArchCoresPerSM[index - 1].Cores;
}

#ifdef __CUDA_RUNTIME_H__
// General GPU Device CUDA Initialization
inline int gpuDeviceInit(int devID) {
  int device_count;
  CUDA_CALL(cudaGetDeviceCount(&device_count));

  if (device_count == 0) {
    spdlog::error("gpuDeviceInit() CUDA error: no devices supporting CUDA.");
    abort();
  }

  if (devID < 0) {
    devID = 0;
  }

  if (devID > device_count - 1) {
    spdlog::error(">> {} CUDA capable GPU device(s) detected. <<", device_count);
    spdlog::error(">> gpuDeviceInit (-device={}) is not a valid GPU device. <<", devID);
    return -devID;
  }

  cudaDeviceProp deviceProp;
  CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));

  if (deviceProp.computeMode == cudaComputeModeProhibited) {
    spdlog::error("Error: device is running in <Compute Mode Prohibited>, no threads can use "
                  "cudaSetDevice().");
    return -1;
  }

  if (deviceProp.major < 1) {
    spdlog::error("gpuDeviceInit(): GPU device does not support CUDA.");
    abort();
  }

  CUDA_CALL(cudaSetDevice(devID));
  spdlog::info("gpuDeviceInit() CUDA Device [{}]: \"{}\"", devID, deviceProp.name);

  return devID;
}

// This function returns the best GPU (with maximum GFLOPS)
inline int gpuGetMaxGflopsDeviceId() {
  int current_device = 0, sm_per_multiproc = 0;
  int max_perf_device    = 0;
  int device_count       = 0;
  int devices_prohibited = 0;

  uint64_t max_compute_perf = 0;
  cudaDeviceProp deviceProp;
  CUDA_CALL(cudaGetDeviceCount(&device_count));

  if (device_count == 0) {
    spdlog::error("gpuGetMaxGflopsDeviceId() CUDA error: no devices supporting CUDA.");
    abort();
  }

  // Find the best CUDA capable GPU device
  current_device = 0;

  while (current_device < device_count) {
    cudaGetDeviceProperties(&deviceProp, current_device);

    // If this GPU is not running on Compute Mode prohibited,
    // then we can add it to the list
    if (deviceProp.computeMode != cudaComputeModeProhibited) {
      if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
        sm_per_multiproc = 1;
      } else {
        sm_per_multiproc = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
      }

      uint64_t compute_perf = static_cast<uint64_t>(deviceProp.multiProcessorCount) *
                              sm_per_multiproc * deviceProp.clockRate;

      if (compute_perf > max_compute_perf) {
        max_compute_perf = compute_perf;
        max_perf_device  = current_device;
      }
    } else {
      devices_prohibited++;
    }

    ++current_device;
  }

  if (devices_prohibited == device_count) {
    spdlog::error(
        "gpuGetMaxGflopsDeviceId() CUDA error: all devices have compute mode prohibited.");
    abort();
  }

  return max_perf_device;
}

// Initialization code to find the best CUDA Device
inline int findBestDevice() {
  cudaDeviceProp deviceProp;
  int devID = 0;

  devID = gpuGetMaxGflopsDeviceId();
  CUDA_CALL(cudaSetDevice(devID));
  CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
  spdlog::info("GPU Device {}: \"{}\" with compute capability {}.{}", devID, deviceProp.name,
               deviceProp.major, deviceProp.minor);

  return devID;
}
#endif
} // namespace cuda