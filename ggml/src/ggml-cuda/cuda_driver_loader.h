#pragma once

// dlopen libcuda.so.1 / nvcuda.dll instead of linking CUDA::cuda_driver, so the
// backend loads where the driver SONAME is unavailable. No-op unless
// GGML_CUDA_DRIVER_DLOPEN is defined; HIP/MUSA are excluded.

#include "vendors/cuda.h"

#if defined(GGML_CUDA_DRIVER_DLOPEN) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include <mutex>

#ifdef _WIN32
#   define WIN32_LEAN_AND_MEAN
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#else
#   include <dlfcn.h>
#endif

typedef CUresult (*cuGetErrorString_t)              (CUresult, const char **);
typedef CUresult (*cuDeviceGet_t)                   (CUdevice *, int);
typedef CUresult (*cuDeviceGetAttribute_t)          (int *, CUdevice_attribute, CUdevice);
typedef CUresult (*cuMemGetAllocationGranularity_t) (size_t *, const CUmemAllocationProp *, CUmemAllocationGranularity_flags);
typedef CUresult (*cuMemCreate_t)                   (CUmemGenericAllocationHandle *, size_t, const CUmemAllocationProp *, unsigned long long);
typedef CUresult (*cuMemAddressReserve_t)           (CUdeviceptr *, size_t, size_t, CUdeviceptr, unsigned long long);
typedef CUresult (*cuMemMap_t)                      (CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long);
typedef CUresult (*cuMemRelease_t)                  (CUmemGenericAllocationHandle);
typedef CUresult (*cuMemSetAccess_t)                (CUdeviceptr, size_t, const CUmemAccessDesc *, size_t);
typedef CUresult (*cuMemUnmap_t)                    (CUdeviceptr, size_t);
typedef CUresult (*cuMemAddressFree_t)              (CUdeviceptr, size_t);

struct ggml_cuda_driver_procs {
    cuGetErrorString_t              cuGetErrorString;
    cuDeviceGet_t                   cuDeviceGet;
    cuDeviceGetAttribute_t          cuDeviceGetAttribute;
    cuMemGetAllocationGranularity_t cuMemGetAllocationGranularity;
    cuMemCreate_t                   cuMemCreate;
    cuMemAddressReserve_t           cuMemAddressReserve;
    cuMemMap_t                      cuMemMap;
    cuMemRelease_t                  cuMemRelease;
    cuMemSetAccess_t                cuMemSetAccess;
    cuMemUnmap_t                    cuMemUnmap;
    cuMemAddressFree_t              cuMemAddressFree;
};

// Loads the driver on first call (thread-safe); false on failure (VMM disabled).
bool ggml_cuda_driver_loaded();

// Forward to the resolved pointers; return CUDA_ERROR_NOT_INITIALIZED if unloaded.
CUresult ggml_cuGetErrorString              (CUresult error, const char **pStr);
CUresult ggml_cuDeviceGet                   (CUdevice *device, int ordinal);
CUresult ggml_cuDeviceGetAttribute          (int *pi, CUdevice_attribute attrib, CUdevice dev);
CUresult ggml_cuMemGetAllocationGranularity (size_t *granularity, const CUmemAllocationProp *prop, CUmemAllocationGranularity_flags option);
CUresult ggml_cuMemCreate                   (CUmemGenericAllocationHandle *handle, size_t size, const CUmemAllocationProp *prop, unsigned long long flags);
CUresult ggml_cuMemAddressReserve           (CUdeviceptr *ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags);
CUresult ggml_cuMemMap                      (CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags);
CUresult ggml_cuMemRelease                  (CUmemGenericAllocationHandle handle);
CUresult ggml_cuMemSetAccess                (CUdeviceptr ptr, size_t size, const CUmemAccessDesc *desc, size_t count);
CUresult ggml_cuMemUnmap                    (CUdeviceptr ptr, size_t size);
CUresult ggml_cuMemAddressFree              (CUdeviceptr ptr, size_t size);

// Redirect bare driver names to the wrappers so call sites stay unchanged.
#define cuGetErrorString              ggml_cuGetErrorString
#define cuDeviceGet                   ggml_cuDeviceGet
#define cuDeviceGetAttribute          ggml_cuDeviceGetAttribute
#define cuMemGetAllocationGranularity ggml_cuMemGetAllocationGranularity
#define cuMemCreate                   ggml_cuMemCreate
#define cuMemAddressReserve           ggml_cuMemAddressReserve
#define cuMemMap                      ggml_cuMemMap
#define cuMemRelease                  ggml_cuMemRelease
#define cuMemSetAccess                ggml_cuMemSetAccess
#define cuMemUnmap                    ggml_cuMemUnmap
#define cuMemAddressFree              ggml_cuMemAddressFree

#endif // GGML_CUDA_DRIVER_DLOPEN && !GGML_USE_HIP && !GGML_USE_MUSA
