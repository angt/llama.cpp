#pragma once

// Load the CUDA driver API (libcuda.so) at runtime via dlopen/dlsym instead of
// linking it directly. This avoids a hard DT_NEEDED on libcuda.so.1 in the
// compiled library/binary, mirroring how libcudart.so and libcublas.so work:
// they only link libdl and dlopen the driver themselves.
//
// The function-pointer types are obtained with decltype(&cuFunc) in an
// unevaluated context, so the real cu* symbols are never referenced and no
// link dependency on libcuda is introduced. The 11 existing call sites use the
// same cu* names as before via the macros below, which redirect the calls to
// the dlsym'd pointers.

#if !defined(GGML_CUDA_DRIVER_DL_H)
#define GGML_CUDA_DRIVER_DL_H

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && !defined(GGML_CUDA_NO_VMM) && !defined(_WIN32)

#include <dlfcn.h>

// The driver API is loaded via dlopen on this platform.

struct ggml_cuda_driver_api {
    decltype(&cuGetErrorString)              p_cuGetErrorString;
    decltype(&cuDeviceGet)                   p_cuDeviceGet;
    decltype(&cuDeviceGetAttribute)           p_cuDeviceGetAttribute;
    decltype(&cuMemGetAllocationGranularity) p_cuMemGetAllocationGranularity;
    decltype(&cuMemCreate)                   p_cuMemCreate;
    decltype(&cuMemAddressReserve)           p_cuMemAddressReserve;
    decltype(&cuMemMap)                      p_cuMemMap;
    decltype(&cuMemSetAccess)                p_cuMemSetAccess;
    decltype(&cuMemRelease)                  p_cuMemRelease;
    decltype(&cuMemUnmap)                    p_cuMemUnmap;
    decltype(&cuMemAddressFree)              p_cuMemAddressFree;
};

// Returns a pointer to the loaded driver API, or nullptr if libcuda could not
// be opened. The library is opened once (thread-safe via a function-local
// static) and stays loaded for the lifetime of the process.
inline const ggml_cuda_driver_api * ggml_cuda_driver() {
    static const ggml_cuda_driver_api * api = []() -> const ggml_cuda_driver_api * {
        void * h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_GLOBAL);
        if (!h) {
            h = dlopen("libcuda.so", RTLD_LAZY | RTLD_GLOBAL);
        }
        if (!h) {
            return nullptr;
        }
        static ggml_cuda_driver_api a;
        #define GGML_CU_DLSYM(x) a.p_##x = (decltype(a.p_##x)) dlsym(h, #x)
        GGML_CU_DLSYM(cuGetErrorString);
        GGML_CU_DLSYM(cuDeviceGet);
        GGML_CU_DLSYM(cuDeviceGetAttribute);
        GGML_CU_DLSYM(cuMemGetAllocationGranularity);
        GGML_CU_DLSYM(cuMemCreate);
        GGML_CU_DLSYM(cuMemAddressReserve);
        GGML_CU_DLSYM(cuMemMap);
        GGML_CU_DLSYM(cuMemSetAccess);
        GGML_CU_DLSYM(cuMemRelease);
        GGML_CU_DLSYM(cuMemUnmap);
        GGML_CU_DLSYM(cuMemAddressFree);
        #undef GGML_CU_DLSYM
        // The VMM code path requires all of these.
        if (!a.p_cuGetErrorString || !a.p_cuDeviceGet || !a.p_cuDeviceGetAttribute ||
            !a.p_cuMemGetAllocationGranularity || !a.p_cuMemCreate || !a.p_cuMemAddressReserve ||
            !a.p_cuMemMap || !a.p_cuMemSetAccess || !a.p_cuMemRelease || !a.p_cuMemUnmap ||
            !a.p_cuMemAddressFree) {
            return nullptr;
        }
        return &a;
    }();
    return api;
}

// True when the CUDA driver API is available. On the dlopen path this reports
// whether libcuda could be loaded; VMM is only used when it is. On platforms
// that link the driver directly (below) it is always true.
inline bool ggml_cuda_driver_available() {
    return ggml_cuda_driver() != nullptr;
}

// Redirect the driver API names to the dlsym'd pointers so existing call sites
// (which use cu* names) work unchanged.
#define cuGetErrorString              ggml_cuda_driver()->p_cuGetErrorString
#define cuDeviceGet                   ggml_cuda_driver()->p_cuDeviceGet
#define cuDeviceGetAttribute          ggml_cuda_driver()->p_cuDeviceGetAttribute
#define cuMemGetAllocationGranularity ggml_cuda_driver()->p_cuMemGetAllocationGranularity
#define cuMemCreate                   ggml_cuda_driver()->p_cuMemCreate
#define cuMemAddressReserve           ggml_cuda_driver()->p_cuMemAddressReserve
#define cuMemMap                      ggml_cuda_driver()->p_cuMemMap
#define cuMemSetAccess                ggml_cuda_driver()->p_cuMemSetAccess
#define cuMemRelease                  ggml_cuda_driver()->p_cuMemRelease
#define cuMemUnmap                    ggml_cuda_driver()->p_cuMemUnmap
#define cuMemAddressFree              ggml_cuda_driver()->p_cuMemAddressFree

#else // driver linked directly (Windows, HIP, MUSA, or VMM disabled)

// On these platforms the CUDA driver is linked normally, so the cu* names are
// the real symbols and the driver is always available.
inline bool ggml_cuda_driver_available() {
    return true;
}

#endif // !GGML_USE_HIP && !GGML_USE_MUSA && !GGML_CUDA_NO_VMM && !_WIN32

#endif // GGML_CUDA_DRIVER_DL_H