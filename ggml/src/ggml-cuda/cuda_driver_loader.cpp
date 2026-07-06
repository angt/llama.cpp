#include "cuda_driver_loader.h"

#if defined(GGML_CUDA_DRIVER_DLOPEN) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// The header redirects bare cu* names to wrappers; the loader must use real names.
#undef cuGetErrorString
#undef cuDeviceGet
#undef cuDeviceGetAttribute
#undef cuMemGetAllocationGranularity
#undef cuMemCreate
#undef cuMemAddressReserve
#undef cuMemMap
#undef cuMemRelease
#undef cuMemSetAccess
#undef cuMemUnmap
#undef cuMemAddressFree

#include "ggml-impl.h"

namespace {
#ifdef _WIN32
using lib_handle_t = HMODULE;

lib_handle_t lib_open(const wchar_t * path) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);
    HMODULE h = LoadLibraryW(path);
    SetErrorMode(old_mode);
    return h;
}

void * lib_sym(lib_handle_t h, const char * name) {
    return (void *) GetProcAddress(h, name);
}
#else
using lib_handle_t = void *;

lib_handle_t lib_open(const char * path) {
    return dlopen(path, RTLD_NOW | RTLD_LOCAL);
}

void * lib_sym(lib_handle_t h, const char * name) {
    return dlsym(h, name);
}
#endif

struct driver_state {
    std::once_flag once;
    ggml_cuda_driver_procs procs{};
    bool ok = false;
};

driver_state & state() {
    static driver_state s;
    return s;
}

void load_once() {
    auto & s = state();

    lib_handle_t handle = nullptr;
#ifdef _WIN32
    handle = lib_open(L"nvcuda.dll");
#else
    // Try the versioned SONAME first, then the unversioned fallback.
    handle = lib_open("libcuda.so.1");
    if (!handle) {
        handle = lib_open("libcuda.so");
    }
#endif
    if (!handle) {
        GGML_LOG_WARN("%s: CUDA driver library not found, VMM will be disabled\n", __func__);
        return;
    }

    ggml_cuda_driver_procs p;
    p.cuGetErrorString              = (cuGetErrorString_t)              lib_sym(handle, "cuGetErrorString");
    p.cuDeviceGet                   = (cuDeviceGet_t)                   lib_sym(handle, "cuDeviceGet");
    p.cuDeviceGetAttribute          = (cuDeviceGetAttribute_t)          lib_sym(handle, "cuDeviceGetAttribute");
    p.cuMemGetAllocationGranularity = (cuMemGetAllocationGranularity_t) lib_sym(handle, "cuMemGetAllocationGranularity");
    p.cuMemCreate                   = (cuMemCreate_t)                   lib_sym(handle, "cuMemCreate");
    p.cuMemAddressReserve           = (cuMemAddressReserve_t)           lib_sym(handle, "cuMemAddressReserve");
    p.cuMemMap                      = (cuMemMap_t)                      lib_sym(handle, "cuMemMap");
    p.cuMemRelease                  = (cuMemRelease_t)                  lib_sym(handle, "cuMemRelease");
    p.cuMemSetAccess                = (cuMemSetAccess_t)                lib_sym(handle, "cuMemSetAccess");
    p.cuMemUnmap                    = (cuMemUnmap_t)                    lib_sym(handle, "cuMemUnmap");
    p.cuMemAddressFree              = (cuMemAddressFree_t)              lib_sym(handle, "cuMemAddressFree");

    if (!p.cuGetErrorString || !p.cuDeviceGet || !p.cuDeviceGetAttribute ||
        !p.cuMemGetAllocationGranularity || !p.cuMemCreate || !p.cuMemAddressReserve ||
        !p.cuMemMap || !p.cuMemRelease || !p.cuMemSetAccess || !p.cuMemUnmap ||
        !p.cuMemAddressFree) {
        GGML_LOG_WARN("%s: failed to resolve CUDA driver symbols, VMM will be disabled\n", __func__);
        return;
    }

    s.procs = p;
    s.ok = true;
}
} // namespace

bool ggml_cuda_driver_loaded() {
    auto & s = state();
    std::call_once(s.once, load_once);
    return s.ok;
}

static const ggml_cuda_driver_procs & procs() {
    return state().procs;
}

#define GGML_CU_FORWARD_OR_FAIL(name, ...)                                   \
    if (!ggml_cuda_driver_loaded()) {                                        \
        return CUDA_ERROR_NOT_INITIALIZED;                                   \
    }                                                                        \
    return procs().name(__VA_ARGS__)

CUresult ggml_cuGetErrorString(CUresult error, const char **pStr) {
    GGML_CU_FORWARD_OR_FAIL(cuGetErrorString, error, pStr);
}
CUresult ggml_cuDeviceGet(CUdevice *device, int ordinal) {
    GGML_CU_FORWARD_OR_FAIL(cuDeviceGet, device, ordinal);
}
CUresult ggml_cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev) {
    GGML_CU_FORWARD_OR_FAIL(cuDeviceGetAttribute, pi, attrib, dev);
}
CUresult ggml_cuMemGetAllocationGranularity(size_t *granularity, const CUmemAllocationProp *prop, CUmemAllocationGranularity_flags option) {
    GGML_CU_FORWARD_OR_FAIL(cuMemGetAllocationGranularity, granularity, prop, option);
}
CUresult ggml_cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size, const CUmemAllocationProp *prop, unsigned long long flags) {
    GGML_CU_FORWARD_OR_FAIL(cuMemCreate, handle, size, prop, flags);
}
CUresult ggml_cuMemAddressReserve(CUdeviceptr *ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags) {
    GGML_CU_FORWARD_OR_FAIL(cuMemAddressReserve, ptr, size, alignment, addr, flags);
}
CUresult ggml_cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags) {
    GGML_CU_FORWARD_OR_FAIL(cuMemMap, ptr, size, offset, handle, flags);
}
CUresult ggml_cuMemRelease(CUmemGenericAllocationHandle handle) {
    GGML_CU_FORWARD_OR_FAIL(cuMemRelease, handle);
}
CUresult ggml_cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc *desc, size_t count) {
    GGML_CU_FORWARD_OR_FAIL(cuMemSetAccess, ptr, size, desc, count);
}
CUresult ggml_cuMemUnmap(CUdeviceptr ptr, size_t size) {
    GGML_CU_FORWARD_OR_FAIL(cuMemUnmap, ptr, size);
}
CUresult ggml_cuMemAddressFree(CUdeviceptr ptr, size_t size) {
    GGML_CU_FORWARD_OR_FAIL(cuMemAddressFree, ptr, size);
}

#undef GGML_CU_FORWARD_OR_FAIL

#endif // GGML_CUDA_DRIVER_DLOPEN && !GGML_USE_HIP && !GGML_USE_MUSA
