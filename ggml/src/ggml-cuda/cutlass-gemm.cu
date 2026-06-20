#include "cutlass-gemm.cuh"

#ifdef GGML_CUDA_USE_CUTLASS

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/epilogue/thread/linear_combination.h"

// ---------------------------------------------------------------------------
// Arch mapping: CUTLASS 2.x GemmUniversal uses DefaultGemmConfiguration
// which only specializes for Sm70, Sm75, Sm80, Sm90.
// Sm86/Sm89 are covered by Sm80 specialization (same tensor core ISA).
// ---------------------------------------------------------------------------

#if GGML_CUDA_CUTLASS_ARCH >= 90
    using CutlassArchTag = cutlass::arch::Sm80;   // Sm90-specific TMA requires 3.x API
#elif GGML_CUDA_CUTLASS_ARCH >= 80
    using CutlassArchTag = cutlass::arch::Sm80;
#elif GGML_CUDA_CUTLASS_ARCH >= 75
    using CutlassArchTag = cutlass::arch::Sm75;
#elif GGML_CUDA_CUTLASS_ARCH >= 70
    using CutlassArchTag = cutlass::arch::Sm70;
#else
    #error "GGML_CUDA_CUTLASS_ARCH must be >= 70"
#endif

// ---------------------------------------------------------------------------
// CUTLASS GEMM type definitions
//
// Layout mapping from cuBLAS to CUTLASS:
//   CUBLAS_OP_T + col-major -> RowMajor
//   CUBLAS_OP_N + col-major -> ColumnMajor
//
// We use float accumulators for all FP16/BF16 paths (matches CUBLAS_COMPUTE_32F).
// The CUBLAS_COMPUTE_16F fallback path also uses F32 accumulation in CUTLASS,
// which is numerically superior and matches what most GPUs prefer.
// ---------------------------------------------------------------------------

// BF16 × BF16 -> BF16 (compute F32)
using GemmBF16BF16BF16 = cutlass::gemm::device::GemmUniversal<
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    CutlassArchTag,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
        cutlass::bfloat16_t, 8,
        float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3
>;

// F16 × F16 -> F32 (compute F32, used for ALL fp16 paths)
using GemmF16F16F32 = cutlass::gemm::device::GemmUniversal<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    float,           cutlass::layout::ColumnMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    CutlassArchTag,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
        float, 4,
        float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3
>;

// F32 × F32 -> F32 (SIMT, scalar epilogue)
using GemmF32F32F32 = cutlass::gemm::device::GemmUniversal<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor,
    float,
    cutlass::arch::OpClassSimt,
    CutlassArchTag,
    cutlass::gemm::GemmShape<128, 128, 8>,
    cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<
        float, 1,
        float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4
>;

// -----------------------------------------------------------------------
// Out-product GEMM types: CUBLAS_OP_N, CUBLAS_OP_N or CUBLAS_OP_T
// op(A) = A (not transposed) → A is ColumnMajor in CUTLASS
// op(B) depends on src1 transpose state:
//   CUBLAS_OP_N → B ColumnMajor
//   CUBLAS_OP_T → B RowMajor
// C is ColumnMajor (same as TN)
// -----------------------------------------------------------------------

// F32 × F32 -> F32, NN (A ColumnMajor, B ColumnMajor)
using GemmF32F32F32_NN = cutlass::gemm::device::GemmUniversal<
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor,
    float,
    cutlass::arch::OpClassSimt,
    CutlassArchTag,
    cutlass::gemm::GemmShape<128, 128, 8>,
    cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<
        float, 1,
        float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4
>;

// F32 × F32 -> F32, NT (A ColumnMajor, B RowMajor)
using GemmF32F32F32_NT = cutlass::gemm::device::GemmUniversal<
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::ColumnMajor,
    float,
    cutlass::arch::OpClassSimt,
    CutlassArchTag,
    cutlass::gemm::GemmShape<128, 128, 8>,
    cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<
        float, 1,
        float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4
>;

// ---------------------------------------------------------------------------
// Minimum M alignment requirements for CUTLASS epilogue vector stores.
// TensorOp kernels use vectorized epilogue writes that require M >= kCount.
// SIMT kernels (F32) use scalar access and have no alignment requirement.
// ---------------------------------------------------------------------------
static constexpr int CUTLASS_MIN_M_F16  = 4;  // LinearCombination<float, 4>
static constexpr int CUTLASS_MIN_M_BF16 = 8;  // LinearCombination<bfloat16_t, 8>
static constexpr int CUTLASS_MIN_M_F32  = 1;  // LinearCombination<float, 1>

// ---------------------------------------------------------------------------
// Run a CUTLASS GEMM, padding M if needed for alignment.
// When M is smaller than the kernel's minimum, we allocate a padded output
// buffer, run the GEMM with padded dimensions, then copy the valid rows back.
// ---------------------------------------------------------------------------

template <typename GemmOp, int MinM>
static cutlass::Status run_cutlass_gemm(
        int M, int N, int K,
        const void * A, int lda,
        const void * B, int ldb,
        void * C, int ldc,
        float alpha, float beta,
        int batch_count,
        int64_t stride_a, int64_t stride_b, int64_t stride_c,
        size_t element_size,
        cudaStream_t stream) {

    // Pad M to minimum alignment for non-batched GEMMs.
    // Strided batched calls typically have large M (model hidden dimension),
    // so padding is only needed for single-batch inference with M < MinM.
    const int M_padded = (M < MinM && batch_count <= 1) ? MinM : M;

    void * C_buf = C;
    int64_t ldc_buf = ldc;
    void * padded_buf_ptr = nullptr;

    if (M_padded != M) {
        // Allocate padded output buffer. Only occurs for small M (rare).
        ldc_buf = M_padded;
        size_t buf_size = (size_t)M_padded * N * element_size;
        CUDA_CHECK(cudaMalloc(&padded_buf_ptr, buf_size));
        C_buf = padded_buf_ptr;
    }

    typename GemmOp::Arguments args(
        batch_count > 1 ? cutlass::gemm::GemmUniversalMode::kBatched
                        : cutlass::gemm::GemmUniversalMode::kGemm,
        {M_padded, N, K},
        batch_count > 1 ? batch_count : 1,
        {alpha, beta},
        A, B, C_buf, C_buf,
        (int64_t)stride_a, (int64_t)stride_b, (int64_t)stride_c, (int64_t)stride_c,
        lda, ldb, (int)ldc_buf, (int)ldc_buf
    );

    GemmOp gemm_op;
    cutlass::Status status = gemm_op.initialize(args);
    if (status != cutlass::Status::kSuccess) {
        if (padded_buf_ptr) CUDA_CHECK(cudaFree(padded_buf_ptr));
        return status;
    }

    status = gemm_op.run(stream);
    if (status != cutlass::Status::kSuccess) {
        if (padded_buf_ptr) CUDA_CHECK(cudaFree(padded_buf_ptr));
        return status;
    }

    // Copy valid rows from padded buffer back to original output
    if (M_padded != M) {
        CUDA_CHECK(cudaMemcpy2DAsync(
            C, ldc * element_size,            // dst pitch
            C_buf, ldc_buf * element_size,    // src pitch
            M * element_size,                 // width: M elements per column
            N,                                // height: N columns (col-major)
            cudaMemcpyDeviceToDevice,
            stream));
        CUDA_CHECK(cudaFree(padded_buf_ptr));
    }

    return cutlass::Status::kSuccess;
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

static cutlass::Status dispatch_cutlass_gemm(
        cudaStream_t stream,
        int M, int N, int K,
        const void * A, int lda,
        const void * B, int ldb,
        void * C, int ldc,
        float alpha, float beta,
        ggml_type A_type, ggml_type B_type, ggml_type C_type,
        int batch_count,
        int64_t stride_a, int64_t stride_b, int64_t stride_c) {

    if (A_type == GGML_TYPE_BF16 && B_type == GGML_TYPE_BF16 && C_type == GGML_TYPE_BF16) {
        return run_cutlass_gemm<GemmBF16BF16BF16, CUTLASS_MIN_M_BF16>(
            M, N, K, A, lda, B, ldb, C, ldc, alpha, beta,
            batch_count, stride_a, stride_b, stride_c, sizeof(__nv_bfloat16), stream);
    }

    // All F16 paths use F32 accumulation and output
    if (A_type == GGML_TYPE_F16 && B_type == GGML_TYPE_F16) {
        // C_type can be F16 or F32, but output element size is always sizeof(half) for F16 C
        size_t c_size = (C_type == GGML_TYPE_F32) ? sizeof(float) : sizeof(half);
        // Alignment min M is based on the epilogue output type (always float for F16 paths)
        return run_cutlass_gemm<GemmF16F16F32, CUTLASS_MIN_M_F16>(
            M, N, K, A, lda, B, ldb, C, ldc, alpha, beta,
            batch_count, stride_a, stride_b, stride_c, sizeof(float), stream);
    }

    if (A_type == GGML_TYPE_F32 && B_type == GGML_TYPE_F32 && C_type == GGML_TYPE_F32) {
        return run_cutlass_gemm<GemmF32F32F32, CUTLASS_MIN_M_F32>(
            M, N, K, A, lda, B, ldb, C, ldc, alpha, beta,
            batch_count, stride_a, stride_b, stride_c, sizeof(float), stream);
    }

    return cutlass::Status::kErrorInvalidProblem;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void ggml_cuda_cutlass_gemm(
        cudaStream_t stream,
        int M, int N, int K,
        const void * A, int lda,
        const void * B, int ldb,
        void * C, int ldc,
        float alpha, float beta,
        ggml_type A_type, ggml_type B_type, ggml_type C_type,
        ggml_type compute_type,
        int cc) {

    cutlass::Status status = dispatch_cutlass_gemm(
        stream, M, N, K, A, lda, B, ldb, C, ldc,
        alpha, beta, A_type, B_type, C_type,
        1, 0, 0, 0);

    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "ggml_cuda_cutlass_gemm: CUTLASS GEMM failed with status %d\n", (int)status);
        GGML_ABORT("CUTLASS GEMM failed");
    }
}

void ggml_cuda_cutlass_gemm_strided_batched(
        cudaStream_t stream,
        int M, int N, int K,
        const void * A, int lda, int64_t stride_a,
        const void * B, int ldb, int64_t stride_b,
        void * C, int ldc, int64_t stride_c,
        int batch_count,
        float alpha, float beta,
        ggml_type A_type, ggml_type B_type, ggml_type C_type,
        ggml_type compute_type,
        int cc) {

    cutlass::Status status = dispatch_cutlass_gemm(
        stream, M, N, K, A, lda, B, ldb, C, ldc,
        alpha, beta, A_type, B_type, C_type,
        batch_count, stride_a, stride_b, stride_c);

    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "ggml_cuda_cutlass_gemm_strided_batched: CUTLASS GEMM failed with status %d\n", (int)status);
        GGML_ABORT("CUTLASS GEMM failed");
    }
}

// ---------------------------------------------------------------------------
// Out-product GEMM (CUBLAS_OP_N + CUBLAS_OP_N or CUBLAS_OP_T)
// ---------------------------------------------------------------------------

void ggml_cuda_cutlass_out_prod(
        cudaStream_t stream,
        int M, int N, int K,
        const void * A, int lda,
        const void * B, int ldb,
        void * C, int ldc,
        float alpha, float beta,
        bool src1_transposed) {

    cutlass::Status status;
    if (src1_transposed) {
        // CUBLAS_OP_N, CUBLAS_OP_N → A: ColumnMajor, B: ColumnMajor
        // src1 is transposed in memory → already k×n column-major, use as-is
        status = run_cutlass_gemm<GemmF32F32F32_NN, CUTLASS_MIN_M_F32>(
            M, N, K, A, lda, B, ldb, C, ldc, alpha, beta,
            1, 0, 0, 0, sizeof(float), stream);
    } else {
        // CUBLAS_OP_N, CUBLAS_OP_T → A: ColumnMajor, B: RowMajor
        // src1 is not transposed → stored as n×k column-major = k×n row-major
        status = run_cutlass_gemm<GemmF32F32F32_NT, CUTLASS_MIN_M_F32>(
            M, N, K, A, lda, B, ldb, C, ldc, alpha, beta,
            1, 0, 0, 0, sizeof(float), stream);
    }

    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "ggml_cuda_cutlass_out_prod: CUTLASS GEMM failed with status %d\n", (int)status);
        GGML_ABORT("CUTLASS GEMM failed");
    }
}

void ggml_cuda_cutlass_out_prod_strided_batched(
        cudaStream_t stream,
        int M, int N, int K,
        const void * A, int lda, int64_t stride_a,
        const void * B, int ldb, int64_t stride_b,
        void * C, int ldc, int64_t stride_c,
        int batch_count,
        float alpha, float beta,
        bool src1_transposed) {

    cutlass::Status status;
    if (src1_transposed) {
        // src1 transposed in memory → already k×n column-major, use as-is
        status = run_cutlass_gemm<GemmF32F32F32_NN, CUTLASS_MIN_M_F32>(
            M, N, K, A, lda, B, ldb, C, ldc, alpha, beta,
            batch_count, stride_a, stride_b, stride_c, sizeof(float), stream);
    } else {
        // src1 not transposed → stored as n×k column-major = k×n row-major
        status = run_cutlass_gemm<GemmF32F32F32_NT, CUTLASS_MIN_M_F32>(
            M, N, K, A, lda, B, ldb, C, ldc, alpha, beta,
            batch_count, stride_a, stride_b, stride_c, sizeof(float), stream);
    }

    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "ggml_cuda_cutlass_out_prod_strided_batched: CUTLASS GEMM failed with status %d\n", (int)status);
        GGML_ABORT("CUTLASS GEMM failed");
    }
}

#endif // GGML_CUDA_USE_CUTLASS