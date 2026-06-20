#pragma once

#include "common.cuh"

#ifdef GGML_CUDA_USE_CUTLASS

// Non-batched GEMM: C = alpha * A * B + beta * C
// A layout: RowMajor (equivalent to cuBLAS CUBLAS_OP_T with column-major data)
// B layout: ColumnMajor (equivalent to cuBLAS CUBLAS_OP_N with column-major data)
// C layout: ColumnMajor (result written directly to ggml row-major dst memory)
void ggml_cuda_cutlass_gemm(
        cudaStream_t stream,
        int M, int N, int K,
        const void * A, int lda,
        const void * B, int ldb,
        void * C, int ldc,
        float alpha, float beta,
        ggml_type A_type, ggml_type B_type, ggml_type C_type,
        ggml_type compute_type, // GGML_TYPE_F32 for CUBLAS_COMPUTE_32F, GGML_TYPE_F16 for CUBLAS_COMPUTE_16F
        int cc);

// Strided batched GEMM
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
        int cc);

// Out-product GEMM: A ColumnMajor, B layout depends on src1 transpose
// For CUBLAS_OP_N, CUBLAS_OP_N (src1 transposed):  B retains k×n column-major → use _NN
// For CUBLAS_OP_N, CUBLAS_OP_T (src1 not transposed): B stored n×k col-major → interpret as k×n row-major → use _NT
void ggml_cuda_cutlass_out_prod(
        cudaStream_t stream,
        int M, int N, int K,
        const void * A, int lda,
        const void * B, int ldb,
        void * C, int ldc,
        float alpha, float beta,
        bool src1_transposed);

// Strided batched out-product GEMM
void ggml_cuda_cutlass_out_prod_strided_batched(
        cudaStream_t stream,
        int M, int N, int K,
        const void * A, int lda, int64_t stride_a,
        const void * B, int ldb, int64_t stride_b,
        void * C, int ldc, int64_t stride_c,
        int batch_count,
        float alpha, float beta,
        bool src1_transposed);

#endif // GGML_CUDA_USE_CUTLASS