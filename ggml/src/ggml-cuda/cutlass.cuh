#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/layout/matrix.h"

#ifndef CUTLASS_TARGET_ARCH
    #warning "CUTLASS_TARGET_ARCH not defined! Defaulting to SM80."
    #define CUTLASS_TARGET_ARCH 80
#endif

#if CUTLASS_TARGET_ARCH >= 90

#include "cute/tensor.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#if CUTLASS_TARGET_ARCH >= 100
    using ArchTag = cutlass::arch::Sm100; // Blackwell
#else
    using ArchTag = cutlass::arch::Sm90;  // Hopper
#endif

struct CutlassConfig {
    using ElementA = cutlass::bfloat16_t;
    using LayoutA  = cutlass::layout::RowMajor;
    constexpr static int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

    using ElementB = cutlass::bfloat16_t;
    using LayoutB  = cutlass::layout::ColumnMajor; // Implicit Transpose
    constexpr static int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

    using ElementC = float;
    using LayoutC  = cutlass::layout::RowMajor;
    constexpr static int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

    using ElementD = float;
    using LayoutD  = cutlass::layout::RowMajor;

    using ElementAccumulator = float;
    using OperatorClass = cutlass::arch::OpClassTensorOp;

    using MmaTileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
    using ClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementA, LayoutA, AlignmentA,
        ElementB, LayoutB, AlignmentB,
        ElementAccumulator,
        MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<sizeof(ElementAccumulator)>,
        cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        MmaTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, LayoutC, AlignmentC,
        ElementD, LayoutD, 128 / cutlass::sizeof_bits<ElementD>::value,
        cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue
    >;

    using GemmHandle = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

namespace cutlass {
    template <class StrideType, class Shape>
    CUTE_HOST_DEVICE auto make_cute_packed_stride(StrideType, Shape const& shape) {
        using namespace cute;
        if constexpr (std::is_same_v<StrideType, cutlass::layout::ColumnMajor> ||
                      std::is_same_v<StrideType, CutlassConfig::GemmHandle::GemmKernel::StrideB>) {
            return make_stride(Int<1>{}, get<0>(shape), get<0>(shape) * get<1>(shape));
        } else {
            return make_stride(get<1>(shape), Int<1>{}, get<0>(shape) * get<1>(shape));
        }
    }
}

using BestGemm = typename CutlassConfig::GemmHandle;

inline int launch_cutlass_gemm(
    int M, int N, int K,
    float alpha,
    const void* A, int lda_elements,
    const void* B, int ldb_elements,
    float beta,
    void* C, int ldc_elements,
    cudaStream_t stream) {

    BestGemm gemm_op;

    // These calls now use the polyfill above
    auto stride_A = cutlass::make_cute_packed_stride(CutlassConfig::GemmHandle::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(CutlassConfig::GemmHandle::GemmKernel::StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(CutlassConfig::GemmHandle::GemmKernel::StrideC{}, {M, N, 1});
    auto stride_D = stride_C;

    typename BestGemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        { // Mainloop Args
            reinterpret_cast<const typename CutlassConfig::ElementA*>(A), stride_A,
            reinterpret_cast<const typename CutlassConfig::ElementB*>(B), stride_B
        },
        { // Epilogue Args
            {alpha, beta},
            reinterpret_cast<const typename CutlassConfig::ElementC*>(C), stride_C,
            reinterpret_cast<typename CutlassConfig::ElementD*>(C), stride_D
        }
    };

    size_t workspace_size = gemm_op.get_workspace_size(args);
    void* workspace = nullptr;
    if (workspace_size > 0) {
        if(cudaMallocAsync(&workspace, workspace_size, stream) != cudaSuccess) return 1;
    }

    auto status = gemm_op.run(args, workspace, stream);
    if (workspace) cudaFreeAsync(workspace, stream);

    return (status == cutlass::Status::kSuccess) ? 0 : 1;
}

#else

#include "cutlass/gemm/device/gemm.h"

#if CUTLASS_TARGET_ARCH >= 80
    using ArchTag = cutlass::arch::Sm80;
    using ElementInput = cutlass::bfloat16_t;
#else
    using ArchTag = cutlass::arch::Sm75;
    using ElementInput = cutlass::half_t;
#endif

using BestGemm = cutlass::gemm::device::Gemm<
    ElementInput, cutlass::layout::RowMajor,
    ElementInput, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    ArchTag
>;

inline int launch_cutlass_gemm(
    int M, int N, int K,
    float alpha,
    const void* A, int lda_elements,
    const void* B, int ldb_elements,
    float beta,
    void* C, int ldc_elements,
    cudaStream_t stream) {

    BestGemm gemm_op;

    typename BestGemm::Arguments args(
        {M, N, K},
        {reinterpret_cast<const typename BestGemm::ElementA*>(A), lda_elements},
        {reinterpret_cast<const typename BestGemm::ElementB*>(B), ldb_elements},
        {reinterpret_cast<typename BestGemm::ElementC*>(C), ldc_elements},
        {reinterpret_cast<typename BestGemm::ElementC*>(C), ldc_elements},
        {alpha, beta}
    );

    size_t workspace_size = gemm_op.get_workspace_size(args);
    void* workspace = nullptr;
    if(workspace_size > 0) cudaMallocAsync(&workspace, workspace_size, stream);

    cutlass::Status status = gemm_op(args, workspace, stream);

    if(workspace) cudaFreeAsync(workspace, stream);

    return (status == cutlass::Status::kSuccess) ? 0 : 1;
}

#endif
