/*
    Grouped GEMM for PyTorch
*/

// // // // // // // // // // // // // // // // // // // // // // // // 

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/device/gemm_universal.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm_complex.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"

#include <torch/extension.h>

namespace py = pybind11;

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous \n")
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor \n")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// // // // // // // // // // // // // // // // // // // // // // // // 


auto getDeviceProps()
{
    int device_idx;
    cudaError_t status = cudaGetDevice(&device_idx);
    TORCH_CHECK(status == cudaSuccess, "cudaGetDevice() failed \n");

    cudaDeviceProp props;
    status = cudaGetDeviceProperties(&props, device_idx);
    TORCH_CHECK(status == cudaSuccess, "cudaGetDeviceProperties() failed \n");

    return props;
}



// // // // // // // // // // // // // // // // // // // // // // // // 

using ListTensor = std::vector<torch::Tensor>;


/*
    Perform Grouped Matrix Multiplication (in a single cuda kernel launch)
    A: list[Tensor]
    B: list[Tensor]
    C: list[Tensor]
    D = [ alpha * A[i] x B[i] + beta * C for i in range(num_matrices) ]
*/
template <
    typename ElementA = cutlass::half_t, typename LayoutA = cutlass::layout::ColumnMajor, int kAlignmentA = 8,
    typename ElementB = cutlass::half_t, typename LayoutB = cutlass::layout::ColumnMajor, int kAlignmentB = 8,
    typename ElementC = cutlass::half_t, typename LayoutC = cutlass::layout::ColumnMajor,
    typename ElementAccumulator = float,
    typename OperatorClass = cutlass::arch::OpClassTensorOp,
    typename ArchTag = cutlass::arch::Sm80
>
void GroupedGEMM(const ListTensor& matrices_A,
                 const ListTensor& matrices_B,
                 const ListTensor& matrices_C,
                 const ListTensor& matrices_D,
                 float alpha = 1.0, 
                 float beta = 0.0
                )
{
    /* some types */
    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
        ElementA, LayoutA, cutlass::ComplexTransform::kNone, kAlignmentA,
        ElementB, LayoutB, cutlass::ComplexTransform::kNone, kAlignmentB,
        ElementC, LayoutC,
        ElementAccumulator, 
        OperatorClass,
        ArchTag,
        cutlass::gemm::GemmShape<128, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>, 
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<
            ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
            ElementAccumulator, ElementAccumulator>,
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 
        4>::GemmKernel;
    using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;
    // other types defined at TestbedGrouped
    using EpilogueOutputOp = typename GemmGrouped::GemmKernel::Epilogue::OutputOp;
    using ElementCompute = typename EpilogueOutputOp::ElementCompute;
    using MatrixCoord = typename LayoutC::TensorCoord;


    /* preapre variables */
    auto props = getDeviceProps();
    auto num_matrices = matrices_A.size();


    /* check */
    // assert list length
    TORCH_CHECK(matrices_A.size() == matrices_B.size() && \
                matrices_B.size() == matrices_C.size() && \
                matrices_C.size() == matrices_D.size(), "num of matrices mismatches \n");
    // assert architecture
    if (std::is_same<ArchTag, cutlass::arch::Sm80>::value ||
        std::is_same<ArchTag, cutlass::arch::Sm86>::value
    )
    {
        TORCH_CHECK(__CUDACC_VER_MAJOR__ >= 11 && props.major >= 8,
                    "compute capability must be >= 80 \n");
    } else
    {
        TORCH_CHECK(false, "unknown cuda arch \n");
    }
    // assert resource available
    int smem_size = int(sizeof(typename GemmGrouped::GemmKernel::SharedStorage));
    int occupancy = std::min(2, int(props.sharedMemPerMultiprocessor / smem_size));
    int threadblock_count = props.multiProcessorCount * occupancy;
    TORCH_CHECK(threadblock_count > 0, "lack hardware resources to launch cuda kernel \n");
    
    /* prepare all problems' size */
    std::vector<cutlass::gemm::GemmCoord> all_problems(num_matrices);
    cutlass::DeviceAllocation<cutlass::gemm::GemmCoord> all_problems_device;
    for (auto i = 0; i < num_matrices; ++i)
    {
        auto& matrix_A = matrices_A[i];
        auto& matrix_B = matrices_B[i];
        auto& matrix_C = matrices_C[i];
        auto& matrix_D = matrices_D[i];

        CHECK_INPUT(matrix_A);
        CHECK_INPUT(matrix_B);
        CHECK_INPUT(matrix_C);
        CHECK_INPUT(matrix_D);

        auto m  = matrix_A.size(0);
        auto k  = matrix_A.size(1);
        auto k2 = matrix_B.size(0);
        auto n  = matrix_B.size(1);
        
        auto mC = matrix_C.size(0);
        auto nC = matrix_C.size(1);

        auto mD = matrix_D.size(0);
        auto nD = matrix_D.size(1);

        // check the hidden dimension - k
        if (k != k2)
        {
            std::stringstream s;
            s << "cannot apply matrix multiplication between two shapes: A=(" << m << ", " << k << ") and B=(" << k2 << ", " << n << ") \n";
            TORCH_CHECK(false, s.str());
        }

        // check shape match - A * B, C
        if (m != mC || n != nC)
        {
            std::stringstream s;
            s << "matrix A * B cannot add with matrix C between two shapes: A*B=(" << m << ", " << n << ") and C=(" << mC << ", " << nC << ") \n";
            TORCH_CHECK(false, s.str());
        }

        // check shape match - A * B, D
        if (m != mD || n != nD)
        {
            std::stringstream s;
            s << "matrix A * B cannot add with matrix D between two shapes: A*B=(" << m << ", " << n << ") and D=(" << mD << ", " << nD << ") \n";
            TORCH_CHECK(false, s.str());
        }

        all_problems[i] = cutlass::gemm::GemmCoord(m, n, k);
    }
    all_problems_device.reset(num_matrices);
    all_problems_device.copy_from_host(all_problems.data());


    /* prepare leading dimension */
    std::vector<int64_t> lda_host(num_matrices);
    std::vector<int64_t> ldb_host(num_matrices);
    std::vector<int64_t> ldc_host(num_matrices);
    for (auto i = 0; i < num_matrices; ++i)
    {
        auto& problem = all_problems[i];
        
        // TODO: use a generic striding scheme @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @
        lda_host[i] = LayoutA::packed({problem.m(), problem.k()}).stride(0);
        ldb_host[i] = LayoutB::packed({problem.k(), problem.n()}).stride(0);
        ldc_host[i] = LayoutC::packed({problem.m(), problem.n()}).stride(0);
    }
    cutlass::DeviceAllocation<int64_t> lda; lda.reset(num_matrices); lda.copy_from_host(lda_host.data());
    cutlass::DeviceAllocation<int64_t> ldb; ldb.reset(num_matrices); ldb.copy_from_host(ldb_host.data());
    cutlass::DeviceAllocation<int64_t> ldc; ldc.reset(num_matrices); ldc.copy_from_host(ldc_host.data());
    

    /* prepare pointers */
    std::vector<ElementA *> ptr_A_host(num_matrices);
    std::vector<ElementB *> ptr_B_host(num_matrices);
    std::vector<ElementC *> ptr_C_host(num_matrices);
    std::vector<ElementC *> ptr_D_host(num_matrices);
    using TorchTypeA = typename std::conditional<std::is_same<ElementA, cutlass::half_t>::value, at::Half, ElementA>::type;
    using TorchTypeB = typename std::conditional<std::is_same<ElementB, cutlass::half_t>::value, at::Half, ElementB>::type;
    using TorchTypeC = typename std::conditional<std::is_same<ElementC, cutlass::half_t>::value, at::Half, ElementC>::type;
    for (auto i = 0; i < num_matrices; ++i)
    {
        ptr_A_host[i] = reinterpret_cast<ElementA *>(matrices_A[i].data_ptr<TorchTypeA>());
        ptr_B_host[i] = reinterpret_cast<ElementB *>(matrices_B[i].data_ptr<TorchTypeB>());
        ptr_C_host[i] = reinterpret_cast<ElementC *>(matrices_C[i].data_ptr<TorchTypeC>());
        ptr_D_host[i] = reinterpret_cast<ElementC *>(matrices_D[i].data_ptr<TorchTypeC>());
    }
    cutlass::DeviceAllocation<ElementA *> ptr_A; ptr_A.reset(num_matrices); ptr_A.copy_from_host(ptr_A_host.data());
    cutlass::DeviceAllocation<ElementB *> ptr_B; ptr_B.reset(num_matrices); ptr_B.copy_from_host(ptr_B_host.data());
    cutlass::DeviceAllocation<ElementC *> ptr_C; ptr_C.reset(num_matrices); ptr_C.copy_from_host(ptr_C_host.data());
    cutlass::DeviceAllocation<ElementC *> ptr_D; ptr_D.reset(num_matrices); ptr_D.copy_from_host(ptr_D_host.data());
    

    /* configurate the GEMM args */
    typename EpilogueOutputOp::Params epilogue_op(alpha, beta);
    typename GemmGrouped::Arguments args(
      all_problems_device.get(),
      num_matrices,
      threadblock_count,
      epilogue_op,
      ptr_A.get(),
      ptr_B.get(),
      ptr_C.get(),
      ptr_D.get(),
      lda.get(),
      ldb.get(),
      ldc.get(),
      ldc.get()
    );


    /* start to launch the GEMM kernel */
    GemmGrouped gemm;
    cutlass::Status status;
    status = gemm.initialize(args);
    TORCH_CHECK(status == cutlass::Status::kSuccess, "GroupedGEMM kernel initialization: failed \n");
    status = gemm.run();
    TORCH_CHECK(status == cutlass::Status::kSuccess, "GroupedGEMM kernel run: failed \n");

    // wait
    cudaError_t error = cudaDeviceSynchronize();
    TORCH_CHECK(error == cudaSuccess, "cudaDeviceSynchronize() failed \n");
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("GroupedGEMM", &GroupedGEMM<cutlass::half_t, cutlass::layout::ColumnMajor, 8,
                                      cutlass::half_t, cutlass::layout::ColumnMajor, 8,
                                      cutlass::half_t, cutlass::layout::ColumnMajor,
                                      float,
                                      cutlass::arch::OpClassTensorOp,
                                      cutlass::arch::Sm80>, 
          "GroupedGEMM (CUDA)", 
          py::arg("matrices_A"), 
          py::arg("matrices_B"), 
          py::arg("matrices_C"), 
          py::arg("matrices_D"),
          py::arg("alpha"),
          py::arg("beta")
        );
}
