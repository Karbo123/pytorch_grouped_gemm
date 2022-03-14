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

// // // // // // // // // // // // // // // // // // // // // // // // 


/* some types */
using ElementOutput = cutlass::half_t;
using ElementAccumulator = float;
using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    cutlass::half_t, 
    cutlass::layout::ColumnMajor, 
    cutlass::ComplexTransform::kNone,
    8,
    cutlass::half_t,
    cutlass::layout::ColumnMajor, 
    cutlass::ComplexTransform::kNone,
    8,
    ElementOutput, cutlass::layout::ColumnMajor,
    ElementAccumulator, 
    cutlass::arch::OpClassTensorOp, 
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>, 
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 
    4>::GemmKernel;
using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;
// other types defined at TestbedGrouped
using ElementA = typename GemmGrouped::ElementA;
using ElementB = typename GemmGrouped::ElementB;
using ElementC = typename GemmGrouped::ElementC;
using ElementAccumulator = typename GemmGrouped::ElementAccumulator;
using EpilogueOutputOp = typename GemmGrouped::GemmKernel::Epilogue::OutputOp;
using ElementCompute = typename EpilogueOutputOp::ElementCompute;
using LayoutA = typename GemmGrouped::LayoutA;
using LayoutB = typename GemmGrouped::LayoutB;
using LayoutC = typename GemmGrouped::LayoutC;
using MatrixCoord = typename LayoutC::TensorCoord;

// // // // // // // // // // // // // // // // // // // // // // // // 



void check_ampere_arch()
{
    // get device index
    int device_idx;
    cudaError_t status = cudaGetDevice(&device_idx);
    if (status != cudaSuccess) {
        throw std::runtime_error("cudaGetDevice() failed.\n");
    }

    // get properties
    cudaDeviceProp props;
    status = cudaGetDeviceProperties(&props, device_idx);
    if (status != cudaSuccess) {
        throw std::runtime_error("cudaGetDeviceProperties() failed.\n");
    }

    // assert
    if (__CUDACC_VER_MAJOR__ < 11 || props.major < 8) {
        throw std::runtime_error("CUTLASS's Grouped GEMM example requires a GPU of NVIDIA's Ampere Architecture" \
                                 " or later (compute capability 80 or greater).\n");
    }
}



int get_threadblock_count() 
{
    // returns the number of threadblocks to launch if the kernel can run on the target device. Otherwise, returns zero.
    // determine SMEM requirements and waive if not satisfied
    int smem_size = int(sizeof(typename GemmGrouped::GemmKernel::SharedStorage));

    int device_idx;
    cudaError_t status = cudaGetDevice(&device_idx);
    if (status != cudaSuccess) {
        throw std::runtime_error("cudaGetDevice() failed.\n");
    }

    cudaDeviceProp properties;
    status = cudaGetDeviceProperties(&properties, device_idx);
    if (status != cudaSuccess) {
        throw std::runtime_error("cudaGetDeviceProperties() failed.\n");
    }

    int occupancy = std::min(2, int(properties.sharedMemPerMultiprocessor / smem_size));

    return properties.multiProcessorCount * occupancy;
}


template <typename Element>
void initialize_tensor_(Element *ptr, size_t capacity, cutlass::Distribution::Kind dist_kind, uint32_t seed)
{
    if (dist_kind == cutlass::Distribution::Uniform) {

        Element scope_max, scope_min;
        int bits_input = cutlass::sizeof_bits<Element>::value;
        int bits_output = cutlass::sizeof_bits<typename GemmGrouped::ElementC>::value;

        if (bits_input == 1) {
            scope_max = 2;
            scope_min = 0;
        } else if (bits_input <= 8) {
            scope_max = 2;
            scope_min = -2;
        } else if (bits_output == 16) {
            if (cutlass::sizeof_bits<ElementAccumulator>::value <= 16) {
                scope_max = 5;
                scope_min = -5;
            }
            else {
                scope_max = 8;
                scope_min = -8;
            }
        } else {
            scope_max = 8;
            scope_min = -8;
        }

        cutlass::reference::device::BlockFillRandomUniform(
            ptr, capacity, seed, scope_max, scope_min, 0);
    } 
    else if (dist_kind == cutlass::Distribution::Gaussian) {
        cutlass::reference::device::BlockFillRandomGaussian(
            ptr, capacity, seed, Element(), Element(0.5f));
    }
    else if (dist_kind == cutlass::Distribution::Sequential) {
        // Fill with increasing elements
        cutlass::reference::device::BlockFillSequential(
            ptr, capacity, Element(1), Element());
    } 
    else {
        // Fill with all 1s
        cutlass::reference::device::BlockFillSequential(
            ptr, capacity, Element(), Element(1));
    }
}




// // // // // // // // // // // // // // // // // // // // // // // // 



int main(void)
{
    /* check: compute compatibility >= 8.0 (ampere arch) */
    check_ampere_arch();
    
    /* configurate the GEMM */
    int problem_count = 15;
    float alpha = 1.0f;
    float beta = 0.0f;

    /* prepare problem sizes */
    srand(999);
    std::vector<cutlass::gemm::GemmCoord> all_problems;
    all_problems.reserve(problem_count);
    for (int i = 0; i < problem_count; ++i)
    {
        int m = 8 * (rand() % 256) + 8;
        int n = 8 * (rand() % 256) + 8;
        int k = 8 * (rand() % 256) + 8;
        cutlass::gemm::GemmCoord problem(m, n, k);
        all_problems.push_back(problem);
        std::cout << "[" << i << "-th problem] m = " << m << ", n = " << n << ", k = " << k << std::endl;
    }

    /* some members defined at TestbedGrouped */
    cutlass::Distribution::Kind init_A = cutlass::Distribution::Uniform;
    cutlass::Distribution::Kind init_B = cutlass::Distribution::Uniform;
    cutlass::Distribution::Kind init_C = cutlass::Distribution::Uniform;
    cutlass::DeviceAllocation<cutlass::gemm::GemmCoord> all_problems_device;
    std::vector<int64_t> offset_A;
    std::vector<int64_t> offset_B;
    std::vector<int64_t> offset_C;
    std::vector<int64_t> offset_D;
    std::vector<int64_t> lda_host;
    std::vector<int64_t> ldb_host;
    std::vector<int64_t> ldc_host;
    std::vector<int64_t> ldd_host;
    cutlass::DeviceAllocation<int64_t> lda;
    cutlass::DeviceAllocation<int64_t> ldb;
    cutlass::DeviceAllocation<int64_t> ldc;
    cutlass::DeviceAllocation<int64_t> ldd;
    cutlass::DeviceAllocation<ElementA> block_A;
    cutlass::DeviceAllocation<ElementB> block_B;
    cutlass::DeviceAllocation<ElementC> block_C;
    cutlass::DeviceAllocation<ElementC> block_D;
    cutlass::DeviceAllocation<ElementA *> ptr_A;
    cutlass::DeviceAllocation<ElementB *> ptr_B;
    cutlass::DeviceAllocation<ElementC *> ptr_C;
    cutlass::DeviceAllocation<ElementC *> ptr_D;

    /* check whether have sufficient resource */
    int threadblock_count = get_threadblock_count();
    if (!threadblock_count) {
        throw std::runtime_error("Active CUDA device lacks hardware resources to run CUTLASS Grouped GEMM kernel.\n");
    }

    /* initialize matrices */
    int64_t total_elements_A = 0;
    int64_t total_elements_B = 0;
    int64_t total_elements_C = 0;
    int64_t total_elements_D = 0;
    lda_host.resize(problem_count);
    ldb_host.resize(problem_count);
    ldc_host.resize(problem_count);
    ldd_host.resize(problem_count);
    for (int32_t i = 0; i < problem_count; ++i)
    {
        auto problem = all_problems.at(i);

        lda_host.at(i) = LayoutA::packed({problem.m(), problem.k()}).stride(0);
        ldb_host.at(i) = LayoutB::packed({problem.k(), problem.n()}).stride(0);
        ldc_host.at(i) = LayoutC::packed({problem.m(), problem.n()}).stride(0);
        ldd_host.at(i) = LayoutC::packed({problem.m(), problem.n()}).stride(0);

        offset_A.push_back(total_elements_A);
        offset_B.push_back(total_elements_B);
        offset_C.push_back(total_elements_C);
        offset_D.push_back(total_elements_D);

        int64_t elements_A = problem.m() * problem.k();
        int64_t elements_B = problem.k() * problem.n();
        int64_t elements_C = problem.m() * problem.n();
        int64_t elements_D = problem.m() * problem.n();

        total_elements_A += elements_A;
        total_elements_B += elements_B;
        total_elements_C += elements_C;
        total_elements_D += elements_D;
    }
    all_problems_device.reset(problem_count);
    all_problems_device.copy_from_host(all_problems.data());
    lda.reset(problem_count);
    ldb.reset(problem_count);
    ldc.reset(problem_count);
    ldd.reset(problem_count);
    lda.copy_from_host(lda_host.data());
    ldb.copy_from_host(ldb_host.data());
    ldc.copy_from_host(ldc_host.data());
    ldd.copy_from_host(ldd_host.data());
    block_A.reset(total_elements_A);
    block_B.reset(total_elements_B);
    block_C.reset(total_elements_C);
    block_D.reset(total_elements_D);
    std::vector<ElementA *> ptr_A_host(problem_count);
    std::vector<ElementB *> ptr_B_host(problem_count);
    std::vector<ElementC *> ptr_C_host(problem_count);
    std::vector<ElementC *> ptr_D_host(problem_count);
    for (int32_t i = 0; i < problem_count; ++i) 
    {
        ptr_A_host.at(i) = block_A.get() + offset_A.at(i);
        ptr_B_host.at(i) = block_B.get() + offset_B.at(i);
        ptr_C_host.at(i) = block_C.get() + offset_C.at(i);
        ptr_D_host.at(i) = block_D.get() + offset_D.at(i);
    }
    ptr_A.reset(problem_count);
    ptr_B.reset(problem_count);
    ptr_C.reset(problem_count);
    ptr_D.reset(problem_count);
    ptr_A.copy_from_host(ptr_A_host.data());
    ptr_B.copy_from_host(ptr_B_host.data());
    ptr_C.copy_from_host(ptr_C_host.data());
    ptr_D.copy_from_host(ptr_D_host.data());
    // fill values
    initialize_tensor_(block_A.get(), total_elements_A, init_A, 123);
    initialize_tensor_(block_B.get(), total_elements_B, init_B, 456);
    initialize_tensor_(block_C.get(), total_elements_C, init_C, 789);
    cutlass::reference::device::BlockFillSequential(
        block_D.get(), total_elements_D, ElementC(), ElementC());

    /* configure the GEMM arguments */
    typename EpilogueOutputOp::Params epilogue_op(alpha, beta);
    typename GemmGrouped::Arguments args(
      all_problems_device.get(),
      problem_count,
      threadblock_count,
      epilogue_op,
      ptr_A.get(),
      ptr_B.get(),
      ptr_C.get(),
      ptr_D.get(),
      lda.get(),
      ldb.get(),
      ldc.get(),
      ldd.get()
    );

    /* some variables */
    GemmGrouped gemm;
    cutlass::Status status;
    status = gemm.initialize(args);
    if (status != cutlass::Status::kSuccess) {
      throw std::runtime_error("Failed to initialize CUTLASS Grouped GEMM kernel.\n");
    }

    /* start to run */
    status = gemm.run();
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("Failed to run CUTLASS Grouped GEMM kernel.\n");
    }

    /* wait for completion */
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess)  {
        throw std::runtime_error("cudaDeviceSynchronize() failed.\n");
    }
    
    /* verify the results */
    bool passed = true;
    for (int32_t i = 0; i < problem_count; ++i) 
    {
        cutlass::gemm::GemmCoord problem = all_problems.at(i);

        LayoutA layout_A(lda_host.at(i));
        LayoutB layout_B(ldb_host.at(i));
        LayoutC layout_C(ldc_host.at(i));
        LayoutC layout_D(ldd_host.at(i));

        MatrixCoord extent_A{problem.m(), problem.k()};
        MatrixCoord extent_B{problem.k(), problem.n()};
        MatrixCoord extent_C{problem.m(), problem.n()};
        
        cutlass::TensorView<ElementA, LayoutA> view_A(block_A.get() + offset_A.at(i), layout_A, extent_A);
        cutlass::TensorView<ElementB, LayoutB> view_B(block_B.get() + offset_B.at(i), layout_B, extent_B);
        cutlass::TensorView<ElementC, LayoutC> view_C(block_C.get() + offset_C.at(i), layout_C, extent_C);

        cutlass::DeviceAllocation<ElementC>    block_Ref(layout_D.capacity(extent_C));
        cutlass::TensorView<ElementC, LayoutC> view_Ref_device(block_Ref.get(), layout_D, extent_C);

        // Reference GEMM
        cutlass::reference::device::GemmComplex<
            ElementA, LayoutA,
            ElementB, LayoutB,
            ElementC, LayoutC, 
            ElementCompute, ElementAccumulator
        >(
            problem,
            alpha, 
            view_A,
            GemmGrouped::kTransformA,
            view_B,
            GemmGrouped::kTransformB,
            beta, 
            view_C, 
            view_Ref_device, 
            ElementAccumulator(0)
        );

        // Copy to host memory
        std::vector<ElementC> matrix_D(layout_D.capacity(extent_C));
        std::vector<ElementC> matrix_Ref(layout_D.capacity(extent_C));

        cutlass::device_memory::copy_to_host(matrix_D.data(),   block_D.get() + offset_D.at(i), matrix_D.size());
        cutlass::device_memory::copy_to_host(matrix_Ref.data(), block_Ref.get(),                matrix_D.size());

        cutlass::TensorView<ElementC, LayoutC> view_D(  matrix_D.data(),   layout_D, extent_C);
        cutlass::TensorView<ElementC, LayoutC> view_Ref(matrix_Ref.data(), layout_D, extent_C);
        
        // Reference check
        passed = cutlass::reference::host::TensorEquals(view_D, view_Ref);

        if (!passed) {
            std::stringstream strstream;
            strstream << "\n***\nError - problem " << i << " failed the QA check\n***\n" << std::endl;
            throw std::runtime_error(strstream.str());
        }
    }
    
    std::cout << "all passed" << std::endl;
}


