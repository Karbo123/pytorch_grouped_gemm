# High Performance Grouped GEMM in PyTorch

computing multiple GEMMs with different matrix sizes

why: pytorch/cublas only support batched GEMM (use same M, N, K), but does not support grouped GEMM (use different M, N, K for each matrix multiplication)

todo: plan to use [cutlass](https://github.com/NVIDIA/cutlass/blob/master/examples/24_gemm_grouped/gemm_grouped.cu) and provide a interface for pytorch

