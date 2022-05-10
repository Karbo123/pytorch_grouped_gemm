# High Performance Grouped GEMM in PyTorch

computing multiple GEMMs with different matrix sizes

why: pytorch/cublas only support batched GEMM (use the same M, N, K), but does not support grouped GEMM (use different M, N, K for each matrix multiplication)

possible applications:
- transformer's attention matrix with different sizes; 
- convolution with different size for each batch (e.g. point clouds with different num of points)

performance overview:
```
testing speed for torch.float16
time for pytorch = 3.5916242599487305
time for cutlass = 0.1151578426361084

testing speed for torch.float32       # note results are for fp32 instead of tf32
time for pytorch = 3.5333731174468994
time for cutlass = 0.14151287078857422

testing speed for torch.float64
time for pytorch = 3.743123769760132
time for cutlass = 0.45430731773376465
```

a build example:
please change `set(COMPILE_CC 80)` in `CMakeLists.txt` to a proper arch before compilation
```
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.7/site-packages/torch/share/cmake/Torch
make VERBOSE=1
cd -
```

run the test script:
```
CUDA_VISIBLE_DEVICES=0 python test.py
```

**note**:
- to use fp32/fp64, must compile with arch >= 50
- to use fp16, must compile with arch >= 75
- matrix sizes of fp16 must be aligned with 8

**todo**:
- [ ] support group_gemm pytorch function (forward + backward)
- [ ] support tf32 for high-end gpu
- [ ] support gemm merged with softmax (see [here](https://github.com/NVIDIA/cutlass/tree/master/examples/35_gemm_softmax))
- [ ] support gemm merged with gather-scatter (see [here](https://github.com/NVIDIA/cutlass/tree/master/examples/36_gather_scatter_fusion))
