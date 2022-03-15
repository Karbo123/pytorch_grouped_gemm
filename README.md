# High Performance Grouped GEMM in PyTorch

computing multiple GEMMs with different matrix sizes

why: pytorch/cublas only support batched GEMM (use the same M, N, K), but does not support grouped GEMM (use different M, N, K for each matrix multiplication)

possible applications:
- transformer's attention matrix with different sizes; 
- convolution with different size for each batch (e.g. point clouds with different num of poitns)

performance overview:
- half (fp16):
  - pytorch = 3.6461 sec
  - cutlass = 0.0759 sec

a build example:
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

**note**: currently the codes only support *ampere(>=80)* cuda architecture and *half(fp16)* precision

todo: 
- add fp32 and fp64 supports
- use other cuda arch
