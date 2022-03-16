# High Performance Grouped GEMM in PyTorch

computing multiple GEMMs with different matrix sizes

why: pytorch/cublas only support batched GEMM (use the same M, N, K), but does not support grouped GEMM (use different M, N, K for each matrix multiplication)

possible applications:
- transformer's attention matrix with different sizes; 
- convolution with different size for each batch (e.g. point clouds with different num of points)

performance overview:
```
testing speed for torch.float16
  time for pytorch = 3.490262269973755
  time for cutlass = 0.07040643692016602

testing speed for torch.float32
  time for pytorch = 3.4165427684783936
  time for cutlass = 0.10037946701049805

testing speed for torch.float64
  time for pytorch = 3.352168083190918
  time for cutlass = 0.41443443298339844
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

