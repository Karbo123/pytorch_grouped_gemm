import sys
import time
import torch
import random
sys.path.append("build")
import PYTORCH_GROUPED_GEMM

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def prepare_data(bs=8, mul_min=32, mul_max=128, mul=8, dtype=torch.half):
    As, Bs, Cs, Ds = [list() for _ in range(4)]
    get = lambda *shape: torch.randn([*shape], dtype=dtype, device="cuda")
    random.seed(0)
    for _ in range(bs):
        m, n, k = [random.randint(mul_min, mul_max) * mul for _ in range(3)]
        As.append(get(m, k))
        Bs.append(get(k, n))
        Cs.append(get(m, n))
        Ds.append(get(m, n))
    return As, Bs, Cs, Ds

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# test correctness
def test_correctness(kwargs_prepare=dict(bs=8, mul_min=32, mul_max=128, mul=8),
                     kwargs_scale=dict(alpha=1.0, beta=0.0),
                     dtype_lst=[torch.half, torch.float32]):
    for dtype in dtype_lst:
        As, Bs, Cs, cutlass_result = prepare_data(**kwargs_prepare, dtype=dtype)
        alpha, beta = kwargs_scale["alpha"], kwargs_scale["beta"]

        pytorch_result = list()
        for A, B, C in zip(As, Bs, Cs):
            pytorch_result.append(alpha * A @ B + beta * C)

        PYTORCH_GROUPED_GEMM.GroupedGEMM(As, Bs, Cs, cutlass_result, alpha, beta)

        # check
        THRES = 1e-2
        for cutlass_res, pytorch_res in zip(cutlass_result, pytorch_result):
            error = (cutlass_res.float() - pytorch_res.float()).abs()
            error_bound = torch.maximum(cutlass_res.float().abs(), pytorch_res.float().abs())
            relative_error = (error / error_bound).mean()
            assert relative_error < THRES, f"relative error {relative_error:.3e} (>= {THRES}) is too large for dtype = {dtype}"
    print("correctness test passed!")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# test speed
def test_speed(kwargs_prepare=dict(bs=8, mul_min=32, mul_max=128, mul=8),
               kwargs_scale=dict(alpha=1.0, beta=0.0),
               try_times=10,
               dtype=torch.half,
            ):
    As, Bs, Cs, cutlass_result = prepare_data(**kwargs_prepare, dtype=dtype)
    alpha, beta = kwargs_scale["alpha"], kwargs_scale["beta"]

    # pytorch's method
    pytorch_result = list()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(try_times):
        for A, B, C in zip(As, Bs, Cs):
            pytorch_result.append(alpha * A @ B + beta * C)
    torch.cuda.synchronize(); t1 = time.time()
    print(f"time for pytorch = {t1 - t0}")

    # cutlass's method
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(try_times):
        PYTORCH_GROUPED_GEMM.GroupedGEMM(As, Bs, Cs, cutlass_result, alpha, beta)
    torch.cuda.synchronize(); t1 = time.time()
    print(f"time for cutlass = {t1 - t0}")

if __name__ == "__main__":
    test_correctness(kwargs_prepare=dict(bs=8, mul_min=32, mul_max=128, mul=8),
                     kwargs_scale=dict(alpha=1.0, beta=1.0),
                     dtype_lst=[torch.half, torch.float, torch.float64])
    
    for dtype in (torch.half, torch.float32, torch.float64):
        print()
        print(f"testing speed for {dtype}")
        test_speed(kwargs_prepare=dict(bs=8192, mul_min=1, mul_max=16, mul=8),
                   kwargs_scale=dict(alpha=1.0, beta=1.0),
                   try_times=10, dtype=dtype)

    """
    correctness test passed!

    testing speed for torch.float16
    time for pytorch = 3.490262269973755
    time for cutlass = 0.07040643692016602

    testing speed for torch.float32
    time for pytorch = 3.4165427684783936
    time for cutlass = 0.10037946701049805

    testing speed for torch.float64
    time for pytorch = 3.352168083190918
    time for cutlass = 0.41443443298339844

    """

