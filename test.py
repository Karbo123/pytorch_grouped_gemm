import sys
import time
import torch
import random
sys.path.append("build")
import PYTORCH_GROUPED_GEMM

# for high-end gpu, it defaults to tf32 instead of fp32
from packaging import version
if version.parse(torch.__version__) >= version.parse("1.7"):
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
    torch.backends.cuda.matmul.allow_tf32 = False

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
        THRES = {torch.float16: 3e-4, torch.float32: 1e-5, torch.float64: 1e-14}[dtype]
        for cutlass_res, pytorch_res in zip(cutlass_result, pytorch_result):
            error = (cutlass_res.double() - pytorch_res.double()).abs()
            error_bound = torch.maximum(cutlass_res.double().abs(), pytorch_res.double().abs())
            relative_error = (error / error_bound).mean()
            print(f"[{dtype}] relative_error = {relative_error.item():.3e}, abs_error = {error.mean().item():.3e}")
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

    """ Tested on a single 3090 GPU
    [torch.float16] relative_error = 2.410e-04, abs_error = 2.441e-03
    [torch.float16] relative_error = 2.099e-04, abs_error = 3.844e-03
    [torch.float16] relative_error = 2.141e-04, abs_error = 3.828e-03
    [torch.float16] relative_error = 2.258e-04, abs_error = 3.049e-03
    [torch.float16] relative_error = 2.193e-04, abs_error = 3.279e-03
    [torch.float16] relative_error = 2.256e-04, abs_error = 2.638e-03
    [torch.float16] relative_error = 2.109e-04, abs_error = 3.966e-03
    [torch.float16] relative_error = 2.248e-04, abs_error = 2.812e-03
    [torch.float32] relative_error = 1.708e-06, abs_error = 4.193e-06
    [torch.float32] relative_error = 5.444e-06, abs_error = 1.017e-05
    [torch.float32] relative_error = 2.794e-06, abs_error = 1.010e-05
    [torch.float32] relative_error = 0.000e+00, abs_error = 0.000e+00
    [torch.float32] relative_error = 2.171e-06, abs_error = 7.692e-06
    [torch.float32] relative_error = 1.768e-06, abs_error = 4.977e-06
    [torch.float32] relative_error = 2.503e-06, abs_error = 1.083e-05
    [torch.float32] relative_error = 1.964e-06, abs_error = 5.143e-06
    [torch.float64] relative_error = 0.000e+00, abs_error = 0.000e+00
    [torch.float64] relative_error = 0.000e+00, abs_error = 0.000e+00
    [torch.float64] relative_error = 3.795e-15, abs_error = 1.619e-14
    [torch.float64] relative_error = 3.415e-15, abs_error = 1.022e-14
    [torch.float64] relative_error = 0.000e+00, abs_error = 0.000e+00
    [torch.float64] relative_error = 0.000e+00, abs_error = 0.000e+00
    [torch.float64] relative_error = 0.000e+00, abs_error = 0.000e+00
    [torch.float64] relative_error = 0.000e+00, abs_error = 0.000e+00
    correctness test passed!

    testing speed for torch.float16
    time for pytorch = 3.5916242599487305
    time for cutlass = 0.1151578426361084

    testing speed for torch.float32
    time for pytorch = 3.5333731174468994
    time for cutlass = 0.14151287078857422

    testing speed for torch.float64
    time for pytorch = 3.743123769760132
    time for cutlass = 0.45430731773376465
    """
