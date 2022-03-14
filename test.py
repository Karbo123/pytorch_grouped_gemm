import sys
import time
import torch
from random import randint
sys.path.append("build")
import PYTORCH_GROUPED_GEMM

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# test correctness

A, B, C, D = [[torch.randn([1024, 1024], dtype=torch.half, device="cuda"), 
               torch.randn([512, 512], dtype=torch.half, device="cuda")] for _ in range(4)]
Ac = [x.clone() for x in A]
Bc = [x.clone() for x in B]
Cc = [x.clone() for x in C]
Dc = [x.clone() for x in D]

alpha, beta = 1.0, 0.0
PYTORCH_GROUPED_GEMM.GroupedGEMM(Ac, Bc, Cc, Dc, alpha, beta)

Dlst = list()
for a, b, c in zip(A, B, C):
    Dlst.append((alpha * a.t() @ b.t() + beta * c.t()).t())

# check
for cutlass, ptgemm in zip(Dc, Dlst):
    err = (cutlass - ptgemm).abs()
    print(f"abs err max = {err.max():.3e}")
    print(f"abs err mean = {err.mean():.3e}")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# test speed

batch_size = 1024

As = list()
Bs = list()
Cs = list()
Ds = list()
for _ in range(batch_size):
    m, n, k = [randint(32, 128) * 8 for _ in range(3)]
    As.append(torch.randn([m, k], dtype=torch.half, device="cuda"))
    Bs.append(torch.randn([k, n], dtype=torch.half, device="cuda"))
    Cs.append(torch.randn([m, n], dtype=torch.half, device="cuda"))
    Ds.append(torch.randn([m, n], dtype=torch.half, device="cuda"))
As_copy = [x.clone() for x in As]
Bs_copy = [x.clone() for x in Bs]
Cs_copy = [x.clone() for x in Cs]
Ds_copy = [x.clone() for x in Ds]

alpha, beta = 1.0, 0.0

# cutlass's method
torch.cuda.synchronize()
t0 = time.time()
PYTORCH_GROUPED_GEMM.GroupedGEMM(As_copy, Bs_copy, Cs_copy, Ds_copy, alpha, beta)
torch.cuda.synchronize()
t1 = time.time()
print(f"time for cutlass = {t1 - t0}")

# pytorch's method
D_collection = list()
linear_fn = torch.nn.functional.linear
torch.cuda.synchronize()
t0 = time.time()
for A, B, C in zip(As, Bs, Cs):
    D_collection.append(A @ B + C)
torch.cuda.synchronize()
t1 = time.time()
print(f"time for pytorch = {t1 - t0}")


"""
abs err max = 0.000e+00
abs err mean = 0.000e+00
abs err max = 3.125e-02
abs err mean = 7.093e-06
time for cutlass = 0.020124435424804688
time for pytorch = 0.15853619575500488

"""

