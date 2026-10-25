"""Is the GPU healthy? Raw peaks with workloads that saturate it: large matmuls (TFLOP/s) and a large copy (GB/s).

RTX 3060 (12 GB) specs, approximately: fp32 12.7 TFLOP/s; tensor cores fp16/bf16 ~25 TFLOP/s dense
(51 with sparsity); TF32 ~12.7 dense; memory bandwidth 360 GB/s.
  PYTHONPATH=. .venv/bin/python experiments/25-10-26/a-gumbel/nn/gpu_peak.py
"""
import time

import torch


def timed(fn, calls):
    fn(); torch.cuda.synchronize()
    t = time.time()
    for _ in range(calls): fn()
    torch.cuda.synchronize()
    return (time.time() - t) / calls


print(torch.cuda.get_device_name())
n = 8192
for name, dtype, tf32 in (('fp32', torch.float32, False), ('tf32', torch.float32, True),
                          ('bf16', torch.bfloat16, False), ('fp16', torch.float16, False)):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    a, b = torch.randn(n, n, device='cuda', dtype=dtype), torch.randn(n, n, device='cuda', dtype=dtype)
    seconds = timed(lambda: a @ b, 10)
    print(f'matmul {n}x{n} {name}: {2 * n ** 3 / seconds / 1e12:5.1f} TFLOP/s')

x = torch.empty(256 * 2 ** 20, device='cuda')  # 1 GiB
y = torch.empty_like(x)
seconds = timed(lambda: y.copy_(x), 20)
print(f'copy 1 GiB: {2 * x.numel() * 4 / seconds / 1e9:5.0f} GB/s (read + write)')
