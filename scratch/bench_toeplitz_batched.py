"""
Why does force_pow2=False lose INSIDE the CG batched pipeline when it wins
as a single-vector microbench? Test: ToeplitzND apply with batched inputs
matching CG shapes.
"""
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from efgpnd import ToeplitzND, compute_convolution_vector_vectorized_dD


def bench(d, M, batch_sizes, iters=10, warmup=3):
    torch.manual_seed(0)
    cdtype = torch.complex128
    rdtype = torch.float64
    x = torch.rand(20_000, d, dtype=rdtype)
    h = 1.0 / (2 * M)
    m_conv = (M - 1) // 2
    v = compute_convolution_vector_vectorized_dD(m_conv, x, h).to(cdtype)

    rows = []
    for bs in batch_sizes:
        out = {}
        for fp in (True, False):
            top = ToeplitzND(v.clone(), force_pow2=fp)
            size = int(np.prod(top.ns))
            if bs == 1:
                beta = torch.randn(size, dtype=cdtype)
            else:
                beta = torch.randn(bs, size, dtype=cdtype)
            for _ in range(warmup):
                top(beta)
            t0 = time.perf_counter()
            for _ in range(iters):
                top(beta)
            t = (time.perf_counter() - t0) / iters
            out[fp] = (list(top.fft_shape), t)
        print(
            f"d={d} M={M:<4} bs={bs:<4} | pow2 {out[True][0]} {out[True][1]*1e3:7.2f}ms"
            f" | smooth {out[False][0]} {out[False][1]*1e3:7.2f}ms"
            f" | speedup {out[True][1]/out[False][1]:.2f}x"
        )


if __name__ == "__main__":
    print("=== d=2 ===")
    for M in (39, 89, 175):
        bench(2, M, batch_sizes=[1, 2, 5, 10, 20])
    print("\n=== d=2 larger ===")
    for M in (257, 513):
        bench(2, M, batch_sizes=[1, 2, 5, 10])
