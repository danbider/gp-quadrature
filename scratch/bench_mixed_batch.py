"""
Test if mixed batch sizes (like in a real CG pipeline) harm the smooth FFT
more than pow2 (e.g. FFT plan cache misses on non-pow2 sizes).

Simulate: alternate bs=1 and bs=5 calls.
"""
import sys, time
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path: sys.path.insert(0, str(_ROOT))
import numpy as np
import torch
from efgpnd import ToeplitzND, compute_convolution_vector_vectorized_dD


def bench_mixed(M, trace_iters=50, mean_iters=5):
    torch.manual_seed(0)
    cdtype = torch.complex128
    rdtype = torch.float64
    d = 2
    x = torch.rand(20_000, d, dtype=rdtype)
    h = 1.0 / (2 * M)
    m_conv = (M - 1) // 2
    v = compute_convolution_vector_vectorized_dD(m_conv, x, h).to(cdtype)

    for fp in (True, False):
        top = ToeplitzND(v.clone(), force_pow2=fp)
        size = int(np.prod(top.ns))
        beta1 = torch.randn(size, dtype=cdtype).view(*top.ns)       # bs=1 shape
        beta5 = torch.randn(5, size, dtype=cdtype).view(5, *top.ns) # bs=5 shape

        # warmup both shapes
        for _ in range(3):
            top(beta1); top(beta5)

        # interleaved calls (like a real grad step alternating mean+trace CG)
        t0 = time.perf_counter()
        for _ in range(mean_iters):
            top(beta1)
        for _ in range(trace_iters):
            top(beta5)
        t_interleaved = time.perf_counter() - t0

        # same total work but without shape switching
        t0 = time.perf_counter()
        for _ in range(mean_iters + trace_iters):
            top(beta5)
        t_uniform = time.perf_counter() - t0

        print(
            f"M={M:<4} fp={fp!s:<5} | interleaved={t_interleaved*1e3:7.1f}ms"
            f" uniform(bs=5)={t_uniform*1e3:7.1f}ms"
        )


if __name__ == "__main__":
    for M in (39, 89, 175, 257):
        bench_mixed(M)
