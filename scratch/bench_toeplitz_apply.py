"""
Two benchmarks, d=2 headline (with d=3 sanity points):
  (A) force_pow2 True vs False — setup + per-apply.
  (B) Two-NUFFT apply vs Toeplitz apply, sweeping N.

Tuned to realistic EFGP params (eps ~1e-3/1e-4 regime). We test two nufft_eps:
  - 6e-8 (repo default)
  - 1e-4 (loose, aligned with cg_tol)

Run: ~/myenv/bin/python scratch/bench_toeplitz_apply.py
"""
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from efgpnd import ToeplitzND, NUFFT, compute_convolution_vector_vectorized_dD


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_fn(fn, device, warmup=2, iters=5):
    for _ in range(warmup):
        fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    return (time.perf_counter() - t0) / iters


def bench_pow2(d, M_list, N_points, device, iters=5):
    device = torch.device(device)
    rdtype = torch.float64
    cdtype = torch.complex128
    rows = []
    for M in M_list:
        x = torch.rand(N_points, d, dtype=rdtype, device=device)
        h = 1.0 / (2 * M)
        m_conv = (M - 1) // 2
        v = compute_convolution_vector_vectorized_dD(m_conv, x, h).to(cdtype)

        res = {}
        for fp in (True, False):
            # setup
            _sync(device)
            t0 = time.perf_counter()
            for _ in range(2):
                top = ToeplitzND(v.clone(), force_pow2=fp)
            _sync(device)
            t_setup = (time.perf_counter() - t0) / 2

            # apply
            top = ToeplitzND(v, force_pow2=fp)
            beta = torch.randn(int(np.prod(top.ns)), dtype=cdtype, device=device)
            t_apply = time_fn(lambda: top(beta), device, warmup=2, iters=iters)
            res[fp] = (list(top.fft_shape), t_setup, t_apply)

        row = dict(
            d=d, M=M, N=N_points,
            pow2_fft=res[True][0], smooth_fft=res[False][0],
            setup_pow2_ms=res[True][1]*1e3, setup_smooth_ms=res[False][1]*1e3,
            apply_pow2_ms=res[True][2]*1e3, apply_smooth_ms=res[False][2]*1e3,
            apply_speedup=res[True][2]/res[False][2],
        )
        rows.append(row)
        print(
            f"d={d} M={M:4d} | pow2={row['pow2_fft']} smooth={row['smooth_fft']}"
            f" | apply pow2={row['apply_pow2_ms']:7.2f}ms smooth={row['apply_smooth_ms']:7.2f}ms"
            f" speedup={row['apply_speedup']:.2f}x"
            f" | setup pow2={row['setup_pow2_ms']:6.1f} smooth={row['setup_smooth_ms']:6.1f}"
        )
    return rows


def bench_two_nufft(d, M, N_list, device, nufft_eps, iters=5):
    """M must be odd (matches mtot = 2*m_conv+1 convention in efgpnd)."""
    assert M % 2 == 1, f"M must be odd, got {M}"
    device = torch.device(device)
    rdtype = torch.float64
    cdtype = torch.complex128

    rows = []
    for N in N_list:
        x = torch.rand(N, d, dtype=rdtype, device=device)
        h = 1.0 / (2 * M)
        xcen = torch.zeros(d, device=device, dtype=rdtype)
        nufft_op = NUFFT(x, xcen, h, nufft_eps, cdtype=cdtype, device=device)

        m_conv = (M - 1) // 2
        v = compute_convolution_vector_vectorized_dD(m_conv, x, h).to(cdtype)
        top_sm = ToeplitzND(v, force_pow2=False)
        ns = tuple(top_sm.ns)  # authoritative grid size per dim
        OUT = ns

        mtot_d = int(np.prod(OUT))
        beta_flat = torch.randn(mtot_d, dtype=cdtype, device=device)
        beta_block = beta_flat.view(*OUT)

        def two_nufft():
            vals = nufft_op.type2(beta_block)
            return nufft_op.type1(vals, out_shape=OUT)

        def toep_sm():
            return top_sm(beta_flat)

        t_nf = time_fn(two_nufft, device, warmup=2, iters=iters)
        t_sm = time_fn(toep_sm, device, warmup=2, iters=iters)

        y_nf = two_nufft().reshape(-1)
        y_tp = toep_sm().reshape(-1)
        rel = (y_nf - y_tp).norm() / y_tp.norm()

        row = dict(
            d=d, M=M, N=N, nufft_eps=nufft_eps,
            two_nufft_ms=t_nf*1e3, toep_smooth_ms=t_sm*1e3,
            speedup=t_sm/t_nf, rel_err=float(rel),
        )
        rows.append(row)
        print(
            f"d={d} M={M:4d} N={N:7d} eps={nufft_eps:.0e}"
            f" | 2NUFFT={row['two_nufft_ms']:7.2f}ms Toep(smooth)={row['toep_smooth_ms']:7.2f}ms"
            f" speedup={row['speedup']:.2f}x rel_err={row['rel_err']:.1e}"
        )
    return rows


if __name__ == "__main__":
    device = "cpu"
    out = {}
    skip_A = "--skip-a" in sys.argv

    if not skip_A:
        print("=== (A) force_pow2 True vs False ===")
        out["pow2_d2"] = bench_pow2(2, [100, 129, 200, 257, 300, 513, 800], N_points=20_000, device=device)
        out["pow2_d3"] = bench_pow2(3, [40, 64, 65, 100, 129], N_points=20_000, device=device)

    print("\n=== (B) Two-NUFFT vs Toeplitz(smooth) — d=2 ===")
    for eps in (6e-8, 1e-4):
        key = f"twonufft_d2_M201_eps{eps:.0e}"
        out[key] = bench_two_nufft(2, 201, [1_000, 10_000, 100_000, 500_000], device=device, nufft_eps=eps)
        key = f"twonufft_d2_M513_eps{eps:.0e}"
        out[key] = bench_two_nufft(2, 513, [1_000, 10_000, 100_000, 500_000], device=device, nufft_eps=eps)

    print("\n=== (B) Two-NUFFT vs Toeplitz(smooth) — d=3 ===")
    for eps in (6e-8, 1e-4):
        key = f"twonufft_d3_M65_eps{eps:.0e}"
        out[key] = bench_two_nufft(3, 65, [1_000, 10_000, 100_000], device=device, nufft_eps=eps)
        key = f"twonufft_d3_M101_eps{eps:.0e}"
        out[key] = bench_two_nufft(3, 101, [1_000, 10_000, 100_000], device=device, nufft_eps=eps)

    out_path = Path(__file__).with_suffix(".json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved results to {out_path}")
