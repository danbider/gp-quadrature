"""Kron vs Jacobi on PRISM tmean (CONUS gridded tmean, ~12M pts)."""
from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..',
                                                 'experiments', 'real', 'prism')))

import torch
import numpy as np
from efgpnd import EFGPND
from kernels.squared_exponential import SquaredExponential
from load_prism import load_prism_dataset

torch.set_default_dtype(torch.float64)
DT = torch.float64

PRISM_DIR = "/Users/colecitrenbaum/Documents/GPs/prism_tmean_us_30s_202602"


def normalize(x):
    mn = x.min(dim=0).values; mx = x.max(dim=0).values
    return (x - mn) / (mx - mn)


def standardize(y):
    y = y.to(DT)
    return (y - y.mean()) / y.std()


def time_grad(x, y, precond, *, ls, var, sig2, K=3, warmup=1,
              eps=1e-3, cg_tol=1e-4, J=1, noise_floor=1e-5, cg_max=3000):
    d = x.shape[1]
    kernel = SquaredExponential(dimension=d, init_lengthscale=ls,
                                init_variance=var)
    model = EFGPND(x, y, kernel=kernel, sigmasq=sig2, eps=eps,
                   estimate_params=False,
                   opts={"mean_cg_preconditioner_type": precond,
                         "max_cg_iterations": cg_max})
    for _ in range(warmup):
        model.compute_gradients(trace_samples=J, cg_tol=cg_tol,
                                noise_floor=noise_floor)
    t_list, iters = [], []
    for _ in range(K):
        t0 = time.perf_counter()
        model.compute_gradients(trace_samples=J, cg_tol=cg_tol,
                                noise_floor=noise_floor)
        t_list.append(time.perf_counter() - t0)
        iters.append(model.last_gradient_stats.get('trace_cg_iters'))
    return dict(times=t_list, iters=iters,
                M=model.last_gradient_stats.get('feature_count'))


def mean(xs):
    xs = [v for v in xs if v is not None]
    return sum(xs) / len(xs) if xs else float('nan')


def run(name, x, y, hypers_list):
    print(f"\n=== {name}: N={x.shape[0]:,}, d={x.shape[1]} ===", flush=True)
    for ls, var, sig2 in hypers_list:
        ran = {}; M = None
        for precond in ["kronecker", "jacobi"]:
            try:
                r = time_grad(x, y, precond, ls=ls, var=var, sig2=sig2)
                ran[precond] = r; M = r['M']
            except Exception as e:
                print(f"  {precond}: FAILED {type(e).__name__}: {e}",
                      flush=True)
        if "kronecker" in ran and "jacobi" in ran:
            rk, rj = ran["kronecker"], ran["jacobi"]
            tk = sum(rk['times']) / len(rk['times'])
            tj = sum(rj['times']) / len(rj['times'])
            ck = mean(rk['iters']); cj = mean(rj['iters'])
            m_dim = int(round(M ** 0.5))
            winner = "KRON" if tk < tj else "JAC "
            print(f"  ℓ={ls:<5g}  M={M} (m={m_dim}, N/m²={x.shape[0]/m_dim**2:.1f})",
                  flush=True)
            print(f"    kron  : {tk:.2f}s, cg={ck:.0f}  "
                  f"jacobi: {tj:.2f}s, cg={cj:.0f}  "
                  f"-> {winner} ({tj/tk:.2f}x)", flush=True)


if __name__ == "__main__":
    print("PRISM tmean benchmark", flush=True)
    t0 = time.perf_counter()
    # Subsample to 1M for comparability with ERA5/CO2/OISST
    x_np, y_np = load_prism_dataset(PRISM_DIR, n_sub=1_000_000, seed=0)
    x = normalize(torch.from_numpy(x_np.astype(np.float64)))
    y = standardize(torch.from_numpy(y_np.astype(np.float64)))
    print(f"Loaded PRISM (sub to 1M) in {time.perf_counter()-t0:.1f}s: "
          f"N={x.shape[0]:,}, d={x.shape[1]}", flush=True)

    run("PRISM tmean (sub 1M)", x, y, [
        (0.02, 1.0, 1e-2),
        (0.01, 1.0, 1e-2),
    ])
