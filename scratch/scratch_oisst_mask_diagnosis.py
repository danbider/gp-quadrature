"""
Diagnose why Jacobi beat Kron on OISST despite N/m² being large.

Three subsets of OISST anom, all normalized to [0,1]^2:
  (a) FULL   — all 691k valid pixels (with land mask, strong non-product)
  (b) OCEAN  — Pacific box (lon ∈ [-160, -110], lat ∈ [-20, 20]),
                almost entirely ocean → close to product
  (c) RAND   — full dataset randomly subsampled to 100k (breaks grid
                regularity but keeps the mask geometry intact)

If Kron wins (b) but loses (a) and (c) → it's the mask.
If Kron wins (b) and (c) but loses (a) → something more subtle.

Run: ~/myenv/bin/python -u scratch/scratch_oisst_mask_diagnosis.py
"""
from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..',
                                                 'experiments', 'real', 'oisst')))

import torch
import numpy as np
from efgpnd import EFGPND
from kernels.squared_exponential import SquaredExponential
from load_oisst import load_oisst_grid, load_oisst

torch.set_default_dtype(torch.float64)
DT = torch.float64


def normalize(x):
    mn = x.min(dim=0).values; mx = x.max(dim=0).values
    return (x - mn) / (mx - mn)


def standardize(y):
    return (y - y.mean()) / y.std()


def load_full():
    x, y = load_oisst(variable="anom")
    x = normalize(torch.from_numpy(x.astype(np.float64)))
    y = standardize(torch.from_numpy(y.astype(np.float64)))
    return x, y


def load_ocean_box(lon_lo=-160., lon_hi=-110., lat_lo=-20., lat_hi=20.):
    lon_grid, lat_grid, vals, _ = load_oisst_grid(variable="anom")
    in_box = ((lon_grid >= lon_lo) & (lon_grid <= lon_hi) &
              (lat_grid >= lat_lo) & (lat_grid <= lat_hi))
    valid = np.isfinite(vals) & in_box
    lon = lon_grid[valid]; lat = lat_grid[valid]; v = vals[valid]
    frac = valid.sum() / in_box.sum()
    print(f"  ocean box fill rate: {frac*100:.1f}% (1.0 = no mask at all)",
          flush=True)
    x = np.column_stack([lon, lat])
    x = normalize(torch.from_numpy(x.astype(np.float64)))
    y = standardize(torch.from_numpy(v.astype(np.float64)))
    return x, y


def load_rand_sub(n_sub=100_000, seed=0):
    x, y = load_oisst(variable="anom", n_sub=n_sub, seed=seed)
    x = normalize(torch.from_numpy(x.astype(np.float64)))
    y = standardize(torch.from_numpy(y.astype(np.float64)))
    return x, y


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


def run(label, x, y, hypers_list):
    print(f"\n=== {label}: N={x.shape[0]:,}, d={x.shape[1]} ===", flush=True)
    for ls, var, sig2 in hypers_list:
        ran = {}; M = None
        for precond in ["kronecker", "jacobi"]:
            try:
                r = time_grad(x, y, precond, ls=ls, var=var, sig2=sig2)
                ran[precond] = r; M = r['M']
            except Exception as e:
                print(f"    {precond}: FAILED {e}", flush=True)
        if "kronecker" in ran and "jacobi" in ran:
            rk, rj = ran["kronecker"], ran["jacobi"]
            tk = sum(rk['times']) / len(rk['times'])
            tj = sum(rj['times']) / len(rj['times'])
            ck = mean(rk['iters']); cj = mean(rj['iters'])
            m_dim = int(round(M ** 0.5))
            winner = "KRON" if tk < tj else "JAC "
            print(f"  ℓ={ls}  M={M} (m={m_dim}, N/m²={x.shape[0]/m_dim**2:.1f})",
                  flush=True)
            print(f"    kron   : {tk:.2f}s, cg={ck:.0f}  "
                  f"jacobi : {tj:.2f}s, cg={cj:.0f}  "
                  f"-> {winner} ({tj/tk:.2f}x)", flush=True)


HYPERS = [
    (0.02, 1.0, 1e-2),
    (0.01, 1.0, 1e-2),
]


if __name__ == "__main__":
    print("OISST mask diagnosis: full vs ocean-box vs random-subsample",
          flush=True)
    print("eps=1e-3, cg_tol=1e-4, K=3 + 1 warmup, pinned hypers", flush=True)

    print("\n-- (a) FULL OISST (strong mask) --", flush=True)
    xa, ya = load_full()
    run("FULL", xa, ya, HYPERS)
    del xa, ya

    print("\n-- (b) OCEAN BOX (mask-free Pacific) --", flush=True)
    xb, yb = load_ocean_box()
    run("OCEAN", xb, yb, HYPERS)
    del xb, yb

    print("\n-- (c) RANDOM SUBSAMPLE of full OISST --", flush=True)
    xc, yc = load_rand_sub(n_sub=100_000)
    run("RAND100k", xc, yc, HYPERS)
