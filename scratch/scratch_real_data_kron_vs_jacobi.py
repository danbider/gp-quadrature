"""
Kron vs Jacobi on real-world datasets: OISST (N≈691k) and CO2 (N≈1.44M).

Both are 2D spatial datasets. OISST is a near-uniform grid with land mask
(gappy but close to product-measure). CO2 is an OCO-2 satellite swath — very
non-uniform track geometry (anti-product-measure).

Recall from synthetic sweeps: Kron's quality depends on how close the joint
measure is to product measure, and on N > m^d with enough slack. OISST should
favor Kron (grid ≈ product). CO2 may stress Kron (tracks are strongly coupled).

For each (dataset, ℓ): run compute_gradients() K times with both preconds at
pinned hypers, report wall/iter.

Run: ~/myenv/bin/python -u scratch/scratch_real_data_kron_vs_jacobi.py
"""
from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'experiments', 'real', 'oisst')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'experiments', 'real', 'co2')))

import torch
import numpy as np
from efgpnd import EFGPND
from kernels.squared_exponential import SquaredExponential

torch.set_default_dtype(torch.float64)
DT = torch.float64


def normalize_to_unit_box(x: torch.Tensor) -> torch.Tensor:
    x = x.to(DT)
    mn = x.min(dim=0).values
    mx = x.max(dim=0).values
    return (x - mn) / (mx - mn)


def standardize(y: torch.Tensor) -> torch.Tensor:
    y = y.to(DT)
    return (y - y.mean()) / y.std()


def load_oisst_xy(n_sub=None, variable="anom"):
    from load_oisst import load_oisst
    x, y = load_oisst(variable=variable, n_sub=n_sub, seed=0)
    x = normalize_to_unit_box(torch.from_numpy(x.astype(np.float64)))
    y = standardize(torch.from_numpy(y.astype(np.float64)))
    return x, y


def load_co2_xy(n_sub=None):
    from load_co2 import load_co2
    x, y = load_co2(n_sub=n_sub, seed=0)
    x = torch.from_numpy(x.T.astype(np.float64))  # (N, 2)
    x = normalize_to_unit_box(x)
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
    t_list, trace_iters, mean_iters = [], [], []
    for _ in range(K):
        t0 = time.perf_counter()
        model.compute_gradients(trace_samples=J, cg_tol=cg_tol,
                                noise_floor=noise_floor)
        t_list.append(time.perf_counter() - t0)
        s = model.last_gradient_stats
        trace_iters.append(s.get('trace_cg_iters'))
        mean_iters.append(s.get('mean_cg_iters'))
    return dict(times=t_list, trace_iters=trace_iters, mean_iters=mean_iters,
                M=model.last_gradient_stats.get('feature_count'))


def mean(xs):
    xs = [v for v in xs if v is not None]
    return sum(xs) / len(xs) if xs else float('nan')


def run_one(name, x, y, hypers, *, eps=1e-3):
    ls, var, sig2 = hypers
    N, d = x.shape
    print(f"\n=== {name}: N={N:,}, d={d}, ℓ={ls}, σ_f²={var}, σ²={sig2} ===",
          flush=True)
    ran = {}
    M = None
    for precond in ["kronecker", "jacobi"]:
        try:
            r = time_grad(x, y, precond, ls=ls, var=var, sig2=sig2, eps=eps)
            ran[precond] = r; M = r['M']
        except Exception as e:
            print(f"  {precond:<9s} FAILED: {type(e).__name__}: {e}", flush=True)
    if "kronecker" in ran and "jacobi" in ran:
        rk, rj = ran["kronecker"], ran["jacobi"]
        tk = sum(rk['times']) / len(rk['times'])
        tj = sum(rj['times']) / len(rj['times'])
        ck = mean(rk['trace_iters']); cj = mean(rj['trace_iters'])
        m_dim = int(round(M ** (1.0/d)))
        ratio = N / (m_dim ** d)
        winner = "KRON" if tk < tj else "JAC "
        print(f"  M={M} (m≈{m_dim}), N/m^d = {ratio:.1f}", flush=True)
        print(f"  kron   : {tk:.2f}s/iter  trace-CG={ck:.1f}", flush=True)
        print(f"  jacobi : {tj:.2f}s/iter  trace-CG={cj:.1f}", flush=True)
        print(f"  -> {winner} wins  (jac/kron = {tj/tk:.2f}x)", flush=True)


if __name__ == "__main__":
    print("Real-data benchmark: Kron vs Jacobi", flush=True)
    print("eps=1e-3, cg_tol=1e-4, K=3 trials + 1 warmup, pinned hypers\n",
          flush=True)

    # ---- OISST -----------------------------------------------------------
    print("Loading OISST (SST anomaly)...", flush=True)
    t0 = time.perf_counter()
    x_oisst, y_oisst = load_oisst_xy(variable="anom")
    print(f"  loaded in {time.perf_counter()-t0:.1f}s, "
          f"N={x_oisst.shape[0]}, d={x_oisst.shape[1]}", flush=True)

    # Spatial scale: features span [0,1]^2 after normalization.
    # Earth-scale anomaly has ~1000-km correlation -> ℓ ≈ 0.03 in [0,1]^2.
    for hypers in [
        (0.02, 1.0, 1e-2),   # moderate ℓ, moderate noise
        (0.01, 1.0, 1e-2),   # tighter ℓ (larger m)
        (0.02, 1.0, 1e-4),   # tight noise
    ]:
        run_one("OISST anom", x_oisst, y_oisst, hypers)

    # ---- CO2 -------------------------------------------------------------
    print("\nLoading CO2 (OCO-2)...", flush=True)
    t0 = time.perf_counter()
    x_co2, y_co2 = load_co2_xy()
    print(f"  loaded in {time.perf_counter()-t0:.1f}s, "
          f"N={x_co2.shape[0]}, d={x_co2.shape[1]}", flush=True)

    for hypers in [
        (0.02, 1.0, 1e-2),
        (0.01, 1.0, 1e-2),
        (0.02, 1.0, 1e-4),
    ]:
        run_one("CO2", x_co2, y_co2, hypers)
