"""
Kron vs Jacobi on ERA5 and spatial transcriptomics (adata.h5ad).

ERA5: ~1M gridded global t2m (sea+land+ice), structure similar to OISST.
Predict: Jacobi wins by a similar margin due to land-mask.

adata: ~30k spatial transcriptomics (tissue section). Different geometry —
      expect dense contiguous support (no big "mask"), but strong non-uniform
      density. Test case for "does Jacobi still win when there's no mask,
      but the sampling is heterogeneous?".

Run: ~/myenv/bin/python -u scratch/scratch_more_real_data_kron_vs_jacobi.py
"""
from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..',
                                                 'experiments', 'real', 'era5')))

import torch
import numpy as np
from efgpnd import EFGPND
from kernels.squared_exponential import SquaredExponential

torch.set_default_dtype(torch.float64)
DT = torch.float64


def normalize(x):
    mn = x.min(dim=0).values; mx = x.max(dim=0).values
    return (x - mn) / (mx - mn)


def standardize(y):
    y = y.to(DT)
    return (y - y.mean()) / y.std()


def load_era5_xy(n_sub=None):
    from load_era5 import load_era5
    x, y = load_era5(n_sub=n_sub, seed=0)
    x = normalize(torch.from_numpy(x.astype(np.float64)))
    y = standardize(torch.from_numpy(y.astype(np.float64)))
    return x, y


def load_adata_xy():
    import anndata
    a = anndata.read_h5ad("adata.h5ad")
    x = np.column_stack([a.obs["x"].values, a.obs["y"].values]).astype(np.float64)
    # Use total UMI as target (dense, non-trivial signal)
    if hasattr(a.X, "toarray"):
        total = np.asarray(a.X.sum(axis=1)).ravel()
    else:
        total = a.X.sum(axis=1)
    y = np.log1p(total.astype(np.float64))
    x = normalize(torch.from_numpy(x))
    y = standardize(torch.from_numpy(y))
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
    print("ERA5 + adata benchmarks, eps=1e-3, cg_tol=1e-4, K=3 + 1 warmup",
          flush=True)

    print("\nLoading ERA5...", flush=True)
    t0 = time.perf_counter()
    x, y = load_era5_xy()
    print(f"  N={x.shape[0]:,} in {time.perf_counter()-t0:.1f}s", flush=True)
    run("ERA5 t2m", x, y, [
        (0.02, 1.0, 1e-2),
        (0.01, 1.0, 1e-2),
    ])
    del x, y

    print("\nLoading adata (spatial transcriptomics)...", flush=True)
    t0 = time.perf_counter()
    x, y = load_adata_xy()
    print(f"  N={x.shape[0]:,} in {time.perf_counter()-t0:.1f}s", flush=True)
    run("adata tissue", x, y, [
        (0.03, 1.0, 1e-2),
        (0.01, 1.0, 1e-2),
    ])
