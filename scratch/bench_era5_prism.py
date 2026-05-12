"""
Capture Kron vs Jacobi on ERA5 and PRISM to JSON, so the numbers persist.

Based on scratch_more_real_data_kron_vs_jacobi.py (ERA5) and
scratch_prism_kron_vs_jacobi.py (PRISM).

Run: ~/myenv/bin/python -u scratch/bench_era5_prism.py
"""
from __future__ import annotations
import sys, os, time, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "real" / "era5"))
sys.path.insert(0, str(ROOT / "experiments" / "real" / "prism"))

import torch
import numpy as np

from efgpnd import EFGPND
from kernels.squared_exponential import SquaredExponential

torch.set_default_dtype(torch.float64)
DT = torch.float64

PRISM_DIR = "/Users/colecitrenbaum/Documents/GPs/prism_tmean_us_30s_202602"


def normalize(x):
    mn = x.min(dim=0).values
    mx = x.max(dim=0).values
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


def load_prism_xy(n_sub=1_000_000):
    from load_prism import load_prism_dataset
    x_np, y_np = load_prism_dataset(PRISM_DIR, n_sub=n_sub, seed=0)
    x = normalize(torch.from_numpy(x_np.astype(np.float64)))
    y = standardize(torch.from_numpy(y_np.astype(np.float64)))
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


def _avg(xs):
    xs = [v for v in xs if v is not None]
    return sum(xs) / len(xs) if xs else None


def run(name, x, y, hypers_list):
    print(f"\n=== {name}: N={x.shape[0]:,}, d={x.shape[1]} ===", flush=True)
    out = {"name": name, "N": int(x.shape[0]), "d": int(x.shape[1]),
           "cases": []}
    for ls, var, sig2 in hypers_list:
        case = {"ls": ls, "variance": var, "sigmasq": sig2}
        for precond in ["kronecker", "jacobi"]:
            try:
                r = time_grad(x, y, precond, ls=ls, var=var, sig2=sig2)
                t_mean = sum(r['times']) / len(r['times'])
                case[precond] = {
                    "times": r['times'],
                    "time_mean": t_mean,
                    "iters": r['iters'],
                    "iters_mean": _avg(r['iters']),
                    "M": r['M'],
                }
            except Exception as e:
                case[precond] = {"error": f"{type(e).__name__}: {e}"}
                print(f"  {precond}: FAILED {type(e).__name__}: {e}",
                      flush=True)
        if "error" not in case.get("kronecker", {}) and \
           "error" not in case.get("jacobi", {}):
            rk, rj = case["kronecker"], case["jacobi"]
            tk, tj = rk["time_mean"], rj["time_mean"]
            ck, cj = rk["iters_mean"], rj["iters_mean"]
            M = rk["M"]
            m_dim = int(round(M ** 0.5))
            winner = "KRON" if tk < tj else "JAC "
            case["winner"] = winner.strip()
            case["speedup_kron_over_jac"] = tj / tk
            print(f"  ℓ={ls:<5g}  M={M} (m={m_dim}, "
                  f"N/m²={x.shape[0]/m_dim**2:.1f})", flush=True)
            print(f"    kron  : {tk:.2f}s, cg={ck:.0f}  "
                  f"jacobi: {tj:.2f}s, cg={cj:.0f}  "
                  f"-> {winner} ({tj/tk:.2f}x)", flush=True)
        out["cases"].append(case)
    return out


def main():
    print("ERA5 + PRISM benchmark, eps=1e-3, cg_tol=1e-4, K=3 + 1 warmup",
          flush=True)
    results = []

    print("\nLoading ERA5...", flush=True)
    t0 = time.perf_counter()
    x, y = load_era5_xy()
    print(f"  N={x.shape[0]:,} in {time.perf_counter()-t0:.1f}s", flush=True)
    results.append(run("ERA5 t2m", x, y, [
        (0.02, 1.0, 1e-2),
        (0.01, 1.0, 1e-2),
    ]))
    del x, y

    print("\nLoading PRISM (sub 1M)...", flush=True)
    t0 = time.perf_counter()
    x, y = load_prism_xy(n_sub=1_000_000)
    print(f"  N={x.shape[0]:,} in {time.perf_counter()-t0:.1f}s", flush=True)
    results.append(run("PRISM tmean (sub 1M)", x, y, [
        (0.02, 1.0, 1e-2),
        (0.01, 1.0, 1e-2),
    ]))

    outp = HERE / "bench_era5_prism.json"
    outp.write_text(json.dumps(results, indent=2))
    print(f"\nwrote -> {outp}", flush=True)


if __name__ == "__main__":
    main()
