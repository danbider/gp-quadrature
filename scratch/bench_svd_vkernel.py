"""
Rank-r SVD-of-v_kernel Kronecker preconditioner on temperature datasets.

Replaces the origin-pinned slice in create_kronecker_precond with the
best rank-r Frobenius approximation of v_kernel (the empirical
characteristic function tensor). d=2 only. Hermitian symmetry on factors.

Run: ~/myenv/bin/python -u scratch/bench_svd_vkernel.py
"""
from __future__ import annotations
import sys, os, time, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "real" / "era5"))
sys.path.insert(0, str(ROOT / "experiments" / "real" / "prism"))
sys.path.insert(0, str(ROOT / "experiments" / "real" / "oisst"))

import torch
import numpy as np
import efgpnd as efgp_mod
from efgpnd import EFGPND
from kernels.squared_exponential import SquaredExponential

torch.set_default_dtype(torch.float64)
DT = torch.float64

PRISM_DIR = "/Users/colecitrenbaum/Documents/GPs/prism_tmean_us_30s_202602"

_ORIG_CREATE_KRON = efgp_mod.create_kronecker_precond
_state = {"installed": False, "last_info": ""}


def _herm_sym_1d(x):
    return 0.5 * (x + torch.flip(x, dims=[-1]).conj())


def _svd_rank_r_vkernel(v_kernel, r):
    assert v_kernel.ndim == 2, f"d=2 only, got ndim={v_kernel.ndim}"
    cdtype = v_kernel.dtype
    V = v_kernel
    U, S, Vh = torch.linalg.svd(V, full_matrices=False)
    r_use = min(r, S.numel())
    U_r = U[:, :r_use]
    S_r = S[:r_use]
    W_r = Vh[:r_use, :].conj()
    U_sym = torch.stack([_herm_sym_1d(U_r[:, k]) for k in range(r_use)], dim=1)
    W_sym = torch.stack([_herm_sym_1d(W_r[k, :]) for k in range(r_use)], dim=0)
    v_rank_r = (U_sym * S_r.to(cdtype)) @ W_sym.conj()
    info = {
        "r_use": r_use,
        "frob_full": float((S ** 2).sum().sqrt()),
        "frob_r": float((S[:r_use] ** 2).sum().sqrt()),
        "top_sing_ratio": (S[:min(5, S.numel())] / S[0]).detach().cpu().numpy().tolist(),
    }
    return v_rank_r.to(cdtype), info


def install_svd_rank(r):
    def patched(ws, v_kernel, sigmasq_scalar, d, mtot_1d,
                device=None, cdtype=torch.complex128, rdtype=torch.float64):
        if d != 2:
            return _ORIG_CREATE_KRON(ws, v_kernel, sigmasq_scalar, d, mtot_1d,
                                     device=device, cdtype=cdtype, rdtype=rdtype)
        v_rank_r, info = _svd_rank_r_vkernel(v_kernel.to(cdtype), r=r)
        keep_frac = info["frob_r"] / max(info["frob_full"], 1e-30)
        _state["last_info"] = {
            "r": r,
            "keep_frac": keep_frac,
            "top_sing_ratio": info["top_sing_ratio"],
        }
        return _ORIG_CREATE_KRON(ws, v_rank_r, sigmasq_scalar, d, mtot_1d,
                                 device=device, cdtype=cdtype, rdtype=rdtype)
    efgp_mod.create_kronecker_precond = patched
    _state["installed"] = True


def uninstall():
    efgp_mod.create_kronecker_precond = _ORIG_CREATE_KRON
    _state["installed"] = False
    _state["last_info"] = ""


def normalize(x):
    mn = x.min(dim=0).values
    mx = x.max(dim=0).values
    return (x - mn) / (mx - mn)


def standardize(y):
    y = y.to(DT)
    return (y - y.mean()) / y.std()


def load_oisst_full():
    from load_oisst import load_oisst
    x, y = load_oisst(variable="anom")
    x = normalize(torch.from_numpy(x.astype(np.float64)))
    y = standardize(torch.from_numpy(y.astype(np.float64)))
    return x, y


def load_era5_xy():
    from load_era5 import load_era5
    x, y = load_era5(n_sub=None, seed=0)
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


def run_dataset(name, x, y, configs, ranks=(1, 3, 10)):
    print(f"\n=== {name}: N={x.shape[0]:,}, d={x.shape[1]} ===", flush=True)
    ds = {"name": name, "N": int(x.shape[0]), "d": int(x.shape[1]),
          "cases": []}
    for ls, var, sig2 in configs:
        case = {"ls": ls, "variance": var, "sigmasq": sig2, "variants": {}}
        print(f"\n  (ℓ={ls}, σ_f²={var}, σ²={sig2})", flush=True)

        # Jacobi baseline
        uninstall()
        r = time_grad(x, y, "jacobi", ls=ls, var=var, sig2=sig2)
        t_mean = sum(r['times']) / len(r['times'])
        case["variants"]["jacobi"] = {
            "time_mean": t_mean, "iters_mean": _avg(r['iters']),
            "times": r['times'], "iters": r['iters'], "M": r['M'],
        }
        print(f"    jacobi        : {t_mean:.2f}s  cg={_avg(r['iters']):.0f}  M={r['M']}",
              flush=True)

        # Plain Kron
        uninstall()
        r = time_grad(x, y, "kronecker", ls=ls, var=var, sig2=sig2)
        t_mean = sum(r['times']) / len(r['times'])
        case["variants"]["kronecker"] = {
            "time_mean": t_mean, "iters_mean": _avg(r['iters']),
            "times": r['times'], "iters": r['iters'], "M": r['M'],
        }
        print(f"    kron (plain)  : {t_mean:.2f}s  cg={_avg(r['iters']):.0f}",
              flush=True)

        # SVD-of-v Kron at several ranks
        for rk in ranks:
            install_svd_rank(rk)
            try:
                r = time_grad(x, y, "kronecker", ls=ls, var=var, sig2=sig2)
                t_mean = sum(r['times']) / len(r['times'])
                case["variants"][f"svd_r{rk}"] = {
                    "time_mean": t_mean, "iters_mean": _avg(r['iters']),
                    "times": r['times'], "iters": r['iters'], "M": r['M'],
                    "info": dict(_state["last_info"]),
                }
                info = _state["last_info"]
                print(f"    SVDkron r={rk:<3d}  : {t_mean:.2f}s  "
                      f"cg={_avg(r['iters']):.0f}  "
                      f"||V_r||/||V||={info.get('keep_frac', float('nan')):.4f}",
                      flush=True)
            except Exception as e:
                case["variants"][f"svd_r{rk}"] = {
                    "error": f"{type(e).__name__}: {e}"}
                print(f"    SVDkron r={rk}: FAILED {type(e).__name__}: {e}",
                      flush=True)
            finally:
                uninstall()

        ds["cases"].append(case)
    return ds


def main():
    print("SVD-of-v Kron preconditioner bench, eps=1e-3, cg_tol=1e-4, K=3",
          flush=True)
    results = []

    for loader, label, configs in [
        (load_era5_xy, "ERA5 t2m",
         [(0.02, 1.0, 1e-2), (0.01, 1.0, 1e-2)]),
        (lambda: load_prism_xy(n_sub=1_000_000), "PRISM tmean (sub 1M)",
         [(0.02, 1.0, 1e-2), (0.01, 1.0, 1e-2)]),
        (load_oisst_full, "OISST anom",
         [(0.02, 1.0, 1e-2)]),
    ]:
        print(f"\nLoading {label}...", flush=True)
        t0 = time.perf_counter()
        try:
            x, y = loader()
        except Exception as e:
            print(f"  SKIP {label}: {type(e).__name__}: {e}", flush=True)
            continue
        print(f"  N={x.shape[0]:,} in {time.perf_counter()-t0:.1f}s",
              flush=True)
        results.append(run_dataset(label, x, y, configs))
        del x, y

    outp = HERE / "bench_svd_vkernel.json"
    outp.write_text(json.dumps(results, indent=2))
    print(f"\nwrote -> {outp}", flush=True)


if __name__ == "__main__":
    main()
