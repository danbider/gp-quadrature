"""Minimal Kron+LR test on OISST — single config, fast feedback."""
from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..',
                                                 'experiments', 'real', 'oisst')))

import torch
import numpy as np
import efgpnd as efgp_mod
from efgpnd import EFGPND, ToeplitzND
from kernels.squared_exponential import SquaredExponential
from load_oisst import load_oisst

# Import monkey-patch machinery from prototype file
from scratch_kron_lowrank_residual import (
    build_M_apply_and_inv, make_lowrank_wrapper, install_lowrank, uninstall,
    _patch_state,
)

torch.set_default_dtype(torch.float64)
DT = torch.float64


def normalize(x):
    mn = x.min(dim=0).values; mx = x.max(dim=0).values
    return (x - mn) / (mx - mn)


def load_full():
    x, y = load_oisst(variable="anom")
    x = normalize(torch.from_numpy(x.astype(np.float64)))
    y = torch.from_numpy(y.astype(np.float64))
    y = (y - y.mean()) / y.std()
    return x, y


def one_shot(x, y, precond_kind, *, ls, var, sig2, r=None,
             eps=1e-3, cg_tol=1e-4, cg_max=3000):
    """Single compute_gradients call; return wall and iter count."""
    d = x.shape[1]
    kernel = SquaredExponential(dimension=d, init_lengthscale=ls,
                                init_variance=var)
    model = EFGPND(x, y, kernel=kernel, sigmasq=sig2, eps=eps,
                   estimate_params=False,
                   opts={"mean_cg_preconditioner_type": precond_kind,
                         "max_cg_iterations": cg_max})
    t0 = time.perf_counter()
    model.compute_gradients(trace_samples=1, cg_tol=cg_tol, noise_floor=1e-5)
    dt = time.perf_counter() - t0
    s = model.last_gradient_stats
    return dt, s.get('trace_cg_iters'), s.get('mean_cg_iters'), s.get('feature_count')


if __name__ == "__main__":
    print("OISST full: minimal Kron+LR test\n", flush=True)
    t0 = time.perf_counter()
    x, y = load_full()
    print(f"Loaded OISST in {time.perf_counter()-t0:.1f}s: "
          f"N={x.shape[0]:,}, d={x.shape[1]}", flush=True)

    LS, VAR, SIG2 = 0.02, 1.0, 1e-2

    print(f"\n(ℓ={LS}, σ_f²={VAR}, σ²={SIG2}), eps=1e-3, cg_tol=1e-4\n",
          flush=True)

    uninstall()
    dt, it_t, it_m, M = one_shot(x, y, "kronecker", ls=LS, var=VAR, sig2=SIG2)
    print(f"  plain Kron                  : {dt:.2f}s  mean-cg={it_m}, trace-cg={it_t}  M={M}",
          flush=True)

    dt, it_t, it_m, _ = one_shot(x, y, "jacobi", ls=LS, var=VAR, sig2=SIG2)
    print(f"  Jacobi                      : {dt:.2f}s  mean-cg={it_m}, trace-cg={it_t}",
          flush=True)

    # Kron+LR (positive truncation)
    for r in [50, 200, 500, 1000]:
        install_lowrank(r=r, p=20)
        try:
            dt, it_t, it_m, _ = one_shot(x, y, "kronecker",
                                         ls=LS, var=VAR, sig2=SIG2)
            print(f"  Kron+LR(r={r:<5d}, pos-trunc): {dt:.2f}s  mean-cg={it_m}, trace-cg={it_t}",
                  flush=True)
            log = _patch_state.get('last_log', '')
            if log:
                print(f"      {log}", flush=True)
        except Exception as e:
            print(f"  Kron+LR(r={r}): FAILED {type(e).__name__}: {e}",
                  flush=True)
        finally:
            uninstall()
