"""
Idea 1: replace marginal-slice symbols with leading-SVD symbols.

Plain Kron builds T_k from v_kernel[ctr, ..., :, ..., ctr] — the slice of V
through the origin. This is an implicit product-measure assumption:
V[l1, l2] ≈ v1(l1) v2(l2) with v_k = slice through origin.

Idea 1: compute SVD of V (viewed as (2m-1) x (2m-1)). The rank-1 truncation
σ1 u1(l1) w1(l2)^* is the best Frobenius rank-1 approx. Use u1 and w1 as
1D symbols.

Monkey-patch: replace v_kernel with its rank-1 SVD approximation before
calling create_kronecker_precond. The existing slicing then recovers the
SVD factors (up to the normalization the orig code applies).

Hermitian symmetry (u[-l] = conj(u[l])) is enforced on each factor so that
v_rank1 inherits the empirical-CF symmetry.

Run: ~/myenv/bin/python -u scratch/scratch_kron_svd_marginal.py
"""
from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..',
                                                 'experiments', 'real', 'oisst')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..',
                                                 'experiments', 'real', 'era5')))

import torch
import numpy as np
import efgpnd as efgp_mod
from efgpnd import EFGPND
from kernels.squared_exponential import SquaredExponential
from load_oisst import load_oisst

torch.set_default_dtype(torch.float64)
DT = torch.float64

_ORIG_CREATE_KRON = efgp_mod.create_kronecker_precond
_state = {"installed": False, "last_info": ""}


def _herm_sym_1d(x):
    """Enforce x[L-1-l] = conj(x[l]) so that index l<->L-1-l maps to frequency negation."""
    return 0.5 * (x + torch.flip(x, dims=[-1]).conj())


def _svd_rank_r_vkernel(v_kernel, r):
    """
    Compute leading-r SVD of V (d=2 only), enforce Hermitian symmetry on
    each factor, and return a rank-r reconstruction with the SAME shape
    as v_kernel. For r=1, yields σ1 u1 * conj(w1).
    """
    assert v_kernel.ndim == 2, f"this prototype is d=2 only, got ndim={v_kernel.ndim}"
    L1, L2 = v_kernel.shape
    cdtype = v_kernel.dtype
    V = v_kernel
    U, S, Vh = torch.linalg.svd(V, full_matrices=False)
    # V[i,j] = sum_k U[i,k] S[k] Vh[k,j]. Our rank-1 model was σ * u[i] * conj(w[j]).
    # So u_k = U[:,k], and conj(w_k) = Vh[k,:], i.e. w_k = conj(Vh[k,:]).
    r_use = min(r, S.numel())
    U_r = U[:, :r_use]
    S_r = S[:r_use]
    W_r = Vh[:r_use, :].conj()  # shape (r, L2)

    # Enforce Hermitian symmetry on each factor independently
    U_sym = torch.stack([_herm_sym_1d(U_r[:, k]) for k in range(r_use)], dim=1)
    W_sym = torch.stack([_herm_sym_1d(W_r[k, :]) for k in range(r_use)], dim=0)

    # Rebuild rank-r v: sum_k σ_k u_k * conj(w_k)
    # shape (L1, L2)
    v_rank_r = (U_sym * S_r.to(cdtype)) @ W_sym.conj()
    info = {
        "S": S.detach().cpu().numpy(),
        "r_use": r_use,
        "sing_ratio": (S[:r_use] / S[0]).detach().cpu().numpy().tolist(),
        "frob_full": float((S ** 2).sum().sqrt()),
        "frob_r": float((S[:r_use] ** 2).sum().sqrt()),
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
        top = info["sing_ratio"][:min(5, len(info["sing_ratio"]))]
        _state["last_info"] = (
            f"[SVDkron r={r}] ||V_r||/||V||={keep_frac:.4f}  "
            f"top σ_k/σ_1: {['{:.3g}'.format(s) for s in top]}"
        )
        return _ORIG_CREATE_KRON(ws, v_rank_r, sigmasq_scalar, d, mtot_1d,
                                 device=device, cdtype=cdtype, rdtype=rdtype)
    efgp_mod.create_kronecker_precond = patched
    _state["installed"] = True


def uninstall():
    efgp_mod.create_kronecker_precond = _ORIG_CREATE_KRON
    _state["installed"] = False
    _state["last_info"] = ""


def normalize(x):
    mn = x.min(dim=0).values; mx = x.max(dim=0).values
    return (x - mn) / (mx - mn)


def standardize(y):
    y = y.to(DT)
    return (y - y.mean()) / y.std()


def load_oisst_full():
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


def one_shot(x, y, precond_kind, *, ls, var, sig2,
             eps=1e-3, cg_tol=1e-4, cg_max=3000):
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


def run_dataset(name, x, y, configs):
    print(f"\n=== {name}: N={x.shape[0]:,}, d={x.shape[1]} ===", flush=True)
    for ls, var, sig2 in configs:
        print(f"\n  (ℓ={ls}, σ_f²={var}, σ²={sig2})", flush=True)

        uninstall()
        dt, it_t, it_m, M = one_shot(x, y, "kronecker", ls=ls, var=var, sig2=sig2)
        print(f"    plain Kron            : {dt:.2f}s  mean-cg={it_m}, trace-cg={it_t}  M={M}",
              flush=True)

        dt, it_t, it_m, _ = one_shot(x, y, "jacobi", ls=ls, var=var, sig2=sig2)
        print(f"    Jacobi                : {dt:.2f}s  mean-cg={it_m}, trace-cg={it_t}",
              flush=True)

        for r in [1, 3, 10, 30]:
            install_svd_rank(r)
            try:
                dt, it_t, it_m, _ = one_shot(x, y, "kronecker",
                                             ls=ls, var=var, sig2=sig2)
                print(f"    SVDkron(r={r:<3d})       : {dt:.2f}s  mean-cg={it_m}, trace-cg={it_t}",
                      flush=True)
                if _state["last_info"]:
                    print(f"      {_state['last_info']}", flush=True)
            except Exception as e:
                print(f"    SVDkron(r={r}): FAILED {type(e).__name__}: {e}",
                      flush=True)
            finally:
                uninstall()


if __name__ == "__main__":
    print("SVD-symbol Kron preconditioner (Idea 1)\n", flush=True)

    t0 = time.perf_counter()
    xo, yo = load_oisst_full()
    print(f"Loaded OISST in {time.perf_counter()-t0:.1f}s: "
          f"N={xo.shape[0]:,}, d={xo.shape[1]}", flush=True)
    run_dataset("OISST anom (full)", xo, yo, [
        (0.02, 1.0, 1e-2),
    ])
    del xo, yo

    t0 = time.perf_counter()
    xe, ye = load_era5_xy()
    print(f"\nLoaded ERA5 in {time.perf_counter()-t0:.1f}s: "
          f"N={xe.shape[0]:,}, d={xe.shape[1]}", flush=True)
    run_dataset("ERA5 t2m (full)", xe, ye, [
        (0.02, 1.0, 1e-2),
    ])
