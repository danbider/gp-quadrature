"""
Circulant preconditioner for EFGP (Idea 3).

System:  A = D T D + σ² I        (x-space CG system)
Change of var y = D x gives
         B = T + σ² D^{-2}        (y-space, still SPD)

For B we can approximate T by an m^d BCCB circulant C (diagonal under FFT),
and absorb the diagonal σ² D^{-2} by a scalar α (mean or median).
Preconditioner:   M_B = C + α I          applied via FFT
Equivalent P_A:  P_A = D M_B D = D (C+αI) D
Apply P_A^{-1}:  D^{-1} F^H (Λ_C + α)^{-1} F D^{-1}   — O(d M log m).

Strang's circulant: C's kernel = central m^d slab of v_kernel, rolled so
origin sits at index 0; eigenvalues = d-dim FFT of that. For v_kernel
Hermitian-symmetric (empirical CF), Λ_C is real; clamp at 0 for PSD.

Run: ~/myenv/bin/python -u scratch/scratch_circ_precond.py
"""
from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..',
                                                 'experiments', 'real', 'oisst')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..',
                                                 'experiments', 'real', 'era5')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..',
                                                 'experiments', 'real', 'prism')))

import torch
import numpy as np
import efgpnd as efgp_mod
from efgpnd import EFGPND
from kernels.squared_exponential import SquaredExponential
from load_oisst import load_oisst

torch.set_default_dtype(torch.float64)
DT = torch.float64
CDT = torch.complex128

_ORIG = efgp_mod.create_kronecker_precond
_state = {"alpha_mode": "mean", "last_info": ""}


def strang_circulant_eigs(v_kernel, m, d):
    """
    Λ_C for Strang's m^d BCCB circulant approximation of T.

    v_kernel shape: (2m-1,)*d, centered at ((m-1),)*d (for mtot odd).
    Strategy: central m^d slab → roll so origin is at (0,)*d → d-dim FFT.
    """
    assert m % 2 == 1, f"expected odd mtot_1d, got m={m}"
    ctr = m - 1
    half = (m - 1) // 2
    slicer = tuple(slice(ctr - half, ctr + half + 1) for _ in range(d))
    c_centered = v_kernel[slicer]  # shape (m,)*d, origin at (half,)*d
    c = torch.roll(c_centered, shifts=(-half,) * d, dims=tuple(range(d)))
    Lam_C = torch.fft.fftn(c, dim=tuple(range(d)))
    # Symmetric c → real Lam_C (up to roundoff)
    return Lam_C.real


def build_circ_precond(ws, v_kernel, sigmasq_scalar, d, mtot_1d,
                       *, alpha_mode="mean",
                       device=None, cdtype=CDT, rdtype=torch.float64):
    m = int(mtot_1d)
    if device is None:
        device = ws.device
    sigsq = float(sigmasq_scalar.detach().real.item()
                  if torch.is_tensor(sigmasq_scalar)
                  else sigmasq_scalar)

    ws_r = ws.real.to(rdtype).clamp_min(torch.finfo(rdtype).tiny)
    # σ² D^{-2} diagonal in x-space becomes σ² / ws² on the B-system.
    # For a constant-α approximation, use mean or median of that.
    sig_D2 = sigsq / (ws_r ** 2)
    if alpha_mode == "mean":
        alpha = float(sig_D2.mean().item())
    elif alpha_mode == "median":
        alpha = float(sig_D2.median().item())
    elif alpha_mode == "geomean":
        alpha = float(sig_D2.log().mean().exp().item())
    elif alpha_mode == "sigma2":
        alpha = float(sigsq)  # naive fallback
    else:
        raise ValueError(f"unknown alpha_mode={alpha_mode}")

    Lam_C = strang_circulant_eigs(v_kernel.to(cdtype), m, d).to(rdtype)
    # Clamp tiny/negative circulant eigenvalues (PSD safety).
    Lam_C = Lam_C.clamp_min(0.0)
    denom = (Lam_C + alpha).to(cdtype)  # (m,)*d

    ws_c = ws_r.to(cdtype)
    _state["last_info"] = (
        f"[circ/{alpha_mode}] α={alpha:.3g}  σ²/ws² range=[{sig_D2.min().item():.3g}, "
        f"{sig_D2.max().item():.3g}]  Λ_C range=[{Lam_C.min().item():.3g}, "
        f"{Lam_C.max().item():.3g}]"
    )

    dims_fft = tuple(range(-d, 0))
    target_shape = (m,) * d

    def M_inv(v):
        is_batch = v.ndim > 1
        if is_batch:
            B = v.shape[0]
            w = v / ws_c
            w_nd = w.reshape(B, *target_shape)
        else:
            w = v / ws_c
            w_nd = w.reshape(*target_shape)
        W = torch.fft.fftn(w_nd, dim=dims_fft)
        W = W / denom
        w_nd = torch.fft.ifftn(W, dim=dims_fft)
        if is_batch:
            w = w_nd.reshape(B, -1)
        else:
            w = w_nd.reshape(-1)
        return w / ws_c

    return M_inv


def install_circ(alpha_mode="mean"):
    def patched(ws, v_kernel, sigmasq_scalar, d, mtot_1d,
                device=None, cdtype=CDT, rdtype=torch.float64):
        return build_circ_precond(ws, v_kernel, sigmasq_scalar, d, mtot_1d,
                                  alpha_mode=alpha_mode, device=device,
                                  cdtype=cdtype, rdtype=rdtype)
    efgp_mod.create_kronecker_precond = patched
    _state["alpha_mode"] = alpha_mode


def uninstall():
    efgp_mod.create_kronecker_precond = _ORIG
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


def load_prism_xy(n_sub=1_000_000):
    from load_prism import load_prism_dataset
    prism_dir = "/Users/colecitrenbaum/Documents/GPs/prism_tmean_us_30s_202602"
    x_np, y_np = load_prism_dataset(prism_dir, n_sub=n_sub, seed=0)
    x = normalize(torch.from_numpy(x_np.astype(np.float64)))
    y = standardize(torch.from_numpy(y_np.astype(np.float64)))
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
        print(f"    plain Kron        : {dt:.2f}s  mean-cg={it_m}, trace-cg={it_t}  M={M}",
              flush=True)

        dt, it_t, it_m, _ = one_shot(x, y, "jacobi", ls=ls, var=var, sig2=sig2)
        print(f"    Jacobi            : {dt:.2f}s  mean-cg={it_m}, trace-cg={it_t}",
              flush=True)

        for mode in ["mean", "median", "geomean"]:
            install_circ(alpha_mode=mode)
            try:
                dt, it_t, it_m, _ = one_shot(x, y, "kronecker",
                                             ls=ls, var=var, sig2=sig2)
                print(f"    Circ α={mode:<7s}: {dt:.2f}s  mean-cg={it_m}, "
                      f"trace-cg={it_t}", flush=True)
                if _state["last_info"]:
                    print(f"      {_state['last_info']}", flush=True)
            except Exception as e:
                print(f"    Circ α={mode}: FAILED {type(e).__name__}: {e}",
                      flush=True)
            finally:
                uninstall()


if __name__ == "__main__":
    print("Circulant preconditioner (Idea 3)\n", flush=True)

    t0 = time.perf_counter()
    xo, yo = load_oisst_full()
    print(f"Loaded OISST in {time.perf_counter()-t0:.1f}s: "
          f"N={xo.shape[0]:,}, d={xo.shape[1]}", flush=True)
    run_dataset("OISST anom (full)", xo, yo, [
        (0.02, 1.0, 1e-2),
        (0.01, 1.0, 1e-2),
    ])
    del xo, yo

    t0 = time.perf_counter()
    xe, ye = load_era5_xy()
    print(f"\nLoaded ERA5 in {time.perf_counter()-t0:.1f}s: "
          f"N={xe.shape[0]:,}, d={xe.shape[1]}", flush=True)
    run_dataset("ERA5 t2m (full)", xe, ye, [
        (0.02, 1.0, 1e-2),
    ])
    del xe, ye

    t0 = time.perf_counter()
    xp, yp = load_prism_xy(n_sub=1_000_000)
    print(f"\nLoaded PRISM in {time.perf_counter()-t0:.1f}s: "
          f"N={xp.shape[0]:,}, d={xp.shape[1]}", flush=True)
    run_dataset("PRISM tmean (sub 1M)", xp, yp, [
        (0.02, 1.0, 1e-2),
    ])
