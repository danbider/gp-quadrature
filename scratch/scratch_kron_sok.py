"""
Sum-of-Kronecker (SoK-R) preconditioner — Idea 2.

Approximate V ≈ Σ_{k=1}^R σ_k u_k[l1] conj(w_k[l2]) via SVD of V.
Build R separate plain-Kron operators:
    M_k = (D_1 T_k^(1) D_1) ⊗ (D_2 T_k^(2) D_2) + σ² I
where T_k^(1), T_k^(2) are the 1D Toeplitz matrices with symbols
σ_k u_k and conj(w_k) respectively (sign/scale absorbed into symbol_1).

Apply M_SoK^{-1} v = Σ_{k=1}^R α_k M_k^{-1} v with weights either:
  - uniform  α_k = 1/R
  - energy   α_k = σ_k / Σ_j σ_j (cap at R leading terms)
  - lambda_opt  pick α_k to minimize ‖A x - M_SoK^{-1} A x‖ on random probes
                (skipped for first pass; can add if the basic version helps)

Each M_k^{-1} apply is O(M m) on d=2, same as plain Kron. So total setup is
R * (eigh of two m×m) and per-apply is R * O(M m). R small (1–8).

Run: ~/myenv/bin/python -u scratch/scratch_kron_sok.py
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
CDT = torch.complex128

_ORIG = efgp_mod.create_kronecker_precond
_state = {"installed": False, "last_info": ""}


def _herm_sym_1d(x):
    return 0.5 * (x + torch.flip(x, dims=[-1]).conj())


def _factor_ws(ws, m, d, cdtype):
    ws_nd = ws.view(*(m,) * d).to(cdtype)
    ctr = m // 2
    idx_ctr = (ctr,) * d
    ws_ctr = ws_nd[idx_ctr]
    if ws_ctr.abs().item() == 0.0:
        ws_ctr = ws_ctr + torch.finfo(torch.float64).tiny
    norm = ws_ctr ** ((d - 1) / d)
    Ds = []
    for k in range(d):
        slc = [ctr] * d
        slc[k] = slice(None)
        Ds.append((ws_nd[tuple(slc)] / norm).to(cdtype))
    return Ds


def build_sok_precond(ws, v_kernel, sigmasq_scalar, d, mtot_1d,
                     *, R, weights="uniform",
                     device=None, cdtype=CDT, rdtype=torch.float64):
    """Sum-of-Kron preconditioner for d=2 only."""
    assert d == 2, "SoK prototype is d=2 only"
    m = int(mtot_1d)
    if device is None:
        device = ws.device
    sigsq = float(sigmasq_scalar.detach().real.item()
                  if torch.is_tensor(sigmasq_scalar)
                  else sigmasq_scalar)

    # 1) factor ws → D_1, D_2 (same for all SVD terms)
    Ds = _factor_ws(ws, m, d, cdtype)

    # 2) SVD of V (shape (L, L)), Hermitian symmetry on each factor
    V = v_kernel.to(cdtype)
    L = V.shape[0]
    ctr_v = (L - 1) // 2
    U, S, Vh = torch.linalg.svd(V, full_matrices=False)
    R_use = min(R, S.numel())

    # 3) build per-rank (diag_inv, V_1, V_2) triples
    a = torch.arange(m, device=device)
    triples = []
    sigs = []
    for k in range(R_use):
        u_k = _herm_sym_1d(U[:, k])
        w_k = _herm_sym_1d(Vh[k, :].conj())
        s_k = S[k].to(cdtype)

        # Symbols: absorb σ_k into axis-0 symbol; axis-1 uses conj(w_k).
        sym_1 = s_k * u_k  # (L,)
        sym_2 = w_k.conj()  # (L,)

        T_1 = sym_1[(a[:, None] - a[None, :]) + ctr_v]
        T_2 = sym_2[(a[:, None] - a[None, :]) + ctr_v]

        H_1 = Ds[0][:, None] * T_1 * Ds[0][None, :]
        H_2 = Ds[1][:, None] * T_2 * Ds[1][None, :]
        H_1 = 0.5 * (H_1 + H_1.conj().T)
        H_2 = 0.5 * (H_2 + H_2.conj().T)

        Lam_1, V_1 = torch.linalg.eigh(H_1)
        Lam_2, V_2 = torch.linalg.eigh(H_2)
        Lam_1 = Lam_1.to(rdtype).clamp_min(0.0)
        Lam_2 = Lam_2.to(rdtype).clamp_min(0.0)

        diag_inv = (1.0 / (Lam_1[:, None] * Lam_2[None, :] + sigsq)).to(cdtype)
        triples.append((diag_inv, V_1.to(cdtype), V_2.to(cdtype)))
        sigs.append(float(S[k].detach().real.item()))

    # 4) weights
    if weights == "uniform":
        alphas = [1.0 / R_use] * R_use
    elif weights == "energy":
        tot = sum(sigs) or 1.0
        alphas = [s / tot for s in sigs]
    elif weights == "sqrt_energy":
        tot = sum(s ** 0.5 for s in sigs) or 1.0
        alphas = [(s ** 0.5) / tot for s in sigs]
    else:
        raise ValueError(f"unknown weights: {weights}")

    _state["last_info"] = (
        f"[SoK R={R_use}, w={weights}] σ/σ_1: "
        f"{['{:.3g}'.format(s / sigs[0]) for s in sigs[:min(8, R_use)]]}  "
        f"α: {['{:.3g}'.format(a) for a in alphas[:min(8, R_use)]]}"
    )

    def M_inv(v):
        is_batch = v.ndim > 1
        if is_batch:
            B = v.shape[0]
            t = v.to(cdtype).reshape(B, m, m)
        else:
            t = v.to(cdtype).reshape(m, m)

        out = torch.zeros_like(t)
        for alpha, (diag_inv, V_1, V_2) in zip(alphas, triples):
            # Apply (V_1^H ⊗ V_2^H) t : for d=2, batched:
            #   w = V_1^H t V_2.conj()
            if is_batch:
                w = torch.einsum('ij,bjk,kl->bil',
                                 V_1.conj().T, t, V_2.conj())
                w = w * diag_inv[None, :, :]
                y = torch.einsum('ij,bjk,kl->bil', V_1, w, V_2.T)
            else:
                w = V_1.conj().T @ t @ V_2.conj()
                w = w * diag_inv
                y = V_1 @ w @ V_2.T
            out = out + alpha * y

        if is_batch:
            return out.reshape(B, m * m)
        return out.reshape(m * m)

    return M_inv


def install_sok(R, weights="uniform"):
    def patched(ws, v_kernel, sigmasq_scalar, d, mtot_1d,
                device=None, cdtype=CDT, rdtype=torch.float64):
        if d != 2:
            return _ORIG(ws, v_kernel, sigmasq_scalar, d, mtot_1d,
                         device=device, cdtype=cdtype, rdtype=rdtype)
        return build_sok_precond(ws, v_kernel, sigmasq_scalar, d, mtot_1d,
                                 R=R, weights=weights, device=device,
                                 cdtype=cdtype, rdtype=rdtype)
    efgp_mod.create_kronecker_precond = patched
    _state["installed"] = True


def uninstall():
    efgp_mod.create_kronecker_precond = _ORIG
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
        print(f"    plain Kron              : {dt:.2f}s  mean-cg={it_m}, trace-cg={it_t}  M={M}",
              flush=True)

        dt, it_t, it_m, _ = one_shot(x, y, "jacobi", ls=ls, var=var, sig2=sig2)
        print(f"    Jacobi                  : {dt:.2f}s  mean-cg={it_m}, trace-cg={it_t}",
              flush=True)

        for weights in ["uniform", "energy"]:
            for R in [1, 2, 4, 8]:
                install_sok(R=R, weights=weights)
                try:
                    dt, it_t, it_m, _ = one_shot(x, y, "kronecker",
                                                 ls=ls, var=var, sig2=sig2)
                    print(f"    SoK(R={R:<2d}, {weights:<8s}): "
                          f"{dt:.2f}s  mean-cg={it_m}, trace-cg={it_t}",
                          flush=True)
                    if _state["last_info"] and R == 1:
                        # Only print the long info row once per weighting block
                        print(f"      {_state['last_info']}", flush=True)
                except Exception as e:
                    print(f"    SoK(R={R}, {weights}): FAILED "
                          f"{type(e).__name__}: {e}", flush=True)
                finally:
                    uninstall()


if __name__ == "__main__":
    print("Sum-of-Kronecker preconditioner (Idea 2)\n", flush=True)

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
