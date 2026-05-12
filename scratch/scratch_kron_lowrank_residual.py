"""
Prototype: Kron + low-rank residual preconditioner.

Idea: A = M_kron + E, where E captures how the true joint structure deviates
from the product-measure assumption baked into Kron (e.g. OISST land mask).
Extract top-r Hermitian eigenpairs of E via randomized symmetric eigendecomp,
then solve via Woodbury.

Validates on OISST (full masked) where plain Kron takes ~100 iters vs Jacobi's
~50.  Target: with r ≈ 100 Ritz vecs, Kron+LR should drop to <20 iters.

Run: ~/myenv/bin/python -u scratch/scratch_kron_lowrank_residual.py
"""
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

torch.set_default_dtype(torch.float64)
DT = torch.float64

_ORIG_CREATE_KRON = efgp_mod.create_kronecker_precond


# ---------- M_forward: duplicate of Kron setup but applies M, not M^{-1} ----

def build_M_apply_and_inv(ws, v_kernel, sigsq, d, m, device, cdtype, rdtype):
    """Return (M_apply, M_inv_apply) both using the same eigendecomp.
    Mirrors create_kronecker_precond's math but exposes the forward operator."""
    ws_nd = ws.view(*(m,) * d).to(cdtype)
    ctr_ws = m // 2
    ws_ctr = ws_nd[(ctr_ws,) * d]
    if ws_ctr.abs().item() == 0.0:
        ws_ctr = ws_ctr + torch.finfo(rdtype).tiny
    norm_ws = ws_ctr ** ((d - 1) / d)
    Ds = []
    for k in range(d):
        slc = [ctr_ws] * d; slc[k] = slice(None)
        Ds.append((ws_nd[tuple(slc)] / norm_ws).to(cdtype))

    L = v_kernel.shape[0]
    ctr_v = (L - 1) // 2
    v_ctr_val = v_kernel[(ctr_v,) * d].to(cdtype)
    if v_ctr_val.abs().item() == 0.0:
        v_ctr_val = v_ctr_val + torch.finfo(rdtype).tiny
    v_norm = v_ctr_val ** ((d - 1) / d)

    Vs, Lams = [], []
    for k in range(d):
        slc = [ctr_v] * d; slc[k] = slice(None)
        v_1d = (v_kernel[tuple(slc)].to(cdtype)) / v_norm
        a = torch.arange(m, device=device)
        T_k = v_1d[(a[:, None] - a[None, :]) + ctr_v]
        H_k = Ds[k][:, None] * T_k * Ds[k][None, :]
        H_k = 0.5 * (H_k + H_k.conj().T)
        Lam_k, V_k = torch.linalg.eigh(H_k)
        Lams.append(Lam_k.to(rdtype).clamp_min(0.0))
        Vs.append(V_k.to(cdtype))

    lam_prod = torch.ones(*(m,) * d, device=device, dtype=rdtype)
    for k in range(d):
        shape = [1] * d; shape[k] = m
        lam_prod = lam_prod * Lams[k].view(shape)
    diag_fwd = (lam_prod + sigsq).to(cdtype)
    diag_inv = (1.0 / diag_fwd).to(cdtype)

    def _apply_kron(t, mats, hermitian):
        nd = t.ndim
        for k in range(d):
            ax = nd - d + k
            t = torch.movedim(t, ax, -1)
            if hermitian:
                t = t @ mats[k].conj()
            else:
                t = t @ mats[k].transpose(-1, -2)
            t = torch.movedim(t, -1, ax)
        return t

    def _apply_with_diag(v, diag):
        is_batch = v.ndim > 1
        if is_batch:
            B = v.shape[0]
            t = v.to(cdtype).reshape(B, *(m,) * d)
        else:
            t = v.to(cdtype).reshape(*(m,) * d)
        t = _apply_kron(t, Vs, hermitian=True)
        t = t * diag
        t = _apply_kron(t, Vs, hermitian=False)
        if is_batch:
            return t.reshape(B, -1)
        return t.reshape(-1)

    def M_apply(v):
        return _apply_with_diag(v, diag_fwd)

    def M_inv(v):
        return _apply_with_diag(v, diag_inv)

    return M_apply, M_inv


# ---------- Randomized symmetric eigendecomp of E = A - M --------------------

@torch.no_grad()
def randomized_eigh_residual(A_apply, M_apply, n: int, r: int, p: int = 10,
                              *, device, cdtype, rdtype, seed: int = 0):
    """Return (U, Lam) with U (n, r) Hermitian-orthonormal cols and Lam (r,)
    real, capturing top-|λ| Hermitian eigenpairs of E = A - M.

    Uses symmetric randomized SVD: Y = E Ω, Q = qr(Y), T = Q^H E Q, eigh(T).
    """
    g = torch.Generator(device=device).manual_seed(seed)
    # Complex Gaussian test matrix
    Om_r = torch.randn(n, r + p, generator=g, device=device, dtype=rdtype)
    Om_i = torch.randn(n, r + p, generator=g, device=device, dtype=rdtype)
    Om = (Om_r + 1j * Om_i).to(cdtype)

    # E Ω = A Ω - M Ω  (batched applies: shape (B, n) convention)
    Y_A = A_apply(Om.T).T
    Y_M = M_apply(Om.T).T
    Y = Y_A - Y_M

    Q, _ = torch.linalg.qr(Y)  # (n, r+p)

    # T = Q^H E Q = Q^H A Q - Q^H M Q
    AQ = A_apply(Q.T).T
    MQ = M_apply(Q.T).T
    Tmat = Q.conj().T @ (AQ - MQ)
    Tmat = 0.5 * (Tmat + Tmat.conj().T)

    Lam_all, V_small = torch.linalg.eigh(Tmat)   # ascending real eigenvalues
    # Only keep POSITIVE eigenvalues — guarantees M + UΛU^H stays SPD.
    # Negative eigenvalues of E would indicate M over-estimates A in that
    # direction; adding negative mass to M could make it indefinite and break
    # CG.  Positive-truncation is a safe one-sided correction.
    pos_mask = Lam_all > 0
    Lam_pos = Lam_all[pos_mask]
    V_pos = V_small[:, pos_mask]
    # Pick top-r of the positives by magnitude
    r_eff = min(r, Lam_pos.numel())
    idx = torch.argsort(Lam_pos, descending=True)[:r_eff]
    Lam = Lam_pos[idx]
    V_small_r = V_pos[:, idx]
    U = Q @ V_small_r
    return U, Lam


# ---------- Monkey-patched preconditioner ------------------------------------

_patch_state = {}


def make_lowrank_wrapper(r: int, p: int = 10, verbose: bool = True):
    def patched(ws, v_kernel, sigmasq_scalar, *args, **kw):
        d = kw.get("d", args[0] if len(args) >= 1 else None)
        mtot_1d = kw.get("mtot_1d", args[1] if len(args) >= 2 else None)
        device = kw.get("device", ws.device)
        cdtype = kw.get("cdtype", torch.complex128)
        rdtype = kw.get("rdtype", torch.float64)
        m = int(mtot_1d)
        M_total = m ** d

        if torch.is_tensor(sigmasq_scalar):
            sigsq = float(sigmasq_scalar.detach().real.item())
        else:
            sigsq = float(sigmasq_scalar)

        t0 = time.perf_counter()
        M_apply, M_inv = build_M_apply_and_inv(ws, v_kernel, sigsq, d, m,
                                                device, cdtype, rdtype)

        # A_apply = D T D + σ² I, where D = diag(ws), T = d-dim Toeplitz
        toep = ToeplitzND(v_kernel, force_pow2=False)

        def A_apply(v):
            # v can be (M,) or (B, M)
            is_batch = v.ndim > 1
            if is_batch:
                Tv = toep(ws * v)
                return ws * Tv + sigsq * v
            else:
                return ws * toep(ws * v) + sigsq * v

        t_setup = time.perf_counter() - t0
        t0 = time.perf_counter()

        U, Lam = randomized_eigh_residual(
            A_apply, M_apply, n=M_total, r=r, p=p,
            device=device, cdtype=cdtype, rdtype=rdtype,
        )
        t_rsvd = time.perf_counter() - t0

        t0 = time.perf_counter()
        # Woodbury precompute: (M + UΛU^H)^{-1}
        # = M^{-1} − M^{-1}U (Λ^{-1} + U^H M^{-1} U)^{-1} U^H M^{-1}
        MinvU = M_inv(U.T).T  # (n, r) — batch along last axis
        S = torch.diag(1.0 / Lam.to(cdtype)) + U.conj().T @ MinvU
        S = 0.5 * (S + S.conj().T)
        S_LU = torch.linalg.lu_factor(S)
        t_wood = time.perf_counter() - t0

        if verbose:
            _patch_state['last_log'] = (
                f"[Kron+LR] setup(M)={t_setup*1000:.1f}ms  "
                f"rSVD(r={r},p={p})={t_rsvd*1000:.1f}ms  "
                f"Woodbury={t_wood*1000:.1f}ms  "
                f"|Λ|∈[{Lam.abs().min().item():.2e}, {Lam.abs().max().item():.2e}]"
            )

        def M_inv_aug(v):
            is_batch = v.ndim > 1
            y = M_inv(v)  # (B, M) or (M,)
            if is_batch:
                z = y @ U.conj()  # (B, r)  — note: y @ U.conj() = (U^H y^T).T^H? careful
                # We need z_i = (U^H y_i) per row. Row i of y is y_i^T (as row vec).
                # (U^H y_i) = (y_i^H U)^H. Doing y @ U.conj() yields Σ_j y[b,j] * U[j,k].conj()
                # = (U^H y_i)[k]. Correct.
                w = torch.linalg.lu_solve(*S_LU, z.T).T  # (B, r)
                y_corr = w @ MinvU.T  # (B, M)
                return y - y_corr
            else:
                z = U.conj().T @ y  # (r,)
                w = torch.linalg.lu_solve(*S_LU, z.unsqueeze(-1)).squeeze(-1)
                y_corr = MinvU @ w  # (M,)
                return y - y_corr

        return M_inv_aug

    return patched


def install_lowrank(r: int, p: int = 10):
    efgp_mod.create_kronecker_precond = make_lowrank_wrapper(r=r, p=p)


def uninstall():
    efgp_mod.create_kronecker_precond = _ORIG_CREATE_KRON


# ---------- Benchmark harness ------------------------------------------------

def normalize(x):
    mn = x.min(dim=0).values; mx = x.max(dim=0).values
    return (x - mn) / (mx - mn)


def load_full():
    x, y = load_oisst(variable="anom")
    x = normalize(torch.from_numpy(x.astype(np.float64)))
    y = torch.from_numpy(y.astype(np.float64))
    y = (y - y.mean()) / y.std()
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
    t_list, trace_iters = [], []
    for _ in range(K):
        t0 = time.perf_counter()
        model.compute_gradients(trace_samples=J, cg_tol=cg_tol,
                                noise_floor=noise_floor)
        t_list.append(time.perf_counter() - t0)
        trace_iters.append(model.last_gradient_stats.get('trace_cg_iters'))
    return dict(times=t_list, iters=trace_iters,
                M=model.last_gradient_stats.get('feature_count'))


def mean(xs):
    xs = [v for v in xs if v is not None]
    return sum(xs) / len(xs) if xs else float('nan')


if __name__ == "__main__":
    print("OISST full: Kron vs Jacobi vs Kron+LR(r)\n", flush=True)
    x, y = load_full()
    print(f"Loaded OISST: N={x.shape[0]:,}, d={x.shape[1]}\n", flush=True)

    HYPER_CASES = [
        (0.02, 1.0, 1e-2),
    ]
    R_LIST = [50, 100, 200]

    for ls, var, sig2 in HYPER_CASES:
        print(f"=== ℓ={ls}, σ_f²={var}, σ²={sig2} ===", flush=True)
        # Baseline: plain Kron
        uninstall()
        r = time_grad(x, y, "kronecker", ls=ls, var=var, sig2=sig2)
        print(f"  plain Kron        : {sum(r['times'])/len(r['times']):.2f}s, "
              f"cg={mean(r['iters']):.0f}, M={r['M']}", flush=True)
        # Baseline: Jacobi
        r = time_grad(x, y, "jacobi", ls=ls, var=var, sig2=sig2)
        print(f"  Jacobi            : {sum(r['times'])/len(r['times']):.2f}s, "
              f"cg={mean(r['iters']):.0f}", flush=True)
        # Nystrom (existing precond in efgpnd)
        for rr in R_LIST:
            try:
                kernel = SquaredExponential(dimension=x.shape[1],
                                            init_lengthscale=ls,
                                            init_variance=var)
                model = EFGPND(x, y, kernel=kernel, sigmasq=sig2, eps=1e-3,
                               estimate_params=False,
                               opts={"mean_cg_preconditioner_type": "nystrom",
                                     "nystrom_rank": rr,
                                     "nystrom_oversample": 10,
                                     "max_cg_iterations": 3000})
                # warmup
                model.compute_gradients(trace_samples=1, cg_tol=1e-4,
                                        noise_floor=1e-5)
                ts = []
                its = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    model.compute_gradients(trace_samples=1, cg_tol=1e-4,
                                            noise_floor=1e-5)
                    ts.append(time.perf_counter() - t0)
                    its.append(model.last_gradient_stats.get('trace_cg_iters'))
                print(f"  Nystrom(r={rr:<4d})  : {sum(ts)/len(ts):.2f}s, "
                      f"cg={mean(its):.0f}", flush=True)
            except Exception as e:
                print(f"  Nystrom(r={rr}): FAILED: {type(e).__name__}: {e}",
                      flush=True)
        # Kron + low-rank residual (positive-truncated)
        for rr in R_LIST:
            install_lowrank(r=rr, p=10)
            try:
                r = time_grad(x, y, "kronecker", ls=ls, var=var, sig2=sig2)
                log = _patch_state.get('last_log', '')
                print(f"  Kron+LR(r={rr:<4d}) : {sum(r['times'])/len(r['times']):.2f}s, "
                      f"cg={mean(r['iters']):.0f}", flush=True)
                if log:
                    print(f"      {log}", flush=True)
            finally:
                uninstall()
        print(flush=True)
