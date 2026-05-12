"""
Randomized Nystrom preconditioner for the EFGP mean system.

System:  A = D T D + sigma^2 I.  Nystrom builds U, Lambda such that
    A_hat = U Lambda U^T  (rank-k PSD, SPD top-k eigenspace of A)
Preconditioner  P = U Lambda U^T + mu (I - U U^T),  with mu = sigma^2
(lower bound on A's spectrum).  Apply:
    P^{-1} r = U Lambda^{-1} (U^T r) + mu^{-1} (r - U U^T r)

Setup: k+p matvecs of A (FFT-backed Toeplitz applies); O(M k^2) for QR/SVD.
Apply: O(M k) per PCG iteration.

Reference: Frangella, Tropp, Udell, "Randomized Nystrom Preconditioning",
SIMAX 2023.  Algorithm 3 (stabilized variant).

Run:  ~/myenv/bin/python scratch/scratch_nystrom_vs_jacobi.py
"""
from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

from kernels.squared_exponential import SquaredExponential
from utils.kernels import get_xis
from efgpnd import ToeplitzND, compute_convolution_vector_vectorized_dD

torch.set_default_dtype(torch.float64)
CDTYPE = torch.complex128
RDTYPE = torch.float64


# ---------------------------------------------------------------------------
# Problem builders (mirror scratch_jacobi_vs_none.py)
# ---------------------------------------------------------------------------
def sample_x(kind: str, N: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    if kind == "uniform":
        x = torch.rand(N, generator=g, dtype=RDTYPE) - 0.5
    elif kind == "gauss-0.05":
        x = 0.05 * torch.randn(N, generator=g, dtype=RDTYPE)
        x = x.clamp(-0.499, 0.499)
    elif kind == "two-cluster":
        n1 = N // 2
        n2 = N - n1
        x1 = 0.02 * torch.randn(n1, generator=g, dtype=RDTYPE) - 0.3
        x2 = 0.02 * torch.randn(n2, generator=g, dtype=RDTYPE) + 0.3
        x = torch.cat([x1, x2]).clamp(-0.499, 0.499)
    elif kind == "one-sided":
        x = 0.5 * torch.rand(N, generator=g, dtype=RDTYPE) - 0.5
    else:
        raise ValueError(kind)
    return x.reshape(-1, 1)


def build(N, x_kind, ell, sigmasq, eps=1e-6):
    x = sample_x(x_kind, N)
    kernel = SquaredExponential(dimension=1, init_lengthscale=ell, init_variance=1.0)
    xis_1d, h, mtot = get_xis(kernel, eps=eps, L=1.0, use_integral=True,
                              l2scaled=False, dtype=RDTYPE)
    xis = xis_1d.reshape(-1, 1)
    h_t = h if torch.is_tensor(h) else torch.tensor(h, dtype=RDTYPE)
    ws = torch.sqrt(kernel.spectral_density(xis).to(CDTYPE) * h_t).reshape(-1).to(CDTYPE)
    D = ws.real.to(RDTYPE)
    m_conv = (mtot - 1) // 2
    v_kernel = compute_convolution_vector_vectorized_dD(m_conv, x, float(h)).to(CDTYPE)
    toep = ToeplitzND(v_kernel, force_pow2=True)
    t_col = v_kernel[2 * m_conv: 2 * m_conv + mtot].clone()
    T_00 = t_col[0].real.item()
    return dict(D=D, toep=toep, sigmasq=sigmasq, T_00=T_00, mtot=mtot, t_col=t_col)


# ---------------------------------------------------------------------------
# Nystrom preconditioner (Frangella-Tropp-Udell, stabilized)
# ---------------------------------------------------------------------------
def build_nystrom(A_apply, M: int, rank: int, oversample: int = 10, seed: int = 0):
    """
    Returns (U_hat, Lam_hat, n_matvecs).  A_hat = U_hat diag(Lam_hat) U_hat^T.
    Uses the Cholesky-stabilized FTU algorithm.
    """
    r = rank + oversample
    g = torch.Generator().manual_seed(seed)
    # Complex Gaussian: both real & imag parts
    Omega = (torch.randn(M, r, generator=g, dtype=RDTYPE)
             + 1j * torch.randn(M, r, generator=g, dtype=RDTYPE)).to(CDTYPE) / (2 ** 0.5)
    # Orthonormalize (stability; also not strictly required)
    Omega, _ = torch.linalg.qr(Omega, mode="reduced")

    # Apply A to each column.
    Y = torch.empty(M, r, dtype=CDTYPE)
    for j in range(r):
        Y[:, j] = A_apply(Omega[:, j])
    n_matvecs = r

    # Shift for stability
    nu = (torch.finfo(RDTYPE).eps ** 0.5) * torch.linalg.norm(Y).real
    Y_nu = Y + nu * Omega
    # Gram matrix: should be Hermitian PD after shift
    C = Omega.conj().T @ Y_nu
    C = 0.5 * (C + C.conj().T)  # symmetrize numerically
    # Cholesky
    L = torch.linalg.cholesky(C)
    B = torch.linalg.solve_triangular(L, Y_nu.conj().T, upper=False).conj().T  # B = Y_nu L^{-H}
    # SVD
    U, S, _ = torch.linalg.svd(B, full_matrices=False)
    Lam = (S ** 2 - nu).clamp_min(0.0)  # undo the shift
    # Truncate to top-rank
    U_hat = U[:, :rank]
    Lam_hat = Lam[:rank].to(RDTYPE)
    return U_hat, Lam_hat, n_matvecs


def make_nystrom_Minv(U_hat, Lam_hat, mu):
    """
    P^{-1} r = U (Lam + mu I)^{-1} ... wait actually for A_hat ~ U Lam U^T,
    and P = U Lam U^T + mu (I - U U^T) (so spectrum: Lam on span U, mu on complement),
    P^{-1} r = U diag(1/Lam) U^T r + (1/mu) (r - U U^T r).
    We use mu = sigmasq (lower bound on spectrum of A).
    """
    Lam_inv = (1.0 / Lam_hat).to(CDTYPE)
    mu_inv = 1.0 / mu

    def Minv(r):
        r_c = r.to(CDTYPE)
        Uh_r = U_hat.conj().T @ r_c                # (k,)
        on_subspace = U_hat @ (Lam_inv * Uh_r)     # (M,)
        off_subspace = mu_inv * (r_c - U_hat @ Uh_r)
        return on_subspace + off_subspace

    return Minv


# ---------------------------------------------------------------------------
# PCG
# ---------------------------------------------------------------------------
def pcg(A_apply, b, M_inv=None, tol=1e-6, max_iter=5000):
    x = torch.zeros_like(b)
    r = b.clone()
    z = M_inv(r) if M_inv is not None else r.clone()
    p = z.clone()
    rz = torch.vdot(r, z).real
    b_norm = torch.linalg.norm(b).real.clamp_min(1e-30)
    for it in range(max_iter):
        Ap = A_apply(p)
        pAp = torch.vdot(p, Ap).real + 1e-30
        alpha = rz / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rel = torch.linalg.norm(r).real / b_norm
        if rel < tol:
            return x, it + 1, rel.item()
        z = M_inv(r) if M_inv is not None else r
        rz_new = torch.vdot(r, z).real
        beta = rz_new / (rz + 1e-30)
        p = z + beta * p
        rz = rz_new
    return x, max_iter, rel.item()


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def run_one(prob, label, M_inv, tol=1e-6, max_iter=5000):
    D, toep, sigmasq = prob["D"], prob["toep"], prob["sigmasq"]
    Dc = D.to(CDTYPE)
    A_apply = lambda v: Dc * toep(Dc * v) + sigmasq * v

    torch.manual_seed(0)
    b = (torch.randn(prob["mtot"], dtype=RDTYPE)
         + 1j * torch.randn(prob["mtot"], dtype=RDTYPE)).to(CDTYPE)
    b = prob["D"].to(CDTYPE) * b

    # warmup
    pcg(A_apply, b, M_inv=M_inv, tol=tol, max_iter=3)
    t0 = time.perf_counter()
    _, iters, rel = pcg(A_apply, b, M_inv=M_inv, tol=tol, max_iter=max_iter)
    dt = time.perf_counter() - t0
    return dict(label=label, iters=iters, time=dt, rel=rel)


def benchmark(regime_label, N, x_kind, ell, sigmasq, tol=1e-6, max_iter=4000,
              nystrom_ranks=(10, 30, 60)):
    prob = build(N, x_kind, ell, sigmasq)
    M = prob["mtot"]
    print(f"\n{regime_label}: {x_kind}, N={N}, ell={ell}, s2={sigmasq}, mtot={M}")

    D = prob["D"]
    Dc = D.to(CDTYPE)
    toep = prob["toep"]
    A_apply = lambda v: Dc * toep(Dc * v) + sigmasq * v

    results = []
    results.append(run_one(prob, "none", M_inv=None, tol=tol, max_iter=max_iter))

    diag = prob["T_00"] * D.pow(2) + sigmasq
    diag_c = diag.to(CDTYPE)
    Minv_jac = lambda r: r / diag_c
    results.append(run_one(prob, "jacobi", M_inv=Minv_jac, tol=tol, max_iter=max_iter))

    for k in nystrom_ranks:
        if k + 10 >= M:
            continue
        t0 = time.perf_counter()
        U_hat, Lam_hat, n_mv = build_nystrom(A_apply, M, rank=k, oversample=10, seed=0)
        t_setup = time.perf_counter() - t0
        Minv_nys = make_nystrom_Minv(U_hat, Lam_hat, mu=sigmasq)
        res = run_one(prob, f"nystrom(k={k})", M_inv=Minv_nys, tol=tol, max_iter=max_iter)
        res["setup_s"] = t_setup
        res["setup_mv"] = n_mv
        res["lam_top"] = Lam_hat[0].item()
        res["lam_bot"] = Lam_hat[-1].item()
        results.append(res)

    print(f"  {'label':<18s} {'iters':>6s} {'setup_mv':>9s} {'setup_s':>9s} "
          f"{'solve_s':>9s} {'total_s':>9s} {'rel':>10s}")
    for r in results:
        setup_mv = r.get("setup_mv", 0)
        setup_s = r.get("setup_s", 0.0)
        total_s = setup_s + r["time"]
        print(f"  {r['label']:<18s} {r['iters']:>6d} {setup_mv:>9d} "
              f"{setup_s:>9.3f} {r['time']:>9.3f} {total_s:>9.3f} {r['rel']:>10.2e}")
    return results


if __name__ == "__main__":
    # Regimes where Jacobi LOST big in the prior sweep
    benchmark("A", N=20_000, x_kind="two-cluster", ell=0.03, sigmasq=1e-4)
    benchmark("B", N=20_000, x_kind="two-cluster", ell=0.01, sigmasq=1e-6)
    benchmark("C", N=20_000, x_kind="gauss-0.05", ell=0.03, sigmasq=1e-4)
    benchmark("D", N=20_000, x_kind="gauss-0.05", ell=0.01, sigmasq=1e-6)
    benchmark("E", N=100_000, x_kind="one-sided", ell=0.01, sigmasq=1e-6)

    # Regimes where Jacobi WON big (make sure Nystrom doesn't regress)
    benchmark("F", N=20_000, x_kind="uniform", ell=0.03, sigmasq=1e-4)
    benchmark("G", N=100_000, x_kind="uniform", ell=0.01, sigmasq=1e-6)
