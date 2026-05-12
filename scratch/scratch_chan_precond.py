"""
Benchmark Chan-circulant-augmented preconditioning for EFGP's mean system.

System:  A = D T D + sigma^2 I  (1D), with T symmetric Toeplitz (FFT-applicable),
D = diag(ws), ws = sqrt(khat(xi) * h).

Three preconditioners compared:
  (0) none
  (1) jacobi:   M_inv = 1 / (T_00 * D^2 + sigma^2)   (= current efgpnd default)
  (2) jacobi+chan: symmetric Jacobi + Chan circulant correction on T.
      Build Chan's optimal circulant C ~ T, symmetrize via J = T_00 D^2 + sigma^2,
      then M_inv r = J^{-1/2} (I + D_tilde C_off D_tilde)^{-1} J^{-1/2} r,
      where D_tilde = D / sqrt(J), C_off = C - T_00 I.
      Inner solve uses fixed-iteration CG via FFT circulant applies.

Regimes:
  easy:  moderate lengthscale + noise -> small mtot, T well-conditioned on active block.
  hard:  small lengthscale + small noise -> larger mtot, ill-conditioned active block.

Reports per config: outer matvec count (= #NUFFT-free Toeplitz FFTs), inner FFT
matvec count, wallclock to reach cg_tol, and final residual.

Run:
  ~/myenv/bin/python scratch/scratch_chan_precond.py
"""

from __future__ import annotations
import sys, os, time, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch

from kernels.squared_exponential import SquaredExponential
from utils.kernels import get_xis
from efgpnd import (
    ToeplitzND,
    compute_convolution_vector_vectorized_dD,
)

torch.set_default_dtype(torch.float64)
CDTYPE = torch.complex128
RDTYPE = torch.float64


# ---------------------------------------------------------------------------
# Counting wrapper
# ---------------------------------------------------------------------------
class Counter:
    def __init__(self, fn, label=""):
        self.fn = fn
        self.count = 0
        self.label = label

    def __call__(self, v):
        self.count += 1
        return self.fn(v)

    def reset(self):
        self.count = 0


# ---------------------------------------------------------------------------
# Problem setup
# ---------------------------------------------------------------------------
def build_problem(N, lengthscale, sigmasq, variance=1.0, eps=1e-6, seed=0,
                  device='cpu', clustered=False):
    torch.manual_seed(seed)
    if clustered:
        # Two tight gaussian clusters at +/-0.3; highly non-uniform density
        n1 = N // 2
        n2 = N - n1
        x1 = 0.02 * torch.randn(n1, 1, dtype=RDTYPE) - 0.3
        x2 = 0.02 * torch.randn(n2, 1, dtype=RDTYPE) + 0.3
        x = torch.cat([x1, x2], dim=0).to(device)
        x = x.clamp(-0.499, 0.499)
    else:
        x = (torch.rand(N, 1, device=device, dtype=RDTYPE) - 0.5)
    kernel = SquaredExponential(
        dimension=1,
        init_lengthscale=lengthscale,
        init_variance=variance,
    )
    L = 1.0  # domain radius
    xis_1d, h, mtot = get_xis(kernel, eps=eps, L=L, use_integral=True,
                              l2scaled=False, dtype=RDTYPE)
    xis = xis_1d.reshape(-1, 1).to(device)
    h_t = h if torch.is_tensor(h) else torch.tensor(h, dtype=RDTYPE)
    ws = torch.sqrt(kernel.spectral_density(xis).to(CDTYPE) * (h_t ** 1))  # (mtot,)
    ws = ws.reshape(-1).to(CDTYPE)
    # Real, nonneg => cast to real for diagonal math
    D = ws.real.to(RDTYPE)

    m_conv = (mtot - 1) // 2
    v_kernel = compute_convolution_vector_vectorized_dD(m_conv, x, float(h)).to(CDTYPE)
    toeplitz = ToeplitzND(v_kernel, force_pow2=True)

    # Toeplitz first column (length mtot)
    # v_kernel has length 4*m_conv + 1; center index = 2*m_conv; T_{k} = v_kernel[2*m_conv + k]
    t_col = v_kernel[2 * m_conv: 2 * m_conv + mtot].clone()  # (mtot,) complex
    T_00 = t_col[0].real.item()

    return {
        "N": N, "mtot": mtot, "h": float(h), "m_conv": m_conv,
        "ws": ws, "D": D, "toeplitz": toeplitz, "t_col": t_col, "T_00": T_00,
        "sigmasq": sigmasq, "lengthscale": lengthscale, "x": x,
    }


# ---------------------------------------------------------------------------
# Chan's optimal circulant approximation
# ---------------------------------------------------------------------------
def chan_circulant_eigs(t_col: torch.Tensor) -> torch.Tensor:
    """
    Chan's optimal circulant approximation of a HERMITIAN Toeplitz T with
    first column t_col (length n, complex).  For Hermitian T, t_{-k} = conj(t_k),
    so the Chan formula c_j = ((n-j)*t_j + j*t_{j-n})/n becomes
        c_j = ((n-j)*t_j + j*conj(t_{n-j})) / n,   j = 0..n-1.
    Returns eigenvalues lambda = FFT(c) (real for Hermitian T).
    """
    n = t_col.shape[0]
    c = torch.empty_like(t_col)
    c[0] = t_col[0]
    j = torch.arange(1, n, device=t_col.device)
    c[1:] = ((n - j).to(t_col.dtype) * t_col[1:]
             + j.to(t_col.dtype) * t_col[n - j].conj()) / n
    lam = torch.fft.fft(c)
    return lam


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------
def make_A_apply(D: torch.Tensor, toeplitz: ToeplitzND, sigmasq: float):
    """A v = D T D v + sigma^2 v.   D real, T via FFT (returns complex)."""
    Dc = D.to(CDTYPE)

    def apply(v):
        v = v.to(CDTYPE)
        return Dc * toeplitz(Dc * v) + sigmasq * v

    return apply


def make_jacobi_Minv(D, T_00, sigmasq):
    diag = T_00 * D.pow(2) + sigmasq
    def Minv(r):
        return r / diag.to(r.dtype)
    return Minv


def _build_circulant_dense(lam_C: torch.Tensor) -> torch.Tensor:
    """Dense n x n circulant with eigenvalues lam_C (columns = shifts of first col c)."""
    n = lam_C.shape[0]
    c = torch.fft.ifft(lam_C)                 # first column
    # Build circulant: col i = roll(c, i)
    idx = (torch.arange(n).unsqueeze(1) - torch.arange(n).unsqueeze(0)) % n  # (n,n), i-j mod n
    return c[idx]


def make_jacobi_chan_Minv_direct(D, lam_C, T_00, sigmasq, counter=None):
    """
    Symmetric Jacobi + Chan circulant correction, inner solved DIRECTLY via
    Cholesky / LU (concept test; O(n^3) factor, O(n^2) apply).
    Inner system:   (I + D_tilde * C_off * D_tilde) w = s,
    C_off = Chan(T) - T_00 I.  Note C_off is indefinite in general, so we LU.
    """
    J = T_00 * D.pow(2) + sigmasq
    Jinv_half = 1.0 / torch.sqrt(J)
    D_tilde = D * Jinv_half
    n = D.shape[0]

    # Build M_inner = I + diag(D_tilde) * C_off * diag(D_tilde) densely.
    C_dense = _build_circulant_dense(lam_C.to(CDTYPE))
    C_off = C_dense - T_00 * torch.eye(n, dtype=CDTYPE)
    Dt = D_tilde.to(CDTYPE)
    M_inner = torch.eye(n, dtype=CDTYPE) + (Dt.unsqueeze(1) * C_off) * Dt.unsqueeze(0)

    # Diagnostic: spectrum of M_inner
    eigs = torch.linalg.eigvalsh(0.5 * (M_inner + M_inner.conj().T)).real
    spd = (eigs.min().item() > 0)

    # LU factor (handles indefinite); if SPD could use cholesky for speed.
    lu, piv = torch.linalg.lu_factor(M_inner)

    Jinv_half_c = Jinv_half.to(CDTYPE)

    def Minv(r):
        if counter is not None:
            counter.count += 1
        s = Jinv_half_c * r.to(CDTYPE)
        w = torch.linalg.lu_solve(lu, piv, s.unsqueeze(-1)).squeeze(-1)
        return Jinv_half_c * w

    return Minv, dict(inner_spd=spd, inner_eig_min=eigs.min().item(),
                      inner_eig_max=eigs.max().item())


def make_chan_only_Minv(lam_C, sigmasq):
    """
    Pure-circulant preconditioner: M = C + sigmasq*I (ignores D).
    M^{-1} via FFT: O(n log n).
    """
    denom = (lam_C + sigmasq).to(CDTYPE)

    def Minv(r):
        r_c = r.to(CDTYPE)
        return torch.fft.ifft(torch.fft.fft(r_c) / denom)

    return Minv


# ---------------------------------------------------------------------------
# Minimal PCG (so we can count matvecs cleanly).
# ---------------------------------------------------------------------------
def pcg(A_apply, b, M_inv=None, tol=1e-6, max_iter=2000):
    x = torch.zeros_like(b)
    r = b.clone()
    z = M_inv(r) if M_inv is not None else r.clone()
    p = z.clone()
    rz = torch.vdot(r, z).real
    b_norm = torch.linalg.norm(b).real.clamp_min(1e-30)
    history = []
    for it in range(max_iter):
        Ap = A_apply(p)
        pAp = torch.vdot(p, Ap).real + 1e-30
        alpha = rz / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rel = torch.linalg.norm(r).real / b_norm
        history.append(rel.item())
        if rel < tol:
            return x, it + 1, history
        z = M_inv(r) if M_inv is not None else r
        rz_new = torch.vdot(r, z).real
        beta = rz_new / (rz + 1e-30)
        p = z + beta * p
        rz = rz_new
    return x, max_iter, history


# ---------------------------------------------------------------------------
# Benchmark one configuration
# ---------------------------------------------------------------------------
def run_one(label, A_apply_counter, b, M_inv, tol, max_iter):
    # Reset outer counter
    A_apply_counter.reset()
    t0 = time.perf_counter()
    x, iters, hist = pcg(A_apply_counter, b, M_inv=M_inv, tol=tol, max_iter=max_iter)
    dt = time.perf_counter() - t0
    # Final true residual
    A_apply_counter.fn  # noqa
    # Compute residual without counting it
    r_final = b - A_apply_counter.fn(x)
    rel = (torch.linalg.norm(r_final) / torch.linalg.norm(b)).real.item()
    return dict(label=label, iters=iters, outer_matvecs=A_apply_counter.count,
                time_s=dt, rel_res=rel)


def benchmark(regime_name, N, lengthscale, sigmasq, tol=1e-6, max_iter=2000,
              clustered=False):
    print(f"\n{'='*70}")
    print(f"Regime: {regime_name}")
    print(f"  N={N}, lengthscale={lengthscale}, sigmasq={sigmasq}, clustered={clustered}")
    prob = build_problem(N=N, lengthscale=lengthscale, sigmasq=sigmasq,
                         clustered=clustered)
    print(f"  mtot={prob['mtot']}, h={prob['h']:.5g}, T_00={prob['T_00']:.5g}")

    D = prob["D"]
    ws = prob["ws"]
    toeplitz = prob["toeplitz"]
    sigmasq_f = prob["sigmasq"]

    # Effective dynamic range of D
    D_min = D[D > 0].min().item() if (D > 0).any() else 0.0
    D_max = D.max().item()
    print(f"  D range: [{D_min:.3e}, {D_max:.3e}]  ratio={D_max / max(D_min,1e-300):.3e}")

    # RHS: random-ish to exercise all modes
    torch.manual_seed(42)
    b = (torch.randn(prob["mtot"], dtype=RDTYPE)
         + 1j * torch.randn(prob["mtot"], dtype=RDTYPE)).to(CDTYPE)
    b = ws * b  # put RHS in the "weighted" shape that arises in practice

    # Operators
    A_raw = make_A_apply(D, toeplitz, sigmasq_f)
    A_cnt = Counter(A_raw, "outer")

    lam_C = chan_circulant_eigs(prob["t_col"])

    results = []
    results.append(run_one("none", A_cnt, b, M_inv=None, tol=tol, max_iter=max_iter))
    results.append(run_one("jacobi", A_cnt, b,
                           M_inv=make_jacobi_Minv(D, prob["T_00"], sigmasq_f),
                           tol=tol, max_iter=max_iter))
    results.append(run_one("chan-only (FFT)", A_cnt, b,
                           M_inv=make_chan_only_Minv(lam_C, sigmasq_f),
                           tol=tol, max_iter=max_iter))

    # ---- SANITY: form A and M densely, compare spectra, verify apply. ----
    if prob["mtot"] <= 200:
        n = prob["mtot"]
        Dc = D.to(CDTYPE)
        # A dense
        C_dense = _build_circulant_dense(lam_C.to(CDTYPE))
        # Get T dense from toeplitz by applying to identity columns
        I_mat = torch.eye(n, dtype=CDTYPE)
        T_dense = torch.stack([toeplitz(I_mat[:, i]) for i in range(n)], dim=1)
        A_dense = (Dc.unsqueeze(1) * T_dense) * Dc.unsqueeze(0) + sigmasq_f * torch.eye(n, dtype=CDTYPE)
        M_dense = (Dc.unsqueeze(1) * C_dense) * Dc.unsqueeze(0) + sigmasq_f * torch.eye(n, dtype=CDTYPE)
        ck_herm_A = (A_dense - A_dense.conj().T).abs().max().item()
        ck_herm_M = (M_dense - M_dense.conj().T).abs().max().item()
        A_s = 0.5 * (A_dense + A_dense.conj().T)
        M_s = 0.5 * (M_dense + M_dense.conj().T)
        evA = torch.linalg.eigvalsh(A_s).real
        evM = torch.linalg.eigvalsh(M_s).real
        TmC = (T_dense - C_dense).abs().max().item()
        relTC = TmC / T_dense.abs().max().item()
        print(f"  A herm_err={ck_herm_A:.2e} eig=[{evA.min():.3e},{evA.max():.3e}]  "
              f"M herm_err={ck_herm_M:.2e} eig=[{evM.min():.3e},{evM.max():.3e}]")
        print(f"  ||T - C||_max / ||T||_max = {relTC:.3e}")
        # Condition number of M^{-1} A
        Minv_A = torch.linalg.solve(M_s, A_s)
        evs = torch.linalg.eigvals(Minv_A).real
        print(f"  spec(M^-1 A) in [{evs.min().item():.3e}, {evs.max().item():.3e}] "
              f"kappa={(evs.max()/evs.min()).item():.3e}")

    # Jacobi + Chan (direct dense inner solve)
    Minv_jc, diag = make_jacobi_chan_Minv_direct(D, lam_C, prob["T_00"], sigmasq_f)
    print(f"  inner (I + D~ C_off D~): spd={diag['inner_spd']}  "
          f"eig range=[{diag['inner_eig_min']:.3e}, {diag['inner_eig_max']:.3e}]")
    res = run_one("jacobi+chan(direct)", A_cnt, b, M_inv=Minv_jc,
                  tol=tol, max_iter=max_iter)
    results.append(res)

    # Print
    print(f"\n  {'label':<22s} {'iters':>6s} {'outerMV':>8s} {'time(s)':>9s} {'rel_res':>10s}")
    for r in results:
        print(f"  {r['label']:<22s} {r['iters']:>6d} {r['outer_matvecs']:>8d} "
              f"{r['time_s']:>9.3f} {r['rel_res']:>10.2e}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Uniform data regimes
    benchmark("EASY (uniform, ell=0.1, s2=1e-2)",
              N=5_000, lengthscale=0.1, sigmasq=1e-2, tol=1e-6, max_iter=2000)
    benchmark("MEDIUM (uniform, ell=0.03, s2=1e-4)",
              N=20_000, lengthscale=0.03, sigmasq=1e-4, tol=1e-6, max_iter=4000)
    benchmark("HARD (uniform, ell=0.01, s2=1e-6)",
              N=50_000, lengthscale=0.01, sigmasq=1e-6, tol=1e-6, max_iter=8000)

    # Very hard: push mtot larger
    benchmark("VERY-HARD (uniform, ell=0.005, s2=1e-8)",
              N=100_000, lengthscale=0.005, sigmasq=1e-8, tol=1e-6, max_iter=8000)

    # Clustered data (non-uniform density): where user expects Chan to shine
    benchmark("CLUSTERED-EASY (ell=0.05, s2=1e-4)",
              N=10_000, lengthscale=0.05, sigmasq=1e-4, tol=1e-6, max_iter=4000,
              clustered=True)
    benchmark("CLUSTERED-HARD (ell=0.02, s2=1e-6)",
              N=50_000, lengthscale=0.02, sigmasq=1e-6, tol=1e-6, max_iter=8000,
              clustered=True)
