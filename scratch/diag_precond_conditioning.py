"""
Diagnostic: does Jacobi (and Kronecker) reduce the condition number / CG-governing
spectrum of the PG feature-space operator A = I + Ds C_w Ds, and how does it trend with N?

For each N we fit EFGP briefly to get a REALISTIC converged delta (Omega) and the exact
spectral state the CG used, build A densely, and compute the spectrum of the
preconditioned operators M^{-1} A for M in {I (none), diag(A) (Jacobi), Kronecker}.
cond(M^{-1}A) is what governs CG iteration count.

Run: ~/myenv/bin/python scratch/diag_precond_conditioning.py
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from vanilla_gp_sampling import sample_gp_spectral_approx
from polyagamma_classification.pg_classifier import (
    PolyagammaGPNegativeBinomialRegressor,
    _build_weighted_toeplitz,
    _build_cg_preconditioner,
)

LS, VAR, R_TRUE = 0.20, 1.0, 2.0


def make_data(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 2, generator=g) * 2 - 1
    f = sample_gp_spectral_approx(x, length_scale=LS, variance=VAR, seed=seed + 1).reshape(-1)
    y = torch.distributions.NegativeBinomial(total_count=R_TRUE, logits=f).sample()
    return x, y


def fitted_state(n):
    """Fit briefly, return (spectral_state, delta) actually used by the solver."""
    x, y = make_data(n)
    reg = PolyagammaGPNegativeBinomialRegressor(
        total_count=1.0, learn_total_count=True, lengthscale_init=LS, variance_init=VAR,
        max_iter=20, e_step_iters=1, final_e_step_iters=2, n_e_probes=1, n_m_probes=1,
        cg_tol=1e-6, nufft_eps=1e-8, spectral_eps=1e-4, trunc_eps=1e-4,
        use_exact_weighted_toeplitz_operator=True, random_state=0, device="cpu", verbose=0,
    )
    reg.fit(x.numpy().astype(np.float64), y.numpy().astype(np.int64))
    return reg._spectral_state_, reg._variational_state_.delta.detach()


def dense_A(spectral, delta):
    ws = spectral.ws
    M = ws.numel()
    WT = _build_weighted_toeplitz(delta.to(ws.dtype), spectral)
    A = torch.zeros(M, M, dtype=ws.dtype)
    eye = torch.eye(M, dtype=ws.dtype)
    for j in range(M):
        u = eye[:, j]
        A[:, j] = u + ws * WT(ws * u)
    return 0.5 * (A + A.conj().T)


def dense_Minv(M_inv, M):
    P = torch.zeros(M, M, dtype=torch.complex128)
    eye = torch.eye(M, dtype=torch.complex128)
    for j in range(M):
        P[:, j] = M_inv(eye[:, j])
    return P


def cond_of(mat):
    ev = torch.linalg.eigvals(mat)
    mag = ev.abs()
    mag = mag[mag > 1e-14]
    return float(mag.max() / mag.min())


def main():
    print(f"{'N':>7s}  {'M':>6s}  {'sum(delta)':>10s}  {'cond(A)':>10s}  "
          f"{'cond(Jac^-1 A)':>14s}  {'cond(Kron^-1 A)':>15s}")
    for n in (2000, 8000, 30000):
        spectral, delta = fitted_state(n)
        A = dense_A(spectral, delta)
        M = A.shape[0]
        cA = cond_of(A)
        Mj = _build_cg_preconditioner(delta, spectral, "jacobi")
        Mk = _build_cg_preconditioner(delta, spectral, "kronecker")
        Pj = dense_Minv(Mj, M)
        Pk = dense_Minv(Mk, M)
        cJ = cond_of(Pj @ A)
        cK = cond_of(Pk @ A)
        print(f"{n:7d}  {M:6d}  {float(delta.sum()):10.1f}  {cA:10.2f}  "
              f"{cJ:14.2f}  {cK:15.2f}")


if __name__ == "__main__":
    main()
