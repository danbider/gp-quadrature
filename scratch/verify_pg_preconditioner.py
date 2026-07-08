"""
Correctness check for the PG feature-space preconditioners under HETEROSCEDASTIC noise.

The PG feature-space operator (used by BOTH the mean solve `_make_feature_space_solver`
and the variance solve `_make_sigma_apply`) is

    A = I + Ds C_w Ds,   Ds = sqrt(spectral density),   C_w = F^H diag(delta) F

where the per-point PG weights delta = Omega are HETEROSCEDASTIC. Note the prior enters as
the "+I" term with coefficient 1 (NOT a divided-out scalar sigma^2 as in efgpnd's A_mean =
D C D + sigma^2 I). efgpnd's preconditioner makers build (weighted-Gram) + sigmasq*I, so we
pass sigmasq_scalar=1 and fold ALL heteroscedasticity into the weighted convolution vector
C_w. This script verifies that mapping by building A densely (small grid) and checking:

  1. Jacobi M^{-1} == exact 1/diag(A)  (diag = 1 + |ws|^2 * sum_i delta_i).
  2. Kronecker M^{-1} A ~ I  (preconditioned operator near-identity), and that it
     drastically reduces the condition number vs unpreconditioned A.

Run: ~/myenv/bin/python scratch/verify_pg_preconditioner.py
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

from kernels import SquaredExponential
from polyagamma_classification.pg_classifier import (
    _build_spectral_state,
    _build_weighted_toeplitz,
    _build_cg_preconditioner,
)


def build_dense_A(spectral, delta):
    """Dense A = I + Ds C_w Ds via the SAME operator the CGs use (A_feat)."""
    ws = spectral.ws
    M = ws.numel()
    WT = _build_weighted_toeplitz(delta.to(ws.dtype), spectral)

    def A_apply(u):
        t = ws * u
        return u + ws * WT(t)

    A = torch.zeros(M, M, dtype=ws.dtype)
    eye = torch.eye(M, dtype=ws.dtype)
    for j in range(M):
        A[:, j] = A_apply(eye[:, j])
    return A


def apply_M_inv_dense(M_inv, M):
    """Materialize the preconditioner as a dense matrix by applying to basis vectors."""
    P = torch.zeros(M, M, dtype=torch.complex128)
    eye = torch.eye(M, dtype=torch.complex128)
    for j in range(M):
        P[:, j] = M_inv(eye[:, j])
    return P


def main():
    torch.manual_seed(0)
    # Small problem so the M x M dense operator is cheap (M = mtot^d).
    n, d = 600, 2
    x = torch.rand(n, d) * 2.0 - 1.0
    kernel = SquaredExponential(dimension=d, init_lengthscale=0.3, init_variance=1.0)
    spectral = _build_spectral_state(
        x, kernel, spectral_eps=1e-3, trunc_eps=1e-3, nufft_eps=1e-10,
        rdtype=torch.float64, cdtype=torch.complex128, device=torch.device("cpu"),
    )
    M = spectral.ws.numel()

    # HETEROSCEDASTIC PG weights: strongly varying per-point delta (not a scalar!).
    delta = 0.02 + 2.0 * torch.rand(n)
    print(f"M (features) = {M},  N = {n},  d = {d}")
    print(f"delta (Omega) heteroscedastic: min={delta.min():.3f} max={delta.max():.3f} "
          f"mean={delta.mean():.3f}  sum={delta.sum():.3f}")

    A = build_dense_A(spectral, delta)
    Aherm_err = float((A - A.conj().T).abs().max())
    condA = float(torch.linalg.cond(A).real)
    print(f"A Hermitian error = {Aherm_err:.2e}   cond(A) = {condA:.3e}")

    # --- 1. Jacobi: exact reciprocal diagonal ---
    Mj = _build_cg_preconditioner(delta, spectral, "jacobi")
    jac_diag = 1.0 / Mj(torch.ones(M, dtype=torch.complex128))   # recover diag_elements
    A_diag = torch.diagonal(A)
    diag_err = float((jac_diag - A_diag).abs().max())
    # Analytic expectation: diag(A) = 1 + |ws|^2 * sum(delta)
    analytic = 1.0 + spectral.ws.abs().pow(2) * float(delta.sum())
    analytic_err = float((jac_diag.real - analytic).abs().max())
    print("\n[Jacobi]")
    print(f"  max|1/M_inv - diag(A)|            = {diag_err:.2e}  (want ~0)")
    print(f"  max|jac_diag - (1+|ws|^2*sum d)|  = {analytic_err:.2e}  (want ~0)")

    # --- 2. Kronecker: preconditioned operator near identity ---
    Mk = _build_cg_preconditioner(delta, spectral, "kronecker")
    P = apply_M_inv_dense(Mk, M)
    PA = P @ A
    I = torch.eye(M, dtype=torch.complex128)
    offdiag = float((PA - I).abs().max())
    condPA = float(torch.linalg.cond(PA).real)
    print("\n[Kronecker]")
    print(f"  cond(M^-1 A) = {condPA:.3e}   (vs cond(A) = {condA:.3e})")
    print(f"  max|M^-1 A - I| = {offdiag:.3e}")

    jac_ok = diag_err < 1e-8 and analytic_err < 1e-8
    kron_ok = condPA < condA           # must improve conditioning
    print("\n=== VERDICT ===")
    print(f"  Jacobi exact-diagonal under heteroscedastic Omega: {'OK' if jac_ok else 'FAIL'}")
    print(f"  Kronecker reduces condition number:                {'OK' if kron_ok else 'FAIL'}")
    return 0 if (jac_ok and kron_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
