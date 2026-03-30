#!/usr/bin/env python3
"""
Verify the specialized sigma_f^2 gradient formulas before wiring them into EFGPND.

This script checks, on a small real-data subset where dense linear algebra is
feasible, that the variance-gradient identities

    tr(K^{-1} dK/dsigma_f^2) = (n - sigma_n^2 tr(K^{-1})) / sigma_f^2

and

    alpha^T (dK/dsigma_f^2) alpha
      = (y^T alpha - sigma_n^2 alpha^T alpha) / sigma_f^2

match the exact dense GP quantities for the current approximate kernel.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch

torch.set_num_threads(1)

REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from efgpnd import NUFFT, _cmplx  # noqa: E402
from kernels.squared_exponential import SquaredExponential  # noqa: E402
from utils.kernels import get_xis  # noqa: E402


def load_usa_temp_subset(n: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.load(REPO_DIR / "data" / "usa_temp_data.pt")
    x = data["x"].to(dtype=torch.float64)[:n]
    y = data["y"].to(dtype=torch.float64)[:n]
    x = (x - x.min(dim=0).values) / (x.max(dim=0).values - x.min(dim=0).values)
    y = (y - y.mean()) / y.std()
    return x, y


def build_explicit_phi(
    x: torch.Tensor,
    *,
    lengthscale: float,
    variance: float,
    eps: float,
    nufft_eps: float,
) -> tuple[torch.Tensor, int]:
    dtype = x.dtype
    cdtype = _cmplx(dtype)
    d = x.shape[1]

    kernel = SquaredExponential(dimension=d)
    kernel.set_hyper("lengthscale", lengthscale)
    kernel.set_hyper("variance", variance)

    L = float((x.max(dim=0).values - x.min(dim=0).values).max().item())
    xis_1d, h, mtot = get_xis(kernel_obj=kernel, eps=eps, L=L, use_integral=True, l2scaled=False)
    xis = torch.stack(torch.meshgrid(*(xis_1d for _ in range(d)), indexing="ij"), dim=-1).view(-1, d)
    ws = torch.sqrt(kernel.spectral_density(xis).to(dtype=cdtype) * h**d)

    out_shape = (mtot,) * d
    nufft = NUFFT(x, torch.zeros(d, dtype=dtype), h, nufft_eps, cdtype=cdtype)

    M = ws.numel()
    phi = torch.empty((x.shape[0], M), dtype=cdtype)
    batch_size = 64
    for start in range(0, M, batch_size):
        stop = min(start + batch_size, M)
        rows = stop - start
        coeffs = torch.zeros((rows, M), dtype=cdtype)
        local = torch.arange(rows)
        coeffs[local, start + local] = ws[start:stop]
        phi[:, start:stop] = nufft.type2(coeffs, out_shape=out_shape).T

    return phi, mtot


def main() -> None:
    x, y = load_usa_temp_subset(128)
    ell = 0.14
    sigma_f2 = 1.9
    sigma_n2 = 0.07
    eps = 1e-4
    nufft_eps = 1e-6

    phi, mtot = build_explicit_phi(
        x,
        lengthscale=ell,
        variance=sigma_f2,
        eps=eps,
        nufft_eps=nufft_eps,
    )

    n = x.shape[0]
    eye_n = torch.eye(n, dtype=phi.dtype)
    k_signal = phi @ phi.conj().T
    k = k_signal + sigma_n2 * eye_n
    k_inv = torch.linalg.inv(k)
    alpha = k_inv @ y.to(dtype=phi.dtype)

    dk_dsigma_f2 = k_signal / sigma_f2
    trace_dense = torch.trace(k_inv @ dk_dsigma_f2).real
    quad_dense = torch.vdot(alpha, dk_dsigma_f2 @ alpha).real
    grad_dense = 0.5 * (trace_dense - quad_dense)

    trace_k_inv = torch.trace(k_inv).real
    trace_special = (n - sigma_n2 * trace_k_inv) / sigma_f2
    quad_special = (
        torch.vdot(y.to(dtype=phi.dtype), alpha).real
        - sigma_n2 * torch.vdot(alpha, alpha).real
    ) / sigma_f2
    grad_special = 0.5 * (trace_special - quad_special)

    print(f"n={n}, mtot={mtot}, M={phi.shape[1]}")
    print(f"dense trace term      = {trace_dense.item():.12e}")
    print(f"specialized trace     = {trace_special.item():.12e}")
    print(f"abs diff trace        = {(trace_dense - trace_special).abs().item():.12e}")
    print(f"dense quadratic term  = {quad_dense.item():.12e}")
    print(f"specialized quadratic = {quad_special.item():.12e}")
    print(f"abs diff quadratic    = {(quad_dense - quad_special).abs().item():.12e}")
    print(f"dense grad            = {grad_dense.item():.12e}")
    print(f"specialized grad      = {grad_special.item():.12e}")
    print(f"abs diff grad         = {(grad_dense - grad_special).abs().item():.12e}")

    atol = 5e-9
    if not torch.allclose(trace_dense, trace_special, atol=atol, rtol=1e-9):
        raise SystemExit("trace identity check failed")
    if not torch.allclose(quad_dense, quad_special, atol=atol, rtol=1e-9):
        raise SystemExit("quadratic identity check failed")
    if not torch.allclose(grad_dense, grad_special, atol=atol, rtol=1e-9):
        raise SystemExit("gradient identity check failed")

    print("all checks passed")


if __name__ == "__main__":
    main()
