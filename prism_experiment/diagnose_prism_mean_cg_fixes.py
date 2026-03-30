#!/usr/bin/env python3
"""
Quick PRISM diagnostic for mean-solve CG fixes in the sad regime.

This script compares four configurations for the shared mean solve

    A_mean beta = D F^* y

on the full standardized PRISM tmean dataset:

1. zero init, no preconditioner
2. zero init, Jacobi preconditioner
3. warm start from previous hyperparameter state, no preconditioner
4. warm start + Jacobi preconditioner

The warm-start test uses the real iter30 -> iter40 transition from the notebook
log because both states have the same feature grid size (`mtot = 25`, `M = 625`)
so the previous `beta` lives in the same feature space.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch

torch.set_num_threads(1)

THIS_DIR = Path(__file__).resolve().parent
REPO_DIR = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from diagnose_efgpnd_learning_curve import RecordingConjugateGradients  # noqa: E402
from efgpnd import (  # noqa: E402
    NUFFT,
    ToeplitzND,
    _cmplx,
    compute_convolution_vector_vectorized_dD,
    create_A_mean,
)
from kernels.squared_exponential import SquaredExponential  # noqa: E402
from load_prism import load_prism_dataset_torch  # noqa: E402
from utils.kernels import get_xis  # noqa: E402


@dataclass
class MeanSolveState:
    name: str
    lengthscale: float
    variance: float
    sigmasq: float
    mtot: int
    M: int
    rhs: torch.Tensor
    A_apply: object
    beta: torch.Tensor
    diag_scale: float


def load_standardized_prism() -> tuple[torch.Tensor, torch.Tensor]:
    x, y = load_prism_dataset_torch("prism_tmean_us_30s_2020_avg_30y")
    x = x.to(dtype=torch.float64)
    y = y.to(dtype=torch.float64)
    x = (x - x.min(dim=0).values) / (x.max(dim=0).values - x.min(dim=0).values)
    y = (y - y.mean()) / y.std()
    return x, y


def build_state(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    name: str,
    lengthscale: float,
    variance: float,
    sigmasq: float,
    eps: float,
    nufft_eps: float,
    cg_tol: float,
) -> MeanSolveState:
    kernel = SquaredExponential(dimension=2)
    kernel.set_hyper("lengthscale", lengthscale)
    kernel.set_hyper("variance", variance)

    dtype = x.dtype
    cdtype = _cmplx(dtype)
    d = x.shape[1]
    L = float((x.max(dim=0).values - x.min(dim=0).values).max().item())

    xis_1d, h, mtot = get_xis(kernel, eps=eps, L=L, use_integral=True, l2scaled=False)
    xis_1d = xis_1d.to(dtype=dtype)
    xis = torch.stack(torch.meshgrid(*(xis_1d for _ in range(d)), indexing="ij"), dim=-1).reshape(-1, d)
    ws = torch.sqrt(kernel.spectral_density(xis).to(dtype=cdtype) * h**d)

    OUT = (mtot,) * d
    nufft = NUFFT(x, torch.zeros(d, dtype=dtype), h, nufft_eps, cdtype=cdtype)
    fadj = lambda v: nufft.type1(v, out_shape=OUT).reshape(-1)

    m_conv = (mtot - 1) // 2
    v_kernel = compute_convolution_vector_vectorized_dD(m_conv, x, h).to(dtype=cdtype)
    toeplitz = ToeplitzND(v_kernel, force_pow2=True)
    A_apply = create_A_mean(ws, toeplitz, torch.tensor(sigmasq, dtype=dtype), cdtype)
    rhs = ws * fadj(y).reshape(-1)

    cg = RecordingConjugateGradients(
        A_apply,
        rhs,
        torch.zeros_like(rhs),
        tol=cg_tol,
        early_stopping=True,
    )
    beta, _ = cg.solve()

    center = tuple(((torch.tensor(v_kernel.shape) - 1) // 2).tolist())
    diag_scale = float(v_kernel[center].real.item())

    return MeanSolveState(
        name=name,
        lengthscale=lengthscale,
        variance=variance,
        sigmasq=sigmasq,
        mtot=mtot,
        M=ws.numel(),
        rhs=rhs,
        A_apply=A_apply,
        beta=beta,
        diag_scale=diag_scale,
    )


def run_suite(
    state: MeanSolveState,
    *,
    warm_start: Optional[torch.Tensor],
    jacobi_diag: torch.Tensor,
    cg_tol: float,
) -> list[Dict[str, float]]:
    cases = [
        ("baseline", None, None),
        ("jacobi", None, lambda v: v / jacobi_diag),
        ("warm_start", warm_start, None),
        ("warm_start+jacobi", warm_start, lambda v: v / jacobi_diag),
    ]

    rows = []
    for label, x0, m_inv in cases:
        solver = RecordingConjugateGradients(
            state.A_apply,
            state.rhs,
            torch.zeros_like(state.rhs) if x0 is None else x0.clone(),
            tol=cg_tol,
            early_stopping=True,
            M_inv_apply=m_inv,
        )
        _, stats = solver.solve()
        rows.append({
            "label": label,
            "iters": int(stats.iters_completed),
            "rel_res": float(stats.rel_res_max),
            "cap_hit": int(stats.iters_completed == 2 * state.M),
        })
    return rows


def main() -> None:
    eps = 1e-4
    nufft_eps = 1e-5
    cg_tol = 1e-5

    print("Loading and standardizing full PRISM tmean dataset...")
    x, y = load_standardized_prism()
    print(f"N={x.shape[0]}")

    prev_state = build_state(
        x, y,
        name="iter30",
        lengthscale=0.09451,
        variance=2.427,
        sigmasq=0.05509,
        eps=eps,
        nufft_eps=nufft_eps,
        cg_tol=cg_tol,
    )
    target_state = build_state(
        x, y,
        name="iter40",
        lengthscale=0.09256,
        variance=3.878,
        sigmasq=0.05202,
        eps=eps,
        nufft_eps=nufft_eps,
        cg_tol=cg_tol,
    )

    if prev_state.mtot != target_state.mtot:
        raise RuntimeError(
            f"Warm-start test expects same feature grid, got mtot {prev_state.mtot} vs {target_state.mtot}"
        )

    kernel = SquaredExponential(dimension=2)
    kernel.set_hyper("lengthscale", target_state.lengthscale)
    kernel.set_hyper("variance", target_state.variance)
    L = float((x.max(dim=0).values - x.min(dim=0).values).max().item())
    xis_1d, h, mtot = get_xis(kernel, eps=eps, L=L, use_integral=True, l2scaled=False)
    xis_1d = xis_1d.to(dtype=x.dtype)
    xis = torch.stack(torch.meshgrid(*(xis_1d for _ in range(x.shape[1])), indexing="ij"), dim=-1).reshape(-1, x.shape[1])
    ws = torch.sqrt(kernel.spectral_density(xis).to(dtype=_cmplx(x.dtype)) * h**x.shape[1])
    jacobi_diag = target_state.diag_scale * ws.abs().pow(2).real + target_state.sigmasq

    rows = run_suite(
        target_state,
        warm_start=prev_state.beta,
        jacobi_diag=jacobi_diag,
        cg_tol=cg_tol,
    )

    print(
        f"\nTarget state: {target_state.name}  "
        f"ell={target_state.lengthscale}  sigma_f^2={target_state.variance}  "
        f"sigma_n^2={target_state.sigmasq}  mtot={target_state.mtot}  M={target_state.M}"
    )
    print(
        f"Warm start source: {prev_state.name}  "
        f"ell={prev_state.lengthscale}  sigma_f^2={prev_state.variance}  sigma_n^2={prev_state.sigmasq}"
    )
    print("\nCase                 iters    rel_res      cap_hit")
    for row in rows:
        print(f"{row['label']:<20} {row['iters']:>6}   {row['rel_res']:.3e}   {bool(row['cap_hit'])}")


if __name__ == "__main__":
    main()
