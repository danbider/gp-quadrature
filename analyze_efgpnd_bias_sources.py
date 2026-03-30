#!/usr/bin/env python3
"""
Bias decomposition for EFGPND gradients.

This script keeps `efgpnd.py` unchanged. It studies three sources of error:

1. Spectral aliasing error from coarse xi spacing.
2. Spectral truncation error from finite xi cutoff.
3. CG solve bias from inexact linear solves in the existing gradient path.

Trace noise is held fixed or reduced; it is not the focus here.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from diagnose_efgpnd_learning_curve import (  # noqa: E402
    instrumented_gradient_step,
    make_dataset,
)
from kernels.squared_exponential import SquaredExponential  # noqa: E402
from utils.kernels import get_xis  # noqa: E402


@dataclass
class GridSpec:
    label: str
    h: float
    hm: int

    @property
    def mtot(self) -> int:
        return 2 * self.hm + 1

    @property
    def xi_max(self) -> float:
        return self.hm * self.h


def make_kernel(lengthscale: float, variance: float) -> SquaredExponential:
    kernel = SquaredExponential(dimension=2)
    kernel.set_hyper("lengthscale", lengthscale)
    kernel.set_hyper("variance", variance)
    return kernel


def exact_se_gradient(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    lengthscale: float,
    variance: float,
    sigmasq: float,
) -> torch.Tensor:
    dist = torch.cdist(x, x)
    K = variance * torch.exp(-0.5 * (dist / lengthscale) ** 2)
    Kn = K + sigmasq * torch.eye(x.shape[0], dtype=x.dtype, device=x.device)
    alpha = torch.linalg.solve(Kn, y)
    KinvdK = torch.linalg.solve(Kn, torch.eye(x.shape[0], dtype=x.dtype, device=x.device))

    dK_dl = K * (dist**2) / (lengthscale**3)
    dK_dv = K / variance
    eye = torch.eye(x.shape[0], dtype=x.dtype, device=x.device)

    grads = []
    for dK in (dK_dl, dK_dv, eye):
        trace_term = torch.trace(KinvdK @ dK)
        quad_term = torch.dot(alpha, dK @ alpha)
        grads.append(0.5 * (trace_term - quad_term))
    return torch.stack(grads)


def xi_grid_1d(h: float, hm: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.arange(-hm, hm + 1, dtype=dtype) * h


def spectral_approx_gradient_dense(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: SquaredExponential,
    sigmasq: float,
    *,
    h: float,
    hm: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    dtype = x.dtype
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    d = x.shape[1]
    xis_1d = xi_grid_1d(h, hm, dtype).to(x.device)
    grids = torch.meshgrid(*(xis_1d for _ in range(d)), indexing="ij")
    xis = torch.stack(grids, dim=-1).reshape(-1, d)
    weights = kernel.spectral_density(xis).to(dtype=dtype) * (h**d)
    dweights = kernel.spectral_grad(xis).to(dtype=dtype) * (h**d)

    phases = 2 * math.pi * (x @ xis.T)
    Phi = torch.exp(1j * phases.to(dtype=dtype)).to(dtype=cdtype)
    W = weights.to(dtype=cdtype)

    K_tilde = (Phi * W.unsqueeze(0)) @ Phi.conj().T
    K_tilde = K_tilde.real
    Kn = K_tilde + sigmasq * torch.eye(x.shape[0], dtype=dtype, device=x.device)
    alpha = torch.linalg.solve(Kn, y)
    KinvdK = torch.linalg.solve(Kn, torch.eye(x.shape[0], dtype=dtype, device=x.device))

    grad_mats = []
    for i in range(2):
        Wi = dweights[:, i].to(dtype=cdtype)
        dK = ((Phi * Wi.unsqueeze(0)) @ Phi.conj().T).real
        grad_mats.append(dK)
    grad_mats.append(torch.eye(x.shape[0], dtype=dtype, device=x.device))

    grads = []
    for dK in grad_mats:
        trace_term = torch.trace(KinvdK @ dK)
        quad_term = torch.dot(alpha, dK @ alpha)
        grads.append(0.5 * (trace_term - quad_term))

    metrics = {
        "mtot": float(2 * hm + 1),
        "M": float((2 * hm + 1) ** d),
        "xi_max": float(hm * h),
        "h": float(h),
    }
    return torch.stack(grads), metrics


def relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    num = torch.linalg.norm(a - b).item()
    den = torch.linalg.norm(b).item()
    return num / den if den > 0 else num


def abs_error_by_param(a: torch.Tensor, b: torch.Tensor) -> List[float]:
    return [float(v) for v in torch.abs(a - b).tolist()]


def build_grid_family(std_h: float, std_hm: int, refine_h: int, refine_cutoff: int) -> List[GridSpec]:
    xi_max_std = std_hm * std_h
    ref_h = std_h / refine_h
    ref_xi_max = xi_max_std * refine_cutoff
    ref_hm = math.ceil(ref_xi_max / ref_h)

    trunc_only_hm = math.ceil(xi_max_std / ref_h)
    alias_only_hm = math.ceil(ref_xi_max / std_h)

    return [
        GridSpec("standard", std_h, std_hm),
        GridSpec("alias_only", std_h, alias_only_hm),
        GridSpec("trunc_only", ref_h, trunc_only_hm),
        GridSpec("reference", ref_h, ref_hm),
    ]


def bias_rows_for_state(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    label: str,
    lengthscale: float,
    variance: float,
    sigmasq: float,
    eps: float,
    refine_h: int,
    refine_cutoff: int,
) -> List[Dict[str, float]]:
    kernel = make_kernel(lengthscale, variance)
    xis_1d, std_h, mtot = get_xis(kernel, eps=eps, L=float((x.max(0).values - x.min(0).values).max().item()), use_integral=True)
    std_hm = (mtot - 1) // 2
    grids = build_grid_family(std_h, std_hm, refine_h, refine_cutoff)

    exact_grad = exact_se_gradient(x, y, lengthscale=lengthscale, variance=variance, sigmasq=sigmasq)

    dense_results = {}
    for grid in grids:
        grad, metrics = spectral_approx_gradient_dense(x, y, kernel, sigmasq, h=grid.h, hm=grid.hm)
        dense_results[grid.label] = (grad, metrics)

    ref_grad = dense_results["reference"][0]
    rows = []
    for grid in grids:
        grad, metrics = dense_results[grid.label]
        row = {
            "state": label,
            "kind": grid.label,
            "lengthscale": lengthscale,
            "variance": variance,
            "sigmasq": sigmasq,
            "eps": eps,
            "h": metrics["h"],
            "hm": float(grid.hm),
            "mtot": metrics["mtot"],
            "M": metrics["M"],
            "xi_max": metrics["xi_max"],
            "rel_err_vs_exact": relative_error(grad, exact_grad),
            "rel_err_vs_reference": relative_error(grad, ref_grad),
            "abs_err_l_vs_exact": abs_error_by_param(grad, exact_grad)[0],
            "abs_err_v_vs_exact": abs_error_by_param(grad, exact_grad)[1],
            "abs_err_n_vs_exact": abs_error_by_param(grad, exact_grad)[2],
            "abs_err_l_vs_reference": abs_error_by_param(grad, ref_grad)[0],
            "abs_err_v_vs_reference": abs_error_by_param(grad, ref_grad)[1],
            "abs_err_n_vs_reference": abs_error_by_param(grad, ref_grad)[2],
        }
        rows.append(row)
    return rows


def cg_rows_for_state(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    label: str,
    lengthscale: float,
    variance: float,
    sigmasq: float,
    eps: float,
    nufft_eps: float,
    trace_samples: int,
    seed: int,
    cg_tols: Iterable[float],
) -> List[Dict[str, float]]:
    x0 = x.min(dim=0).values
    x1 = x.max(dim=0).values
    rows = []

    for cg_tol in cg_tols:
        kernel = make_kernel(lengthscale, variance)
        grad, metrics = instrumented_gradient_step(
            x=x,
            y=y,
            sigmasq=torch.tensor(sigmasq, dtype=x.dtype, device=x.device),
            kernel=kernel,
            eps=eps,
            trace_samples=trace_samples,
            x0=x0,
            x1=x1,
            nufft_eps=nufft_eps,
            cg_tol=cg_tol,
            seed=seed,
        )
        rows.append(
            {
                "state": label,
                "cg_tol": float(cg_tol),
                "trace_samples": float(trace_samples),
                "lengthscale": lengthscale,
                "variance": variance,
                "sigmasq": sigmasq,
                "grad_l": float(grad[0].item()),
                "grad_v": float(grad[1].item()),
                "grad_n": float(grad[2].item()),
                "mean_cg_iters": metrics["mean_cg_iters"],
                "trace_cg_iters_median": metrics["trace_cg_iters_median"],
                "trace_cg_iters_max": metrics["trace_cg_iters_max"],
                "stage_total_sec": metrics["stage_total_sec"],
                "stage_trace_cg_sec": metrics["stage_trace_cg_sec"],
                "M": metrics["M"],
            }
        )

    ref = rows[-1]
    for row in rows:
        g = torch.tensor([row["grad_l"], row["grad_v"], row["grad_n"]], dtype=x.dtype)
        gref = torch.tensor([ref["grad_l"], ref["grad_v"], ref["grad_n"]], dtype=x.dtype)
        row["rel_err_vs_tight_cg"] = relative_error(g, gref)
        row["abs_err_l_vs_tight_cg"], row["abs_err_v_vs_tight_cg"], row["abs_err_n_vs_tight_cg"] = abs_error_by_param(g, gref)
    return rows


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_bias_summary(rows: List[Dict[str, float]]) -> None:
    states = sorted(set(r["state"] for r in rows))
    for state in states:
        sub = [r for r in rows if r["state"] == state]
        by_kind = {r["kind"]: r for r in sub}
        print(f"\nState: {state}")
        print(
            "  rel grad error vs reference | alias_only={:.3e}, trunc_only={:.3e}, standard={:.3e}".format(
                by_kind["alias_only"]["rel_err_vs_reference"],
                by_kind["trunc_only"]["rel_err_vs_reference"],
                by_kind["standard"]["rel_err_vs_reference"],
            )
        )
        print(
            "  rel grad error vs exact     | reference={:.3e}, standard={:.3e}".format(
                by_kind["reference"]["rel_err_vs_exact"],
                by_kind["standard"]["rel_err_vs_exact"],
            )
        )
        harder = "aliasing" if by_kind["alias_only"]["rel_err_vs_reference"] > by_kind["trunc_only"]["rel_err_vs_reference"] else "truncation"
        print(f"  larger isolated source at this state: {harder}")


def print_cg_summary(rows: List[Dict[str, float]]) -> None:
    states = sorted(set(r["state"] for r in rows))
    for state in states:
        sub = [r for r in rows if r["state"] == state]
        print(f"\nCG state: {state}")
        for row in sub:
            print(
                "  tol={:.0e} | rel_err_vs_tight={:.3e} | total={:.4f}s | meanCG={:.0f} | trace med/max={:.0f}/{:.0f}".format(
                    row["cg_tol"],
                    row["rel_err_vs_tight_cg"],
                    row["stage_total_sec"],
                    row["mean_cg_iters"],
                    row["trace_cg_iters_median"],
                    row["trace_cg_iters_max"],
                )
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--d", type=int, default=2)
    parser.add_argument("--noise-variance", type=float, default=0.2)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--nufft-eps", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--refine-h", type=int, default=4)
    parser.add_argument("--refine-cutoff", type=int, default=2)
    parser.add_argument("--trace-samples", type=int, default=128)
    parser.add_argument(
        "--bias-csv",
        type=Path,
        default=REPO_DIR / "experiments" / "efgpnd_bias_decomposition.csv",
    )
    parser.add_argument(
        "--cg-csv",
        type=Path,
        default=REPO_DIR / "experiments" / "efgpnd_cg_bias_sweep.csv",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    x, y = make_dataset(args.n, args.d, args.noise_variance, dtype, args.seed)

    states = [
        ("early", 0.50, 0.80, 0.16),
        ("mid", 0.23, 1.50, 0.25),
        ("late", 0.18, 1.30, 0.18),
    ]

    bias_rows = []
    cg_rows = []
    for label, lengthscale, variance, sigmasq in states:
        bias_rows.extend(
            bias_rows_for_state(
                x,
                y,
                label=label,
                lengthscale=lengthscale,
                variance=variance,
                sigmasq=sigmasq,
                eps=args.eps,
                refine_h=args.refine_h,
                refine_cutoff=args.refine_cutoff,
            )
        )
        cg_rows.extend(
            cg_rows_for_state(
                x,
                y,
                label=label,
                lengthscale=lengthscale,
                variance=variance,
                sigmasq=sigmasq,
                eps=args.eps,
                nufft_eps=args.nufft_eps,
                trace_samples=args.trace_samples,
                seed=12345,
                cg_tols=[1e-1, 1e-2, 1e-3, 1e-4, 1e-6],
            )
        )

    write_csv(args.bias_csv, bias_rows)
    write_csv(args.cg_csv, cg_rows)
    print_bias_summary(bias_rows)
    print_cg_summary(cg_rows)
    print(f"\nBias CSV written to {args.bias_csv}")
    print(f"CG CSV written to {args.cg_csv}")


if __name__ == "__main__":
    main()
