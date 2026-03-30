#!/usr/bin/env python3
"""
Benchmark cheap diagonal preconditioners for EFGPND CG solves on real data.

This script does not modify `efgpnd.py`. It builds the same linear systems used
by the gradient code, then compares:

- no preconditioner
- diagonal `|w|^2 + sigma_n^2`
- diagonal `c |w|^2 + sigma_n^2` for a few cheap scalar choices

The scalar `c` is motivated by the exact Toeplitz diagonal level, which on this
operator is the zero-lag convolution value and is equal to `N`.
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
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch

torch.set_num_threads(1)

REPO_DIR = Path(__file__).resolve().parent
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
from utils.kernels import get_xis  # noqa: E402


@dataclass
class OperatorBundle:
    regime: str
    n: int
    M: int
    mtot: int
    h: float
    sigmasq: float
    diag_toeplitz: float
    ws_abs2: torch.Tensor
    rhs_mean: torch.Tensor
    rhs_trace: torch.Tensor
    A_apply: object


def load_usa_temp_standardized() -> Tuple[torch.Tensor, torch.Tensor]:
    data = torch.load(REPO_DIR / "data" / "usa_temp_data.pt")
    x = data["x"].to(dtype=torch.float64)
    y = data["y"].to(dtype=torch.float64)
    x = (x - x.min(dim=0).values) / (x.max(dim=0).values - x.min(dim=0).values)
    y = (y - y.mean()) / y.std()
    return x, y


def make_kernel(lengthscale: float, variance: float) -> SquaredExponential:
    kernel = SquaredExponential(dimension=2)
    kernel.set_hyper("lengthscale", lengthscale)
    kernel.set_hyper("variance", variance)
    return kernel


def build_bundle(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    regime: str,
    lengthscale: float,
    variance: float,
    sigmasq: float,
    eps: float,
    nufft_eps: float,
    trace_samples: int,
    seed: int,
) -> OperatorBundle:
    kernel = make_kernel(lengthscale, variance)
    dtype = x.dtype
    cdtype = _cmplx(dtype)
    d = x.shape[1]
    L = float((x.max(dim=0).values - x.min(dim=0).values).max().item())

    xis_1d, h, mtot = get_xis(kernel, eps=eps, L=L, use_integral=True, l2scaled=False)
    xis_1d = xis_1d.to(dtype=dtype)
    xis = torch.stack(torch.meshgrid(*(xis_1d for _ in range(d)), indexing="ij"), dim=-1).reshape(-1, d)
    ws = torch.sqrt(kernel.spectral_density(xis).to(dtype=cdtype) * h**d)
    ws_abs2 = ws.abs().pow(2).real
    Dprime = (h**d * kernel.spectral_grad(xis)).to(cdtype)

    OUT = (mtot,) * d
    xcen = torch.zeros(d, dtype=dtype)
    nufft = NUFFT(x, xcen, h, nufft_eps, cdtype=cdtype)
    fadj = lambda v: nufft.type1(v, out_shape=OUT).reshape(-1)

    m_conv = (mtot - 1) // 2
    v_kernel = compute_convolution_vector_vectorized_dD(m_conv, x, h).to(dtype=cdtype)
    toeplitz = ToeplitzND(v_kernel, force_pow2=True)
    A_apply = create_A_mean(ws, toeplitz, torch.tensor(sigmasq, dtype=dtype), cdtype)

    Fy = fadj(y).reshape(-1)
    rhs_mean = ws * Fy

    torch.manual_seed(seed)
    Z = torch.empty((trace_samples, x.shape[0]), dtype=dtype)
    Z.bernoulli_(0.5)
    Z.mul_(2).sub_(1)
    Z = Z.to(cdtype)
    fadjZ = fadj(Z).reshape(trace_samples, -1)
    Di_FZ_all = torch.stack([Dprime[:, i] * fadjZ for i in range(2)], dim=0).reshape(-1, fadjZ.shape[-1])
    B_all_kernel = ws * toeplitz(Di_FZ_all).reshape(2, trace_samples, -1)
    B_noise = ws * fadjZ
    rhs_trace = torch.cat((B_all_kernel, B_noise.unsqueeze(0)), dim=0).reshape(3 * trace_samples, -1)

    center = tuple(((torch.tensor(v_kernel.shape) - 1) // 2).tolist())
    diag_toeplitz = float(v_kernel[center].real.item())
    return OperatorBundle(
        regime=regime,
        n=x.shape[0],
        M=ws.numel(),
        mtot=mtot,
        h=h,
        sigmasq=sigmasq,
        diag_toeplitz=diag_toeplitz,
        ws_abs2=ws_abs2,
        rhs_mean=rhs_mean,
        rhs_trace=rhs_trace,
        A_apply=A_apply,
    )


def make_preconditioner(name: str, bundle: OperatorBundle):
    if name == "none":
        return None, "none", float("nan")
    if name == "diag_ws2":
        c = 1.0
    elif name == "diag_10ws2":
        c = 10.0
    elif name == "diag_100ws2":
        c = 100.0
    elif name == "diag_1000ws2":
        c = 1000.0
    elif name == "diag_Nws2":
        c = bundle.diag_toeplitz
    else:
        raise ValueError(f"Unknown preconditioner {name}")

    denom = c * bundle.ws_abs2 + bundle.sigmasq
    def M_inv(v):
        return v / denom

    return M_inv, name, c


def solve_with_preconditioner(
    bundle: OperatorBundle,
    *,
    solve_kind: str,
    M_inv,
    tol: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    rhs = bundle.rhs_mean if solve_kind == "mean" else bundle.rhs_trace
    solver = RecordingConjugateGradients(
        bundle.A_apply,
        rhs,
        torch.zeros_like(rhs),
        tol=tol,
        early_stopping=True,
        M_inv_apply=M_inv,
    )
    soln, stats = solver.solve()
    row = {
        "iters_completed": float(stats.iters_completed),
        "iters_median": float(stats.iters_median),
        "iters_max": float(stats.iters_max),
        "solve_time_sec": stats.solve_time_sec,
        "apply_time_sec": stats.apply_time_sec,
        "rel_res_mean": stats.rel_res_mean,
        "rel_res_median": stats.rel_res_median,
        "rel_res_max": stats.rel_res_max,
        "converged_fraction": stats.converged_fraction,
    }
    return soln, row


def relative_solution_error(soln: torch.Tensor, ref: torch.Tensor) -> float:
    num = torch.linalg.norm((soln - ref).reshape(-1)).item()
    den = torch.linalg.norm(ref.reshape(-1)).item()
    return num / den if den > 0 else num


def run_suite_for_bundle(
    bundle: OperatorBundle,
    *,
    tol: float,
    reference_tol: float,
    reference_prec: str,
    prec_names: Iterable[str],
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    ref_minv, _, _ = make_preconditioner(reference_prec, bundle)
    ref_mean, _ = solve_with_preconditioner(bundle, solve_kind="mean", M_inv=ref_minv, tol=reference_tol)
    ref_trace, _ = solve_with_preconditioner(bundle, solve_kind="trace", M_inv=ref_minv, tol=reference_tol)

    for prec_name in prec_names:
        M_inv, _, c = make_preconditioner(prec_name, bundle)
        for solve_kind, ref_soln in (("mean", ref_mean), ("trace", ref_trace)):
            soln, stats = solve_with_preconditioner(bundle, solve_kind=solve_kind, M_inv=M_inv, tol=tol)
            row = {
                "regime": bundle.regime,
                "solve_kind": solve_kind,
                "preconditioner": prec_name,
                "c_scale": c,
                "n": float(bundle.n),
                "M": float(bundle.M),
                "mtot": float(bundle.mtot),
                "h": float(bundle.h),
                "diag_toeplitz": bundle.diag_toeplitz,
                "tol": tol,
                "reference_tol": reference_tol,
                "rel_soln_err_vs_ref": relative_solution_error(soln, ref_soln),
            }
            row.update(stats)
            rows.append(row)
    return rows


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: List[Dict[str, float]]) -> None:
    regimes = sorted(set(r["regime"] for r in rows))
    for regime in regimes:
        print(f"\nRegime: {regime}")
        for solve_kind in ("mean", "trace"):
            sub = [r for r in rows if r["regime"] == regime and r["solve_kind"] == solve_kind]
            base = next(r for r in sub if r["preconditioner"] == "none")
            print(f"  {solve_kind}: M={int(base['M'])}, tol={base['tol']:.0e}")
            for row in sub:
                speedup = base["solve_time_sec"] / row["solve_time_sec"]
                if solve_kind == "mean":
                    iter_str = f"{int(row['iters_completed'])}"
                else:
                    iter_str = f"{row['iters_median']:.0f}/{row['iters_max']:.0f}"
                print(
                    "    {:>11} | iters {} | time {:.3f}s | speedup {:.2f}x | soln err {:.2e}".format(
                        row["preconditioner"],
                        iter_str,
                        row["solve_time_sec"],
                        speedup,
                        row["rel_soln_err_vs_ref"],
                    )
                )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--nufft-eps", type=float, default=1e-5)
    parser.add_argument("--trace-samples", type=int, default=3)
    parser.add_argument("--cg-tol", type=float, default=1e-3)
    parser.add_argument("--reference-tol", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--csv",
        type=Path,
        default=REPO_DIR / "experiments" / "cg_preconditioning_realdata.csv",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    x, y = load_usa_temp_standardized()

    regimes = [
        ("hard", 0.04, 10.0, 1e-4),
        ("very_hard", 0.03, 10.0, 1e-4),
    ]
    prec_names = ["none", "diag_ws2", "diag_10ws2", "diag_100ws2", "diag_1000ws2", "diag_Nws2"]

    rows: List[Dict[str, float]] = []
    for regime, lengthscale, variance, sigmasq in regimes:
        bundle = build_bundle(
            x,
            y,
            regime=regime,
            lengthscale=lengthscale,
            variance=variance,
            sigmasq=sigmasq,
            eps=args.eps,
            nufft_eps=args.nufft_eps,
            trace_samples=args.trace_samples,
            seed=args.seed,
        )
        rows.extend(
            run_suite_for_bundle(
                bundle,
                tol=args.cg_tol,
                reference_tol=args.reference_tol,
                reference_prec="diag_100ws2",
                prec_names=prec_names,
            )
        )

    write_csv(args.csv, rows)
    print_summary(rows)
    print(f"\nCSV written to {args.csv}")


if __name__ == "__main__":
    main()
