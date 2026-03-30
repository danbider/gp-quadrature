#!/usr/bin/env python3
"""
Show which batched trace-CG right-hand sides are responsible for the long tail.

This script uses the same real-data operator as the gradient code and reports
per-RHS CG iteration counts for the lengthscale, variance, and noise blocks.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch

torch.set_num_threads(1)

REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from benchmark_cg_preconditioning_realdata import build_bundle, load_usa_temp_standardized, make_preconditioner  # noqa: E402


def per_rhs_iterations(bundle, tol: float, M_inv=None) -> List[int]:
    b = bundle.rhs_trace
    x = torch.zeros_like(b)
    r = b - bundle.A_apply(x)
    z = M_inv(r) if M_inv is not None else r.clone()
    p = z.clone()
    r_dot_z = torch.sum(r.conj() * z, dim=1).real
    b_norm = torch.linalg.norm(b, dim=1).real
    denom = torch.where(b_norm > 0, b_norm, torch.ones_like(b_norm))
    active = torch.ones(x.shape[0], dtype=torch.bool)
    iters = torch.zeros(x.shape[0], dtype=torch.int64)
    div_eps = 1e-16

    for i in range(2 * b.shape[1]):
        idx = torch.where(active)[0]
        if idx.numel() == 0:
            break
        Ap = bundle.A_apply(p[idx])
        pAp = torch.sum(p[idx].conj() * Ap, dim=1).real + div_eps
        alpha = r_dot_z[idx] / pAp
        x[idx] += alpha.unsqueeze(1) * p[idx]
        r[idx] -= alpha.unsqueeze(1) * Ap
        z_new = M_inv(r[idx]) if M_inv is not None else r[idx]
        r_dot_z_new = torch.sum(r[idx].conj() * z_new, dim=1).real
        beta = r_dot_z_new / (r_dot_z[idx] + div_eps)
        p[idx] = z_new + beta.unsqueeze(1) * p[idx]
        r_dot_z[idx] = r_dot_z_new
        r_norm = torch.linalg.norm(r[idx], dim=1).real
        rel_res = r_norm / (denom[idx] + div_eps)
        converged = (rel_res < tol) | (r_norm < 1e-12)
        if torch.any(converged):
            done = idx[converged]
            iters[done] = i + 1
            active[done] = False

    iters[active] = 2 * b.shape[1]
    return iters.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengthscale", type=float, default=0.03)
    parser.add_argument("--variance", type=float, default=10.0)
    parser.add_argument("--sigmasq", type=float, default=1e-4)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--nufft-eps", type=float, default=1e-5)
    parser.add_argument("--trace-samples", type=int, default=3)
    parser.add_argument("--cg-tol", type=float, default=1e-3)
    args = parser.parse_args()

    x, y = load_usa_temp_standardized()
    bundle = build_bundle(
        x,
        y,
        regime="trace_blocks",
        lengthscale=args.lengthscale,
        variance=args.variance,
        sigmasq=args.sigmasq,
        eps=args.eps,
        nufft_eps=args.nufft_eps,
        trace_samples=args.trace_samples,
        seed=0,
    )

    labels = (
        [f"dlengthscale_{i+1}" for i in range(args.trace_samples)]
        + [f"dvariance_{i+1}" for i in range(args.trace_samples)]
        + [f"dsigmanoise_{i+1}" for i in range(args.trace_samples)]
    )

    for prec_name in ("none", "diag_ws2", "diag_100ws2"):
        M_inv, _, _ = make_preconditioner(prec_name, bundle)
        iters = per_rhs_iterations(bundle, args.cg_tol, M_inv)
        print(f"\nPreconditioner: {prec_name}")
        for label, it in zip(labels, iters):
            print(f"  {label:<16} {it}")


if __name__ == "__main__":
    main()
