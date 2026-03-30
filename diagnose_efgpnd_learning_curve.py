#!/usr/bin/env python3
"""
Standalone diagnostics for EFGPND hyperparameter-learning bottlenecks.

This script does not modify `efgpnd.py`. It mirrors the current gradient path,
records stage timings, feature-grid sizes, and conjugate-gradient behavior, and
then runs a learning curve similar to `test_timing_profiling.py`.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch.optim import Adam

REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from efgpnd import (  # noqa: E402
    EFGPND,
    NUFFT,
    ToeplitzND,
    _cmplx,
    compute_convolution_vector_vectorized_dD,
    create_A_mean,
)
from utils.kernels import get_xis  # noqa: E402


def now() -> float:
    return time.perf_counter()


@dataclass
class CGStats:
    is_batched: bool
    batch_size: int
    system_size: int
    iters_completed: int
    max_iter: int
    apply_calls: int
    apply_time_sec: float
    solve_time_sec: float
    rel_res_min: float
    rel_res_median: float
    rel_res_mean: float
    rel_res_max: float
    iters_min: float
    iters_median: float
    iters_mean: float
    iters_max: float
    converged_fraction: float


class RecordingConjugateGradients:
    """
    External copy of the current CG routine with extra instrumentation.
    """

    def __init__(
        self,
        A_apply_function,
        b: torch.Tensor,
        x0: torch.Tensor,
        *,
        tol: float = 1e-6,
        max_iter: int | None = None,
        early_stopping: bool = True,
        M_inv_apply=None,
    ):
        self.device = b.device
        self.dtype = x0.dtype
        self.A_apply_function = A_apply_function
        self.b = b.to(dtype=self.dtype, device=self.device)
        self.x0 = x0.to(dtype=self.dtype, device=self.device)
        self.is_batched = self.b.ndim > 1
        self.tol = tol
        self.div_eps = 1e-16
        self.early_stopping = early_stopping
        self.M_inv_apply = M_inv_apply
        system_size = self.b.shape[1] if self.is_batched else self.b.numel()
        self.max_iter = max_iter if max_iter is not None else 2 * system_size
        self.iters_completed = 0
        self.apply_calls = 0
        self.apply_time_sec = 0.0

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        t0 = now()
        out = self.A_apply_function(x)
        self.apply_time_sec += now() - t0
        self.apply_calls += 1
        return out

    def solve(self) -> Tuple[torch.Tensor, CGStats]:
        if self.is_batched:
            return self._solve_batched()
        return self._solve_single()

    def _solve_single(self) -> Tuple[torch.Tensor, CGStats]:
        t_start = now()
        with torch.no_grad():
            x = self.x0.clone()
            r = self.b - self._apply(x)

            z = self.M_inv_apply(r) if self.M_inv_apply is not None else r.clone()
            p = z.clone()
            r_dot_z = torch.dot(torch.conj(r), z).real
            b_norm = torch.linalg.norm(self.b).real
            denom = b_norm if b_norm > 0 else torch.tensor(1.0, device=self.device, dtype=self.dtype)

            converged = False
            for i in range(self.max_iter):
                Ap = self._apply(p)
                pAp = torch.dot(torch.conj(p), Ap).real + self.div_eps
                alpha = r_dot_z / pAp
                x = x + alpha * p
                r = r - alpha * Ap

                r_norm = torch.linalg.norm(r).real
                rel_res = r_norm / (denom + self.div_eps)
                if self.early_stopping and (rel_res < self.tol or r_norm < 1e-12):
                    converged = True
                    self.iters_completed = i + 1
                    break

                z_new = self.M_inv_apply(r) if self.M_inv_apply is not None else r
                r_dot_z_new = torch.dot(torch.conj(r), z_new).real
                beta = r_dot_z_new / (r_dot_z + self.div_eps)
                p = z_new + beta * p
                r_dot_z = r_dot_z_new

            if self.iters_completed == 0:
                self.iters_completed = self.max_iter
                r_norm = torch.linalg.norm(r).real
                rel_res = r_norm / (denom + self.div_eps)

        solve_time = now() - t_start
        rel_res_val = float(rel_res.item())
        iters_val = float(self.iters_completed)
        stats = CGStats(
            is_batched=False,
            batch_size=1,
            system_size=self.b.numel(),
            iters_completed=self.iters_completed,
            max_iter=self.max_iter,
            apply_calls=self.apply_calls,
            apply_time_sec=self.apply_time_sec,
            solve_time_sec=solve_time,
            rel_res_min=rel_res_val,
            rel_res_median=rel_res_val,
            rel_res_mean=rel_res_val,
            rel_res_max=rel_res_val,
            iters_min=iters_val,
            iters_median=iters_val,
            iters_mean=iters_val,
            iters_max=iters_val,
            converged_fraction=1.0 if converged else 0.0,
        )
        return x, stats

    def _solve_batched(self) -> Tuple[torch.Tensor, CGStats]:
        t_start = now()
        with torch.no_grad():
            x = self.x0.clone()
            r = self.b - self._apply(x)

            z = self.M_inv_apply(r) if self.M_inv_apply is not None else r.clone()
            p = z.clone()
            r_dot_z = torch.sum(r.conj() * z, dim=1).real
            b_norm = torch.linalg.norm(self.b, dim=1).real
            denom = torch.where(b_norm > 0, b_norm, torch.ones_like(b_norm))

            active = torch.ones(x.shape[0], dtype=torch.bool, device=self.device)
            iters_per_system = torch.zeros(x.shape[0], dtype=torch.int64, device=self.device)

            for i in range(self.max_iter):
                idx = torch.where(active)[0]
                if idx.numel() == 0:
                    self.iters_completed = i
                    break

                Ap = self._apply(p[idx])
                pAp = torch.sum(p[idx].conj() * Ap, dim=1).real + self.div_eps
                alpha = r_dot_z[idx] / pAp

                x[idx] += alpha.unsqueeze(1) * p[idx]
                r[idx] -= alpha.unsqueeze(1) * Ap

                z_new = self.M_inv_apply(r[idx]) if self.M_inv_apply is not None else r[idx]
                r_dot_z_new = torch.sum(r[idx].conj() * z_new, dim=1).real
                beta = r_dot_z_new / (r_dot_z[idx] + self.div_eps)
                p[idx] = z_new + beta.unsqueeze(1) * p[idx]
                r_dot_z[idx] = r_dot_z_new

                if self.early_stopping:
                    r_norm = torch.linalg.norm(r[idx], dim=1).real
                    rel_res = r_norm / (denom[idx] + self.div_eps)
                    converged = (rel_res < self.tol) | (r_norm < 1e-12)
                    if torch.any(converged):
                        done_idx = idx[converged]
                        iters_per_system[done_idx] = i + 1
                        active[done_idx] = False

            if self.iters_completed == 0:
                self.iters_completed = self.max_iter

            unfinished = active & (iters_per_system == 0)
            iters_per_system[unfinished] = self.iters_completed
            r_norm_all = torch.linalg.norm(r, dim=1).real
            rel_res_all = r_norm_all / (denom + self.div_eps)

        solve_time = now() - t_start
        rel_res_cpu = rel_res_all.detach().cpu()
        iters_cpu = iters_per_system.detach().cpu().to(torch.float64)
        converged_fraction = float((rel_res_cpu < self.tol).to(torch.float64).mean().item())
        stats = CGStats(
            is_batched=True,
            batch_size=self.b.shape[0],
            system_size=self.b.shape[1],
            iters_completed=self.iters_completed,
            max_iter=self.max_iter,
            apply_calls=self.apply_calls,
            apply_time_sec=self.apply_time_sec,
            solve_time_sec=solve_time,
            rel_res_min=float(rel_res_cpu.min().item()),
            rel_res_median=float(rel_res_cpu.median().item()),
            rel_res_mean=float(rel_res_cpu.mean().item()),
            rel_res_max=float(rel_res_cpu.max().item()),
            iters_min=float(iters_cpu.min().item()),
            iters_median=float(iters_cpu.median().item()),
            iters_mean=float(iters_cpu.mean().item()),
            iters_max=float(iters_cpu.max().item()),
            converged_fraction=converged_fraction,
        )
        return x, stats


def synthetic_function(x: torch.Tensor) -> torch.Tensor:
    return (
        torch.sin(3 * x[:, 0]) * torch.cos(4 * x[:, 1])
        + 0.5 * torch.exp(-((x[:, 0] - 0.3) ** 2 + (x[:, 1] + 0.3) ** 2) / 0.3)
        + 0.7 * torch.sin(2 * torch.pi * (x[:, 0] ** 2 + x[:, 1] ** 2))
    )


def make_dataset(n: int, d: int, noise_variance: float, dtype: torch.dtype, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    x = torch.rand(n, d, dtype=dtype) * 2 - 1
    if d != 2:
        raise ValueError("This diagnostic script currently mirrors the 2D timing example only.")
    y = synthetic_function(x) + torch.randn(n, dtype=dtype) * math.sqrt(noise_variance)
    return x, y


def instrumented_gradient_step(
    x: torch.Tensor,
    y: torch.Tensor,
    sigmasq: torch.Tensor,
    kernel,
    eps: float,
    trace_samples: int,
    x0: torch.Tensor,
    x1: torch.Tensor,
    *,
    nufft_eps: float,
    cg_tol: float,
    early_stopping: bool = True,
    seed: int | None = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    metrics: Dict[str, float] = {}
    t_total = now()

    device = x.device
    dtype = x.dtype
    cdtype = _cmplx(dtype)
    if x.ndim == 1:
        x = x.unsqueeze(-1)
    d = x.shape[1]
    n = x.shape[0]
    L = float((x1 - x0).max().item())
    sigmasq_scalar = sigmasq

    t0 = now()
    xis_1d, h, mtot = get_xis(kernel_obj=kernel, eps=eps, L=L, use_integral=True, l2scaled=False)
    xis_1d = xis_1d.to(device=device, dtype=dtype)
    grids = torch.meshgrid(*(xis_1d for _ in range(d)), indexing="ij")
    xis = torch.stack(grids, dim=-1).view(-1, d)
    ws = torch.sqrt(kernel.spectral_density(xis).to(dtype=cdtype) * h**d)
    Dprime = (h**d * kernel.spectral_grad(xis)).to(cdtype)
    metrics["stage_frequency_grid_sec"] = now() - t0

    M = int(xis.shape[0])
    num_hypers = kernel.num_hypers
    metrics["n"] = float(n)
    metrics["d"] = float(d)
    metrics["mtot"] = float(mtot)
    metrics["M"] = float(M)
    metrics["M_over_n"] = float(M / n)
    metrics["B"] = float(num_hypers * trace_samples)
    metrics["batch_unknowns"] = float(num_hypers * trace_samples * M)
    metrics["h"] = float(h)
    metrics["lengthscale"] = float(kernel.get_hyper("lengthscale"))
    metrics["variance"] = float(kernel.get_hyper("variance"))
    metrics["sigmasq"] = float(sigmasq.item())

    t0 = now()
    OUT = (mtot,) * d
    xcen = torch.zeros(d, device=device, dtype=dtype)
    nufft_op = NUFFT(x, xcen, h, nufft_eps, cdtype=cdtype, device=device)
    fadj = lambda v: nufft_op.type1(v, out_shape=OUT).reshape(-1)
    fwd = lambda fk: nufft_op.type2(fk, out_shape=OUT)
    metrics["stage_nufft_setup_sec"] = now() - t0

    t0 = now()
    m_conv = (mtot - 1) // 2
    v_kernel = compute_convolution_vector_vectorized_dD(m_conv, x, h).to(dtype=cdtype)
    toeplitz = ToeplitzND(v_kernel, force_pow2=True)
    A_apply = create_A_mean(ws, toeplitz, sigmasq_scalar, cdtype)
    metrics["stage_toeplitz_setup_sec"] = now() - t0

    t0 = now()
    Fy = fadj(y).reshape(-1)
    rhs = ws * Fy
    metrics["stage_mean_rhs_sec"] = now() - t0

    mean_solver = RecordingConjugateGradients(
        A_apply,
        rhs,
        torch.zeros_like(rhs),
        tol=cg_tol,
        early_stopping=early_stopping,
    )
    beta, mean_stats = mean_solver.solve()
    beta.mul_(ws)

    t0 = now()
    z = fwd(beta)
    alpha = y.to(dtype=cdtype).clone()
    alpha.sub_(z)
    alpha.div_(sigmasq_scalar)
    metrics["stage_mean_post_sec"] = now() - t0

    t0 = now()
    fadj_alpha = Fy.clone().sub_(toeplitz(beta)).div(sigmasq_scalar)
    term2 = torch.stack(
        (
            torch.vdot(fadj_alpha, Dprime[:, 0] * fadj_alpha),
            torch.vdot(fadj_alpha, Dprime[:, 1] * fadj_alpha),
            torch.vdot(alpha, alpha),
        )
    )
    metrics["stage_term2_sec"] = now() - t0

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed % (2**32 - 1))

    t0 = now()
    T = trace_samples
    Z = torch.empty((T, n), device=device, dtype=dtype)
    Z.bernoulli_(0.5)
    Z.mul_(2).sub_(1)
    Z = Z.to(cdtype)
    fadjZ = fadj(Z)
    fadjZ_flat = fadjZ.reshape(T, -1)
    Hk = num_hypers - 1
    Di_FZ_all = torch.stack([Dprime[:, i] * fadjZ_flat for i in range(Hk)], dim=0).reshape(-1, fadjZ_flat.shape[-1])
    metrics["stage_trace_probe_setup_sec"] = now() - t0

    t0 = now()
    rhs_all_kernel = fwd(Di_FZ_all).reshape(Hk, T, -1)
    rhs_noise = Z
    R_all = torch.cat((rhs_all_kernel, rhs_noise.unsqueeze(0)), dim=0).reshape(num_hypers * T, -1)
    metrics["stage_trace_rhs_build_sec"] = now() - t0

    t0 = now()
    B_all_kernel = ws * toeplitz(Di_FZ_all).reshape(Hk, T, -1)
    B_noise = ws * fadjZ_flat
    B_all = torch.cat((B_all_kernel, B_noise.unsqueeze(0)), dim=0).reshape(num_hypers * T, -1)
    metrics["stage_trace_B_build_sec"] = now() - t0

    trace_solver = RecordingConjugateGradients(
        A_apply,
        B_all,
        torch.zeros_like(B_all),
        tol=cg_tol,
        early_stopping=early_stopping,
    )
    Beta_all, trace_stats = trace_solver.solve()
    Beta_all.mul_(ws)

    t0 = now()
    fwdBeta = fwd(Beta_all)
    R_all.sub_(fwdBeta)
    R_all.div_(sigmasq_scalar)
    Alpha_batch = R_all.view(num_hypers, T, -1)
    term1 = (Z.unsqueeze(0) * Alpha_batch).sum(dim=2).mean(1)
    grad = 0.5 * (term1 - term2)
    metrics["stage_trace_post_sec"] = now() - t0

    metrics["stage_total_sec"] = now() - t_total
    metrics["stage_mean_cg_sec"] = mean_stats.solve_time_sec
    metrics["stage_trace_cg_sec"] = trace_stats.solve_time_sec

    metrics["mean_cg_iters"] = float(mean_stats.iters_completed)
    metrics["mean_cg_apply_calls"] = float(mean_stats.apply_calls)
    metrics["mean_cg_apply_time_sec"] = mean_stats.apply_time_sec
    metrics["mean_cg_non_apply_time_sec"] = mean_stats.solve_time_sec - mean_stats.apply_time_sec
    metrics["mean_cg_rel_res"] = mean_stats.rel_res_max

    metrics["trace_cg_iters_max"] = float(trace_stats.iters_max)
    metrics["trace_cg_iters_mean"] = float(trace_stats.iters_mean)
    metrics["trace_cg_iters_median"] = float(trace_stats.iters_median)
    metrics["trace_cg_apply_calls"] = float(trace_stats.apply_calls)
    metrics["trace_cg_apply_time_sec"] = trace_stats.apply_time_sec
    metrics["trace_cg_non_apply_time_sec"] = trace_stats.solve_time_sec - trace_stats.apply_time_sec
    metrics["trace_cg_rel_res_max"] = trace_stats.rel_res_max
    metrics["trace_cg_rel_res_mean"] = trace_stats.rel_res_mean
    metrics["trace_cg_rel_res_median"] = trace_stats.rel_res_median
    metrics["trace_cg_converged_fraction"] = trace_stats.converged_fraction
    metrics["M_log2_mtot"] = float(M * math.log2(max(mtot, 2)))
    metrics["trace_work_proxy"] = float(trace_stats.apply_calls * M * math.log2(max(mtot, 2)))
    return grad.real, metrics


def raw_gradient_from_positive_gradient(model: EFGPND, grads: torch.Tensor) -> torch.Tensor:
    pos_vec = model._gp_params.pos.to(device=model.x.device)
    param_dtype = model._gp_params.raw.dtype
    return torch.stack(
        [
            grads[i].detach().to(device=model.x.device, dtype=param_dtype) * pos_vec[i]
            for i in range(len(grads))
        ],
        dim=0,
    )


def summarize_rows(rows: List[Dict[str, float]]) -> str:
    if not rows:
        return "No rows recorded."

    first = rows[0]
    mid = rows[len(rows) // 2]
    last = rows[-1]
    worst = max(rows, key=lambda r: r["stage_total_sec"])

    def ratio(a: float, b: float) -> float:
        return float("inf") if b == 0 else a / b

    lines = [
        "Learning-curve summary",
        f"  total step time: {first['stage_total_sec']:.4f}s -> {last['stage_total_sec']:.4f}s "
        f"({ratio(last['stage_total_sec'], first['stage_total_sec']):.2f}x)",
        f"  M = m^d: {int(first['M'])} -> {int(last['M'])} "
        f"({ratio(last['M'], first['M']):.2f}x)",
        f"  mean CG iterations: {first['mean_cg_iters']:.0f} -> {last['mean_cg_iters']:.0f} "
        f"({ratio(last['mean_cg_iters'], first['mean_cg_iters']):.2f}x)",
        f"  trace CG median iterations: {first['trace_cg_iters_median']:.0f} -> {last['trace_cg_iters_median']:.0f} "
        f"({ratio(last['trace_cg_iters_median'], first['trace_cg_iters_median']):.2f}x)",
        f"  trace CG max iterations: {first['trace_cg_iters_max']:.0f} -> {last['trace_cg_iters_max']:.0f} "
        f"({ratio(last['trace_cg_iters_max'], first['trace_cg_iters_max']):.2f}x)",
        f"  lengthscale: {first['lengthscale']:.4g} -> {last['lengthscale']:.4g}",
        f"  variance: {first['variance']:.4g} -> {last['variance']:.4g}",
        f"  worst iteration: {int(worst['iter'])} with total {worst['stage_total_sec']:.4f}s, "
        f"M={int(worst['M'])}, trace CG max iters={worst['trace_cg_iters_max']:.0f}",
        f"  midpoint snapshot: iter {int(mid['iter'])}, "
        f"M={int(mid['M'])}, total={mid['stage_total_sec']:.4f}s, "
        f"trace CG max iters={mid['trace_cg_iters_max']:.0f}",
    ]
    return "\n".join(lines)


def write_csv(rows: List[Dict[str, float]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_learning_curve(args) -> List[Dict[str, float]]:
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    x, y = make_dataset(args.n, args.d, args.noise_variance, dtype, args.seed)
    model = EFGPND(x, y, kernel="SquaredExponential", eps=args.eps)
    optimizer = Adam(model.parameters(), lr=args.lr)

    rows: List[Dict[str, float]] = []
    x0 = x.min(dim=0).values
    x1 = x.max(dim=0).values

    for it in range(args.max_iters):
        trace_samples = args.early_trace_samples if it < args.late_after else args.late_trace_samples
        cg_tol = args.early_cg_tol if it < args.late_after else args.late_cg_tol
        optimizer.zero_grad()

        grads, metrics = instrumented_gradient_step(
            x=model.x,
            y=model.y,
            sigmasq=model.sigmasq,
            kernel=model.kernel,
            eps=model.eps,
            trace_samples=trace_samples,
            x0=x0,
            x1=x1,
            nufft_eps=args.nufft_eps,
            cg_tol=cg_tol,
            early_stopping=True,
            seed=args.seed + it,
        )

        raw_grad = raw_gradient_from_positive_gradient(model, grads)
        with torch.no_grad():
            model._gp_params.raw.grad = raw_grad.detach().clone()
        optimizer.step()
        model._update_param_cache()

        row = {
            "iter": float(it),
            "trace_samples": float(trace_samples),
            "cg_tol": float(cg_tol),
            "post_lengthscale": float(model.kernel.get_hyper("lengthscale")),
            "post_variance": float(model.kernel.get_hyper("variance")),
            "post_sigmasq": float(model.sigmasq.item()),
        }
        row.update(metrics)
        rows.append(row)

        if args.print_every and (it % args.print_every == 0 or it == args.max_iters - 1):
            print(
                "iter {it:>3} | total {total:.4f}s | M {M:>7} | "
                "ls {ls:.4g} -> {post_ls:.4g} | var {var:.4g} -> {post_var:.4g} | "
                "mean CG {mean_it:.0f} | trace CG med/max {trace_med:.0f}/{trace_max:.0f}".format(
                    it=it,
                    total=metrics["stage_total_sec"],
                    M=int(metrics["M"]),
                    ls=metrics["lengthscale"],
                    post_ls=row["post_lengthscale"],
                    var=metrics["variance"],
                    post_var=row["post_variance"],
                    mean_it=metrics["mean_cg_iters"],
                    trace_med=metrics["trace_cg_iters_median"],
                    trace_max=metrics["trace_cg_iters_max"],
                )
            )

    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--d", type=int, default=2)
    parser.add_argument("--noise-variance", type=float, default=0.2)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--nufft-eps", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--max-iters", type=int, default=30)
    parser.add_argument("--late-after", type=int, default=24)
    parser.add_argument("--early-trace-samples", type=int, default=5)
    parser.add_argument("--late-trace-samples", type=int, default=10)
    parser.add_argument("--early-cg-tol", type=float, default=1e-3)
    parser.add_argument("--late-cg-tol", type=float, default=1e-4)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument(
        "--csv",
        type=Path,
        default=REPO_DIR / "experiments" / "efgpnd_learning_curve_diagnostics.csv",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    print(
        f"Running EFGPND diagnostics with n={args.n}, d={args.d}, eps={args.eps}, "
        f"iters={args.max_iters}, dtype={args.dtype}"
    )
    rows = run_learning_curve(args)
    write_csv(rows, args.csv)
    print(summarize_rows(rows))
    print(f"CSV written to {args.csv}")


if __name__ == "__main__":
    main()
