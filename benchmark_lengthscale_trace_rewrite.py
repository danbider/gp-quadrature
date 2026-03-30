#!/usr/bin/env python3
"""
Benchmark the proposed feature-space lengthscale trace rewrite against the
current lengthscale trace block.

This script does not modify `efgpnd.py`.

It performs:
1. A small exact algebra check on a real-data subset.
2. A real-data benchmark on the full PRISM tmean raster in the bad regime.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch

torch.set_num_threads(1)

REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
PRISM_DIR = REPO_DIR / "prism_experiment"
if str(PRISM_DIR) not in sys.path:
    sys.path.insert(0, str(PRISM_DIR))

from diagnose_efgpnd_learning_curve import RecordingConjugateGradients  # noqa: E402
from efgpnd import (  # noqa: E402
    NUFFT,
    ToeplitzND,
    _cmplx,
    compute_convolution_vector_vectorized_dD,
    create_A_mean,
    create_jacobi_precond,
)
from kernels.squared_exponential import SquaredExponential  # noqa: E402
from load_prism import load_prism_dataset_torch  # noqa: E402
from utils.kernels import get_xis  # noqa: E402


@dataclass
class Bundle:
    x: torch.Tensor
    y: torch.Tensor
    dtype: torch.dtype
    cdtype: torch.dtype
    N: int
    M: int
    d: int
    mtot: int
    sigmasq: float
    variance: float
    ws: torch.Tensor
    d_l: torch.Tensor
    c0: float
    A_apply: object
    M_inv: object | None
    toeplitz: object
    fadj: object
    fwd: object


def load_usa_temp_subset(n: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.load(REPO_DIR / "data" / "usa_temp_data.pt")
    x = data["x"].to(dtype=torch.float64)[:n]
    y = data["y"].to(dtype=torch.float64)[:n]
    x = (x - x.min(dim=0).values) / (x.max(dim=0).values - x.min(dim=0).values)
    y = (y - y.mean()) / y.std()
    return x, y


def load_prism_standardized() -> tuple[torch.Tensor, torch.Tensor]:
    x, y = load_prism_dataset_torch("prism_tmean_us_30s_2020_avg_30y")
    x = x.to(dtype=torch.float64)
    y = y.to(dtype=torch.float64)
    x = (x - x.min(dim=0).values) / (x.max(dim=0).values - x.min(dim=0).values)
    y = (y - y.mean()) / y.std()
    return x, y


def make_bundle(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    lengthscale: float,
    variance: float,
    sigmasq: float,
    eps: float,
    nufft_eps: float,
    use_preconditioner: bool,
) -> Bundle:
    dtype = x.dtype
    cdtype = _cmplx(dtype)
    d = x.shape[1]
    N = x.shape[0]

    kernel = SquaredExponential(dimension=d)
    kernel.set_hyper("lengthscale", lengthscale)
    kernel.set_hyper("variance", variance)

    L = float((x.max(dim=0).values - x.min(dim=0).values).max().item())
    xis_1d, h, mtot = get_xis(kernel_obj=kernel, eps=eps, L=L, use_integral=True, l2scaled=False)
    xis = torch.stack(torch.meshgrid(*(xis_1d for _ in range(d)), indexing="ij"), dim=-1).view(-1, d)
    ws = torch.sqrt(kernel.spectral_density(xis).to(dtype=cdtype) * h**d)
    d_l = (h**d * kernel.spectral_grad(xis)[:, 0]).to(cdtype)

    out_shape = (mtot,) * d
    nufft = NUFFT(x, torch.zeros(d, dtype=dtype), h, nufft_eps, cdtype=cdtype)
    fadj = lambda v: nufft.type1(v, out_shape=out_shape).reshape(v.shape[:-1] + (-1,))
    fwd = lambda fk: nufft.type2(fk, out_shape=out_shape)

    m_conv = (mtot - 1) // 2
    v_kernel = compute_convolution_vector_vectorized_dD(m_conv, x, h).to(dtype=cdtype)
    toeplitz = ToeplitzND(v_kernel, force_pow2=True)
    A_apply = create_A_mean(ws, toeplitz, torch.tensor(sigmasq, dtype=dtype), cdtype)

    center = tuple(((torch.tensor(v_kernel.shape) - 1) // 2).tolist())
    c0 = float(v_kernel[center].real.item())
    M_inv = None
    if use_preconditioner:
        M_inv = create_jacobi_precond(ws, torch.tensor(sigmasq, dtype=dtype), diag_scale=v_kernel[center].real)

    return Bundle(
        x=x,
        y=y,
        dtype=dtype,
        cdtype=cdtype,
        N=N,
        M=ws.numel(),
        d=d,
        mtot=mtot,
        sigmasq=sigmasq,
        variance=variance,
        ws=ws,
        d_l=d_l,
        c0=c0,
        A_apply=A_apply,
        M_inv=M_inv,
        toeplitz=toeplitz,
        fadj=fadj,
        fwd=fwd,
    )


def make_rademacher(shape: tuple[int, ...], *, dtype: torch.dtype, cdtype: torch.dtype, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    z = torch.empty(shape, dtype=dtype)
    z.bernoulli_(0.5)
    z.mul_(2).sub_(1)
    return z.to(cdtype)


def build_explicit_f_matrix(bundle: Bundle, batch_size: int = 64) -> torch.Tensor:
    F = torch.empty((bundle.N, bundle.M), dtype=bundle.cdtype)
    eye = torch.eye(bundle.M, dtype=bundle.cdtype)
    for start in range(0, bundle.M, batch_size):
        stop = min(start + batch_size, bundle.M)
        coeffs = eye[start:stop]
        F[:, start:stop] = bundle.fwd(coeffs).T
    return F


def build_explicit_c(bundle: Bundle, batch_size: int = 64) -> torch.Tensor:
    C = torch.empty((bundle.M, bundle.M), dtype=bundle.cdtype)
    eye = torch.eye(bundle.M, dtype=bundle.cdtype)
    for start in range(0, bundle.M, batch_size):
        stop = min(start + batch_size, bundle.M)
        cols = bundle.toeplitz(eye[start:stop]).T
        C[:, start:stop] = cols
    return C


def exact_validation() -> None:
    print("== Exact algebra validation on usa_temp subset ==")
    x, y = load_usa_temp_subset(128)
    bundle = make_bundle(
        x,
        y,
        lengthscale=0.14,
        variance=1.9,
        sigmasq=0.07,
        eps=1e-4,
        nufft_eps=1e-6,
        use_preconditioner=False,
    )

    F = build_explicit_f_matrix(bundle)
    C = build_explicit_c(bundle)
    D = torch.diag(bundle.ws)
    S_l = torch.diag(bundle.d_l)
    K = F @ D @ D.conj().T @ F.conj().T + bundle.sigmasq * torch.eye(bundle.N, dtype=bundle.cdtype)
    dK_l = F @ S_l @ F.conj().T
    K_inv = torch.linalg.inv(K)
    trace_dense = torch.trace(K_inv @ dK_l).real

    A = D @ C @ D + bundle.sigmasq * torch.eye(bundle.M, dtype=bundle.cdtype)
    H_l = D @ C @ S_l @ C @ D
    trace_formula = (torch.trace(C @ S_l) - torch.trace(torch.linalg.solve(A, H_l))).real / bundle.sigmasq

    print(f"N={bundle.N}, mtot={bundle.mtot}, M={bundle.M}")
    print(f"dense trace            = {trace_dense.item():.12e}")
    print(f"feature-space formula  = {trace_formula.item():.12e}")
    print(f"abs diff               = {(trace_dense - trace_formula).abs().item():.12e}")
    rel_diff = (trace_dense - trace_formula).abs() / trace_dense.abs().clamp_min(1e-12)
    print(f"rel diff               = {rel_diff.item():.12e}")
    print(f"center(C)              = {bundle.c0:.6f}")
    print(f"mean diag(C)           = {torch.diagonal(C).real.mean().item():.6f}")
    if not torch.allclose(trace_dense, trace_formula, atol=1e-4, rtol=1e-7):
        raise SystemExit("lengthscale trace identity check failed")
    print("all checks passed")


def run_old_lengthscale(bundle: Bundle, *, trace_samples: int, cg_tol: float, seed: int) -> dict[str, float]:
    total_start = time.perf_counter()

    t0 = time.perf_counter()
    Z = make_rademacher((trace_samples, bundle.N), dtype=bundle.dtype, cdtype=bundle.cdtype, seed=seed)
    fadjZ = bundle.fadj(Z).reshape(trace_samples, -1)
    Di_FZ = bundle.d_l * fadjZ
    rhs_data = bundle.fwd(Di_FZ).reshape(trace_samples, -1)
    B_old = (bundle.ws * bundle.toeplitz(Di_FZ)).reshape(trace_samples, -1)
    rhs_build_sec = time.perf_counter() - t0

    solver = RecordingConjugateGradients(
        bundle.A_apply,
        B_old,
        torch.zeros_like(B_old),
        tol=cg_tol,
        early_stopping=True,
        M_inv_apply=bundle.M_inv,
    )
    Beta_old, stats = solver.solve()

    t1 = time.perf_counter()
    Alpha_old = (rhs_data - bundle.fwd(Beta_old * bundle.ws).reshape(trace_samples, -1)) / bundle.sigmasq
    estimate = (Z.conj() * Alpha_old).sum(dim=1).real.mean()
    post_sec = time.perf_counter() - t1

    return {
        "estimate": float(estimate.item()),
        "iters_completed": float(stats.iters_completed),
        "iters_mean": float(stats.iters_mean),
        "iters_median": float(stats.iters_median),
        "iters_max": float(stats.iters_max),
        "rhs_build_sec": rhs_build_sec,
        "solve_sec": stats.solve_time_sec,
        "post_sec": post_sec,
        "total_sec": time.perf_counter() - total_start,
        "apply_time_sec": stats.apply_time_sec,
        "rel_res_mean": float(stats.rel_res_mean),
        "rel_res_max": float(stats.rel_res_max),
    }


def run_new_lengthscale(bundle: Bundle, *, trace_samples: int, cg_tol: float, seed: int) -> dict[str, float]:
    total_start = time.perf_counter()

    t0 = time.perf_counter()
    V = make_rademacher((trace_samples, bundle.M), dtype=bundle.dtype, cdtype=bundle.cdtype, seed=seed)
    tmp = bundle.toeplitz(bundle.ws * V).reshape(trace_samples, -1)
    B_new = (bundle.ws * bundle.toeplitz(bundle.d_l * tmp)).reshape(trace_samples, -1)
    trace_const = bundle.c0 * bundle.d_l.real.sum() / bundle.sigmasq
    rhs_build_sec = time.perf_counter() - t0

    solver = RecordingConjugateGradients(
        bundle.A_apply,
        B_new,
        torch.zeros_like(B_new),
        tol=cg_tol,
        early_stopping=True,
        M_inv_apply=bundle.M_inv,
    )
    Beta_new, stats = solver.solve()

    t1 = time.perf_counter()
    estimate = trace_const - ((V.conj() * Beta_new).sum(dim=1).real / bundle.sigmasq).mean()
    post_sec = time.perf_counter() - t1

    return {
        "estimate": float(estimate.item()),
        "iters_completed": float(stats.iters_completed),
        "iters_mean": float(stats.iters_mean),
        "iters_median": float(stats.iters_median),
        "iters_max": float(stats.iters_max),
        "rhs_build_sec": rhs_build_sec,
        "solve_sec": stats.solve_time_sec,
        "post_sec": post_sec,
        "total_sec": time.perf_counter() - total_start,
        "apply_time_sec": stats.apply_time_sec,
        "rel_res_mean": float(stats.rel_res_mean),
        "rel_res_max": float(stats.rel_res_max),
    }


def benchmark_prism() -> None:
    print("\n== Full PRISM bad-regime benchmark ==")
    x, y = load_prism_standardized()
    print(f"N={x.shape[0]}")

    states = [
        ("iter40", 0.09256, 3.878, 0.05202),
        ("final", 0.07518, 5.258, 0.05606),
    ]

    for cg_tol in (1e-4, 1e-5):
        print(f"\n---- cg_tol={cg_tol:.0e}, trace_samples=10, jacobi=True ----")
        for name, ell, var, sig2 in states:
            bundle = make_bundle(
                x,
                y,
                lengthscale=ell,
                variance=var,
                sigmasq=sig2,
                eps=1e-4,
                nufft_eps=1e-5,
                use_preconditioner=True,
            )
            old_row = run_old_lengthscale(bundle, trace_samples=10, cg_tol=cg_tol, seed=123)
            new_row = run_new_lengthscale(bundle, trace_samples=10, cg_tol=cg_tol, seed=123)

            print(f"\nState {name}: ell={ell}, sigma_f^2={var}, sigma_n^2={sig2}, mtot={bundle.mtot}, M={bundle.M}")
            print(
                f"  old: iters median/max={old_row['iters_median']:.0f}/{old_row['iters_max']:.0f}  "
                f"rhs={old_row['rhs_build_sec']:.2f}s  solve={old_row['solve_sec']:.2f}s  "
                f"post={old_row['post_sec']:.2f}s  total={old_row['total_sec']:.2f}s"
            )
            print(
                f"  new: iters median/max={new_row['iters_median']:.0f}/{new_row['iters_max']:.0f}  "
                f"rhs={new_row['rhs_build_sec']:.2f}s  solve={new_row['solve_sec']:.2f}s  "
                f"post={new_row['post_sec']:.2f}s  total={new_row['total_sec']:.2f}s"
            )


def main() -> None:
    exact_validation()
    benchmark_prism()


if __name__ == "__main__":
    main()
