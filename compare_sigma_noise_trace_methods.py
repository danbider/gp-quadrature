#!/usr/bin/env python3
"""
Compare the current sigma-noise trace solve against a feature-space alternative.

This script does not modify `efgpnd.py`. It uses the same NUFFT and Toeplitz
matvecs already present in the codebase.

It runs two experiments:

1. A small exact validation:
   - builds dense matrices from the existing matvecs
   - checks that the feature-space formulation matches dense linear algebra
   - checks that the current sigma-noise solve matches the same dense system

2. A real-data hard-regime benchmark:
   - standardized `usa_temp_data`
   - compares CG iteration counts for
       a) current sigma-noise block: solve A_mean beta = D F^* z
       b) feature-space trace: solve A_mean u = v
       c) optional scaled feature-space trace: solve A_var gamma = v
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

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
    create_Gv,
    create_A_mean,
    create_A_var,
)
from kernels.squared_exponential import SquaredExponential  # noqa: E402
from utils.kernels import get_xis  # noqa: E402


@dataclass
class Bundle:
    x: torch.Tensor
    y: torch.Tensor
    kernel: SquaredExponential
    sigmasq: float
    eps: float
    nufft_eps: float
    h: float
    mtot: int
    M: int
    N: int
    d: int
    cdtype: torch.dtype
    ws: torch.Tensor
    G_apply: object
    A_mean: object
    A_var: object
    fadj: object
    fwd: object


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
    lengthscale: float,
    variance: float,
    sigmasq: float,
    eps: float,
    nufft_eps: float,
) -> Bundle:
    dtype = x.dtype
    cdtype = _cmplx(dtype)
    d = x.shape[1]
    N = x.shape[0]

    kernel = make_kernel(lengthscale, variance)
    L = float((x.max(dim=0).values - x.min(dim=0).values).max().item())
    xis_1d, h, mtot = get_xis(kernel, eps=eps, L=L, use_integral=True, l2scaled=False)
    xis_1d = xis_1d.to(dtype=dtype)
    xis = torch.stack(torch.meshgrid(*(xis_1d for _ in range(d)), indexing="ij"), dim=-1).reshape(-1, d)
    ws = torch.sqrt(kernel.spectral_density(xis).to(dtype=cdtype) * h**d)

    OUT = (mtot,) * d
    xcen = torch.zeros(d, dtype=dtype)
    nufft = NUFFT(x, xcen, h, nufft_eps, cdtype=cdtype)
    fadj = lambda v: nufft.type1(v, out_shape=OUT).reshape(v.shape[:-1] + (-1,))
    fwd = lambda fk: nufft.type2(fk, out_shape=OUT)

    m_conv = (mtot - 1) // 2
    v_kernel = compute_convolution_vector_vectorized_dD(m_conv, x, h).to(dtype=cdtype)
    toeplitz = ToeplitzND(v_kernel, force_pow2=True)
    G_apply = create_Gv(ws, toeplitz, cdtype)
    A_mean = create_A_mean(ws, toeplitz, torch.tensor(sigmasq, dtype=dtype), cdtype)
    A_var = create_A_var(ws, toeplitz, torch.tensor(sigmasq, dtype=dtype), cdtype)

    return Bundle(
        x=x,
        y=y,
        kernel=kernel,
        sigmasq=sigmasq,
        eps=eps,
        nufft_eps=nufft_eps,
        h=h,
        mtot=mtot,
        M=ws.numel(),
        N=N,
        d=d,
        cdtype=cdtype,
        ws=ws,
        G_apply=G_apply,
        A_mean=A_mean,
        A_var=A_var,
        fadj=fadj,
        fwd=fwd,
    )


def make_rademacher(shape: Tuple[int, ...], *, dtype: torch.dtype, cdtype: torch.dtype, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    z = torch.empty(shape, dtype=dtype)
    z.bernoulli_(0.5)
    z.mul_(2).sub_(1)
    return z.to(cdtype)


def build_explicit_phi(bundle: Bundle, batch_size: int = 128) -> torch.Tensor:
    phi = torch.empty((bundle.N, bundle.M), dtype=bundle.cdtype)
    for start in range(0, bundle.M, batch_size):
        stop = min(start + batch_size, bundle.M)
        rows = stop - start
        coeffs = torch.zeros((rows, bundle.M), dtype=bundle.cdtype)
        local = torch.arange(rows)
        coeffs[local, start + local] = bundle.ws[start:stop]
        phi[:, start:stop] = bundle.fwd(coeffs).T
    return phi


def build_explicit_matrix_from_apply(A_apply, M: int, cdtype: torch.dtype, batch_size: int = 128) -> torch.Tensor:
    A = torch.empty((M, M), dtype=cdtype)
    for start in range(0, M, batch_size):
        stop = min(start + batch_size, M)
        rows = stop - start
        basis = torch.zeros((rows, M), dtype=cdtype)
        local = torch.arange(rows)
        basis[local, start + local] = 1
        cols = A_apply(basis).T
        A[:, start:stop] = cols
    return A


def current_noise_rhs(bundle: Bundle, probes_z: torch.Tensor) -> torch.Tensor:
    fadj_z = bundle.fadj(probes_z).reshape(probes_z.shape[0], -1)
    return bundle.ws * fadj_z


def current_noise_trace_estimate(
    bundle: Bundle,
    probes_z: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    fwd_beta = bundle.fwd(bundle.ws * beta)
    alpha = (probes_z - fwd_beta) / bundle.sigmasq
    return (probes_z.conj() * alpha).sum(dim=1).real


def feature_trace_estimate_from_amean(
    bundle: Bundle,
    probes_v: torch.Tensor,
    soln: torch.Tensor,
) -> torch.Tensor:
    offset = (bundle.N - bundle.M) / bundle.sigmasq
    return offset + (probes_v.conj() * soln).sum(dim=1).real


def projected_feature_rhs(bundle: Bundle, probes_v: torch.Tensor) -> torch.Tensor:
    return bundle.G_apply(probes_v)


def projected_feature_trace_estimate(
    bundle: Bundle,
    probes_v: torch.Tensor,
    soln: torch.Tensor,
) -> torch.Tensor:
    return bundle.N / bundle.sigmasq - (probes_v.conj() * soln).sum(dim=1).real / bundle.sigmasq


def feature_trace_estimate_from_avar(
    bundle: Bundle,
    probes_v: torch.Tensor,
    soln: torch.Tensor,
) -> torch.Tensor:
    offset = (bundle.N - bundle.M) / bundle.sigmasq
    return offset + (probes_v.conj() * soln).sum(dim=1).real / bundle.sigmasq


def solve_with_stats(A_apply, rhs: torch.Tensor, tol: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    solver = RecordingConjugateGradients(
        A_apply,
        rhs,
        torch.zeros_like(rhs),
        tol=tol,
        early_stopping=True,
    )
    soln, stats = solver.solve()
    row = {
        "iters_completed": float(stats.iters_completed),
        "iters_min": float(stats.iters_min),
        "iters_median": float(stats.iters_median),
        "iters_mean": float(stats.iters_mean),
        "iters_max": float(stats.iters_max),
        "apply_calls": float(stats.apply_calls),
        "solve_time_sec": stats.solve_time_sec,
        "rel_res_mean": stats.rel_res_mean,
        "rel_res_max": stats.rel_res_max,
        "converged_fraction": stats.converged_fraction,
    }
    return soln, row


def validation_run(args) -> None:
    print("== Small exact validation ==")
    x_full, y_full = load_usa_temp_standardized()
    x = x_full[: args.validation_n].clone()
    y = y_full[: args.validation_n].clone()

    bundle = build_bundle(
        x,
        y,
        lengthscale=args.validation_lengthscale,
        variance=args.validation_variance,
        sigmasq=args.validation_sigmasq,
        eps=args.validation_eps,
        nufft_eps=args.nufft_eps,
    )
    print(
        f"validation problem: n={bundle.N}, mtot={bundle.mtot}, M={bundle.M}, "
        f"ell={args.validation_lengthscale}, sigma_f^2={args.validation_variance}, sigma_n^2={bundle.sigmasq}"
    )

    phi = build_explicit_phi(bundle, batch_size=args.dense_batch)
    A_from_phi = phi.conj().T @ phi + bundle.sigmasq * torch.eye(bundle.M, dtype=bundle.cdtype)
    A_from_apply = build_explicit_matrix_from_apply(bundle.A_mean, bundle.M, bundle.cdtype, batch_size=args.dense_batch)

    dense_apply_close = torch.allclose(A_from_phi, A_from_apply, rtol=1e-6, atol=1e-7)
    max_apply_err = float((A_from_phi - A_from_apply).abs().max().item())
    print(f"A_from_phi vs A_from_apply allclose: {dense_apply_close} (max abs err {max_apply_err:.3e})")

    K_dense = phi @ phi.conj().T + bundle.sigmasq * torch.eye(bundle.N, dtype=bundle.cdtype)
    trace_k_dense = torch.trace(torch.linalg.inv(K_dense)).real
    trace_identity = ((bundle.N - bundle.M) / bundle.sigmasq + torch.trace(torch.linalg.inv(A_from_phi)).real)
    trace_identity_close = torch.allclose(trace_k_dense, trace_identity, rtol=1e-8, atol=1e-8)
    print(
        "trace(K^-1) identity allclose: "
        f"{trace_identity_close} "
        f"(dense {trace_k_dense.item():.12e}, identity {trace_identity.item():.12e})"
    )

    probes_v = make_rademacher(
        (args.validation_probes, bundle.M),
        dtype=bundle.x.dtype,
        cdtype=bundle.cdtype,
        seed=args.seed,
    )
    cg_feature, _ = solve_with_stats(bundle.A_mean, probes_v, tol=args.validation_cg_tol)
    dense_feature = torch.linalg.solve(A_from_phi, probes_v.T).T
    feature_close = torch.allclose(cg_feature, dense_feature, rtol=1e-3, atol=1e-5)
    feature_rel_err = float(
        (torch.linalg.norm((cg_feature - dense_feature).reshape(-1)) / torch.linalg.norm(dense_feature.reshape(-1))).item()
    )
    print(f"feature-space solve allclose: {feature_close} (relative soln error {feature_rel_err:.3e})")

    probes_z = make_rademacher(
        (args.validation_probes, bundle.N),
        dtype=bundle.x.dtype,
        cdtype=bundle.cdtype,
        seed=args.seed + 1,
    )
    rhs_noise = current_noise_rhs(bundle, probes_z)
    cg_current, _ = solve_with_stats(bundle.A_mean, rhs_noise, tol=args.validation_cg_tol)
    dense_current = torch.linalg.solve(A_from_phi, rhs_noise.T).T
    current_close = torch.allclose(cg_current, dense_current, rtol=1e-3, atol=1e-5)
    current_rel_err = float(
        (torch.linalg.norm((cg_current - dense_current).reshape(-1)) / torch.linalg.norm(dense_current.reshape(-1))).item()
    )
    print(f"current noise-block solve allclose: {current_close} (relative soln error {current_rel_err:.3e})")

    rhs_projected = projected_feature_rhs(bundle, probes_v)
    cg_projected, _ = solve_with_stats(bundle.A_mean, rhs_projected, tol=args.validation_cg_tol)
    dense_projected = torch.linalg.solve(A_from_phi, rhs_projected.T).T
    projected_close = torch.allclose(cg_projected, dense_projected, rtol=1e-3, atol=1e-5)
    projected_rel_err = float(
        (torch.linalg.norm((cg_projected - dense_projected).reshape(-1)) / torch.linalg.norm(dense_projected.reshape(-1))).item()
    )
    print(f"projected feature solve allclose: {projected_close} (relative soln error {projected_rel_err:.3e})")

    dense_trace_exact = float(trace_k_dense.item())
    feature_trace_samples = feature_trace_estimate_from_amean(bundle, probes_v, cg_feature)
    current_trace_samples = current_noise_trace_estimate(bundle, probes_z, cg_current)
    projected_trace_samples = projected_feature_trace_estimate(bundle, probes_v, cg_projected)
    print(
        "trace estimates vs exact dense trace: "
        f"feature mean={feature_trace_samples.mean().item():.6e}, "
        f"projected mean={projected_trace_samples.mean().item():.6e}, "
        f"current mean={current_trace_samples.mean().item():.6e}, "
        f"exact={dense_trace_exact:.6e}"
    )


def benchmark_run(args) -> None:
    print("\n== Real-data hard-regime benchmark ==")
    x, y = load_usa_temp_standardized()
    bundle = build_bundle(
        x,
        y,
        lengthscale=args.lengthscale,
        variance=args.variance,
        sigmasq=args.sigmasq,
        eps=args.eps,
        nufft_eps=args.nufft_eps,
    )
    print(
        f"benchmark problem: n={bundle.N}, mtot={bundle.mtot}, M={bundle.M}, "
        f"ell={args.lengthscale}, sigma_f^2={args.variance}, sigma_n^2={args.sigmasq}, "
        f"J={args.trace_probes}, cg_tol={args.cg_tol}"
    )

    probes_z = make_rademacher(
        (args.trace_probes, bundle.N),
        dtype=bundle.x.dtype,
        cdtype=bundle.cdtype,
        seed=args.seed,
    )
    rhs_current = current_noise_rhs(bundle, probes_z)
    _, current_stats = solve_with_stats(bundle.A_mean, rhs_current, tol=args.cg_tol)

    probes_v = make_rademacher(
        (args.trace_probes, bundle.M),
        dtype=bundle.x.dtype,
        cdtype=bundle.cdtype,
        seed=args.seed + 17,
    )
    _, feature_mean_stats = solve_with_stats(bundle.A_mean, probes_v, tol=args.cg_tol)
    _, feature_var_stats = solve_with_stats(bundle.A_var, probes_v, tol=args.cg_tol)
    rhs_projected = projected_feature_rhs(bundle, probes_v)
    _, projected_stats = solve_with_stats(bundle.A_mean, rhs_projected, tol=args.cg_tol)

    print("\nCG iteration summary")
    print(
        f"{'method':<28} {'iters_med':>10} {'iters_max':>10} {'apply_calls':>12} "
        f"{'time_sec':>10} {'rel_res_max':>12}"
    )
    print(
        f"{'current: A_mean^{-1}(D F^* z)':<28} "
        f"{current_stats['iters_median']:>10.1f} {current_stats['iters_max']:>10.1f} "
        f"{current_stats['apply_calls']:>12.0f} {current_stats['solve_time_sec']:>10.3f} "
        f"{current_stats['rel_res_max']:>12.3e}"
    )
    print(
        f"{'feature: A_mean^{-1} v':<28} "
        f"{feature_mean_stats['iters_median']:>10.1f} {feature_mean_stats['iters_max']:>10.1f} "
        f"{feature_mean_stats['apply_calls']:>12.0f} {feature_mean_stats['solve_time_sec']:>10.3f} "
        f"{feature_mean_stats['rel_res_max']:>12.3e}"
    )
    print(
        f"{'feature: A_var^{-1} v':<28} "
        f"{feature_var_stats['iters_median']:>10.1f} {feature_var_stats['iters_max']:>10.1f} "
        f"{feature_var_stats['apply_calls']:>12.0f} {feature_var_stats['solve_time_sec']:>10.3f} "
        f"{feature_var_stats['rel_res_max']:>12.3e}"
    )
    print(
        f"{'projected: A_mean^{-1} G v':<28} "
        f"{projected_stats['iters_median']:>10.1f} {projected_stats['iters_max']:>10.1f} "
        f"{projected_stats['apply_calls']:>12.0f} {projected_stats['solve_time_sec']:>10.3f} "
        f"{projected_stats['rel_res_max']:>12.3e}"
    )

    med_speedup = current_stats["iters_median"] / max(projected_stats["iters_median"], 1.0)
    max_speedup = current_stats["iters_max"] / max(projected_stats["iters_max"], 1.0)
    print(
        "\nprojected feature-space solve speedup vs current noise block: "
        f"median iters x{med_speedup:.2f}, max iters x{max_speedup:.2f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nufft-eps", type=float, default=6e-8)

    parser.add_argument("--validation-n", type=int, default=96)
    parser.add_argument("--validation-lengthscale", type=float, default=0.18)
    parser.add_argument("--validation-variance", type=float, default=10.0)
    parser.add_argument("--validation-sigmasq", type=float, default=1e-3)
    parser.add_argument("--validation-eps", type=float, default=1e-4)
    parser.add_argument("--validation-probes", type=int, default=4)
    parser.add_argument("--validation-cg-tol", type=float, default=1e-10)
    parser.add_argument("--dense-batch", type=int, default=96)

    parser.add_argument("--lengthscale", type=float, default=0.03)
    parser.add_argument("--variance", type=float, default=10.0)
    parser.add_argument("--sigmasq", type=float, default=1e-4)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--trace-probes", type=int, default=8)
    parser.add_argument("--cg-tol", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validation_run(args)
    benchmark_run(args)


if __name__ == "__main__":
    main()
