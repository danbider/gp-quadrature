#!/usr/bin/env python3
"""
Diagnose whether late PRISM variance gradients are a modeling/objective effect
or a numerical gradient artifact.

For frozen late PRISM states, we compare:
1. Exact formed-MxM gradient of the current approximate EFGP objective.
2. Finite differences of that exact approximate objective.
3. Current EFGPND.compute_gradients() at several cg_tol values.
4. A local exact objective scan versus sigma_f^2.

All results are for the current approximate feature-space objective, not the
full dense N x N GP.
"""

from __future__ import annotations

import math
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

from efgpnd import EFGPND, NUFFT, ToeplitzND, _cmplx, compute_convolution_vector_vectorized_dD  # noqa: E402
from kernels.squared_exponential import SquaredExponential  # noqa: E402
from load_prism import load_prism_dataset_torch  # noqa: E402
from utils.kernels import get_xis  # noqa: E402


@dataclass
class State:
    label: str
    lengthscale: float
    variance: float
    sigmasq: float


@dataclass
class ExactBundle:
    state: State
    eps: float
    nufft_eps: float
    N: int
    M: int
    mtot: int
    yty: float
    sigmasq: float
    variance: float
    lengthscale: float
    ws: torch.Tensor
    Fy: torch.Tensor
    C: torch.Tensor
    G: torch.Tensor
    A: torch.Tensor
    b: torch.Tensor
    beta: torch.Tensor
    noise_trace_exact: float
    y_alpha_exact: float
    alpha_norm_exact: float
    raw_variance_grad_exact: float
    nll_exact: float
    var_scan: list[tuple[float, float]]
    mtot_scan: list[tuple[float, int]]


def load_prism_standardized() -> tuple[torch.Tensor, torch.Tensor]:
    x, y = load_prism_dataset_torch("prism_tmean_us_30s_2020_avg_30y")
    x = x.to(dtype=torch.float64)
    y = y.to(dtype=torch.float64)
    x = (x - x.min(dim=0).values) / (x.max(dim=0).values - x.min(dim=0).values)
    y = (y - y.mean()) / y.std()
    return x, y


def build_explicit_c(toeplitz, M: int, cdtype: torch.dtype, batch_size: int = 64) -> torch.Tensor:
    eye = torch.eye(M, dtype=cdtype)
    cols = []
    for start in range(0, M, batch_size):
        stop = min(start + batch_size, M)
        cols.append(toeplitz(eye[start:stop]).T)
    return torch.cat(cols, dim=1)


def make_kernel(lengthscale: float, variance: float, d: int) -> SquaredExponential:
    kernel = SquaredExponential(dimension=d)
    kernel.set_hyper("lengthscale", lengthscale)
    kernel.set_hyper("variance", variance)
    return kernel


def exact_objective_terms_from_scaling(
    *,
    yty: float,
    G_base: torch.Tensor,
    b_base: torch.Tensor,
    sigmasq: float,
    scale: float,
) -> tuple[float, float, float]:
    cdtype = G_base.dtype
    M = G_base.shape[0]
    A = scale * G_base + sigmasq * torch.eye(M, dtype=cdtype)
    b = math.sqrt(scale) * b_base
    beta = torch.linalg.solve(A, b)
    y_alpha = (yty - torch.vdot(b, beta).real.item()) / sigmasq
    sign, logabsdet = torch.linalg.slogdet(A)
    if sign.real <= 0:
        raise RuntimeError("nonpositive determinant sign in exact objective")
    nll = 0.5 * ((A.shape[0]) * 0.0 + logabsdet.real.item())  # partial, caller adds constants
    return nll, y_alpha, float(torch.vdot(b, beta).real.item())


def build_exact_bundle(x: torch.Tensor, y: torch.Tensor, state: State, *, eps: float, nufft_eps: float) -> ExactBundle:
    dtype = x.dtype
    cdtype = _cmplx(dtype)
    d = x.shape[1]
    N = x.shape[0]
    kernel = make_kernel(state.lengthscale, state.variance, d)

    x0 = x.min(dim=0).values
    x1 = x.max(dim=0).values
    L = float((x1 - x0).max().item())
    xis_1d, h, mtot = get_xis(kernel_obj=kernel, eps=eps, L=L, use_integral=True, l2scaled=False)
    grids = torch.meshgrid(*(xis_1d for _ in range(d)), indexing="ij")
    xis = torch.stack(grids, dim=-1).view(-1, d)
    ws = torch.sqrt(kernel.spectral_density(xis).to(dtype=cdtype) * h**d)

    out_shape = (mtot,) * d
    nufft = NUFFT(x, torch.zeros(d, dtype=dtype), h, nufft_eps, cdtype=cdtype)
    fadj = lambda v: nufft.type1(v, out_shape=out_shape).reshape(v.shape[:-1] + (-1,))

    m_conv = (mtot - 1) // 2
    v_kernel = compute_convolution_vector_vectorized_dD(m_conv, x, h).to(dtype=cdtype)
    toeplitz = ToeplitzND(v_kernel, force_pow2=True)
    C = build_explicit_c(toeplitz, ws.numel(), cdtype)

    D = torch.diag(ws)
    G = D @ C @ D
    A = G + state.sigmasq * torch.eye(ws.numel(), dtype=cdtype)
    Fy = fadj(y.to(cdtype)).reshape(-1)
    b = ws * Fy
    beta = torch.linalg.solve(A, b)

    yty = float(torch.dot(y, y).item())
    b_beta = float(torch.vdot(b, beta).real.item())
    y_alpha_exact = (yty - b_beta) / state.sigmasq
    Gbeta = G @ beta
    alpha_norm_exact = (
        yty - 2.0 * b_beta + float(torch.vdot(beta, Gbeta).real.item())
    ) / (state.sigmasq**2)
    noise_trace_exact = (
        N / state.sigmasq
        - torch.trace(torch.linalg.solve(A, G)).real.item() / state.sigmasq
    )
    raw_variance_grad_exact = 0.5 * (
        N
        - state.sigmasq * noise_trace_exact
        - y_alpha_exact
        + state.sigmasq * alpha_norm_exact
    )

    sign, logabsdet = torch.linalg.slogdet(A)
    if sign.real <= 0:
        raise RuntimeError("nonpositive determinant sign in exact objective")
    nll_exact = 0.5 * (
        (N - ws.numel()) * math.log(state.sigmasq)
        + logabsdet.real.item()
        + y_alpha_exact
        + N * math.log(2 * math.pi)
    )

    var_factors = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    G_base = G / state.variance
    b_base = b / math.sqrt(state.variance)
    var_scan = []
    mtot_scan = []
    for factor in var_factors:
        var_val = state.variance * factor
        scale = factor
        A_var = scale * G_base + state.sigmasq * torch.eye(ws.numel(), dtype=cdtype)
        b_var = math.sqrt(scale) * b_base
        beta_var = torch.linalg.solve(A_var, b_var)
        y_alpha_var = (yty - torch.vdot(b_var, beta_var).real.item()) / state.sigmasq
        sign_var, logabsdet_var = torch.linalg.slogdet(A_var)
        if sign_var.real <= 0:
            raise RuntimeError("nonpositive determinant sign in variance scan")
        nll_var = 0.5 * (
            (N - ws.numel()) * math.log(state.sigmasq)
            + logabsdet_var.real.item()
            + y_alpha_var
            + N * math.log(2 * math.pi)
        )
        var_scan.append((var_val, nll_var))

        kernel_var = make_kernel(state.lengthscale, var_val, d)
        _, _, mtot_var = get_xis(kernel_obj=kernel_var, eps=eps, L=L, use_integral=True, l2scaled=False)
        mtot_scan.append((var_val, mtot_var))

    return ExactBundle(
        state=state,
        eps=eps,
        nufft_eps=nufft_eps,
        N=N,
        M=ws.numel(),
        mtot=mtot,
        yty=yty,
        sigmasq=state.sigmasq,
        variance=state.variance,
        lengthscale=state.lengthscale,
        ws=ws,
        Fy=Fy,
        C=C,
        G=G,
        A=A,
        b=b,
        beta=beta,
        noise_trace_exact=noise_trace_exact,
        y_alpha_exact=y_alpha_exact,
        alpha_norm_exact=alpha_norm_exact,
        raw_variance_grad_exact=raw_variance_grad_exact,
        nll_exact=nll_exact,
        var_scan=var_scan,
        mtot_scan=mtot_scan,
    )


def finite_diff_raw_variance(bundle: ExactBundle, delta: float = 1e-4) -> float:
    cdtype = bundle.G.dtype
    G_base = bundle.G / bundle.variance
    b_base = bundle.b / math.sqrt(bundle.variance)
    raw0 = math.log(bundle.variance)

    def objective_at_raw(raw_val: float) -> float:
        factor = math.exp(raw_val - raw0)
        A = factor * G_base + bundle.sigmasq * torch.eye(bundle.M, dtype=cdtype)
        b = math.sqrt(factor) * b_base
        beta = torch.linalg.solve(A, b)
        y_alpha = (bundle.yty - torch.vdot(b, beta).real.item()) / bundle.sigmasq
        sign, logabsdet = torch.linalg.slogdet(A)
        if sign.real <= 0:
            raise RuntimeError("nonpositive determinant sign in finite diff")
        return 0.5 * (
            (bundle.N - bundle.M) * math.log(bundle.sigmasq)
            + logabsdet.real.item()
            + y_alpha
            + bundle.N * math.log(2 * math.pi)
        )

    return (objective_at_raw(raw0 + delta) - objective_at_raw(raw0 - delta)) / (2 * delta)


def exact_raw_variance_from_model(model: EFGPND, *, trace_samples: int, cg_tol: float) -> tuple[float, dict[str, float], float]:
    torch.manual_seed(0)
    t0 = time.perf_counter()
    grad = model.compute_gradients(
        trace_samples=trace_samples,
        cg_tol=cg_tol,
        nufft_eps=model.nufft_eps,
        apply_gradients=False,
    )
    elapsed = time.perf_counter() - t0
    return float(grad[1].item()), dict(model.last_gradient_stats), elapsed


def make_model(x: torch.Tensor, y: torch.Tensor, state: State, *, eps: float, nufft_eps: float) -> EFGPND:
    model = EFGPND(
        x,
        y,
        kernel="SE",
        sigmasq=state.sigmasq,
        eps=eps,
        nufft_eps=nufft_eps,
        opts={
            "mean_cg_preconditioner": True,
            "trace_cg_preconditioner": True,
            "mean_cg_warm_start": False,
        },
        estimate_params=False,
    )
    model.kernel.set_hyper("lengthscale", state.lengthscale)
    model.kernel.set_hyper("variance", state.variance)
    with torch.no_grad():
        model._gp_params.raw.data[-1] = torch.log(torch.tensor(state.sigmasq, dtype=model._gp_params.raw.dtype))
    model._update_param_cache()
    model._last_gradient_beta = None
    return model


def best_scan_direction(var_scan: list[tuple[float, float]], current_variance: float) -> str:
    current = next(v for v in var_scan if abs(v[0] - current_variance) < 1e-12)
    left = [v for v in var_scan if v[0] < current_variance]
    right = [v for v in var_scan if v[0] > current_variance]
    best = min(var_scan, key=lambda pair: pair[1])
    if best[0] > current_variance:
        return f"downhill to larger variance (best scanned {best[0]:.4g})"
    if best[0] < current_variance:
        return f"downhill to smaller variance (best scanned {best[0]:.4g})"
    return "current variance best among scanned points"


def main() -> None:
    x, y = load_prism_standardized()
    states = [
        State("iter40", lengthscale=0.09256, variance=3.878, sigmasq=0.05202),
        State("final", lengthscale=0.07518, variance=5.258, sigmasq=0.05606),
    ]
    eps_list = [1e-3, 3e-4, 1e-4]
    cg_tols = [1e-4, 1e-5]
    trace_samples = 1

    for state in states:
        print(f"\n=== {state.label} ===")
        print(
            f"ell={state.lengthscale:.5g}  sigma_f^2={state.variance:.5g}  sigma_n^2={state.sigmasq:.5g}"
        )
        for eps in eps_list:
            nufft_eps = eps * 0.1
            print(f"\n-- eps={eps:g}  nufft_eps={nufft_eps:g} --")
            exact = build_exact_bundle(x, y, state, eps=eps, nufft_eps=nufft_eps)
            fd = finite_diff_raw_variance(exact)
            print(
                f"formed-MxM exact: mtot={exact.mtot}  M={exact.M}  "
                f"raw grad_sigma_f^2={exact.raw_variance_grad_exact:.6e}  "
                f"finite-diff={fd:.6e}  abs diff={abs(fd - exact.raw_variance_grad_exact):.3e}"
            )
            print(
                f"exact pieces: trace_noise={exact.noise_trace_exact:.6e}  "
                f"y^T alpha={exact.y_alpha_exact:.6e}  alpha^2={exact.alpha_norm_exact:.6e}"
            )
            print(f"exact approximate NLL={exact.nll_exact:.6e}")
            print(f"variance scan: {best_scan_direction(exact.var_scan, state.variance)}")
            for var_val, nll_val in exact.var_scan:
                marker = "*" if abs(var_val - state.variance) < 1e-12 else " "
                print(f"  {marker} var={var_val:.6g}  nll={nll_val:.6e}")
            mtot_parts = ", ".join([f"{var_val:.4g}->{mtot}" for var_val, mtot in exact.mtot_scan])
            print(f"mtot under variance scan: {mtot_parts}")

            for cg_tol in cg_tols:
                model = make_model(x, y, state, eps=eps, nufft_eps=nufft_eps)
                grad_var, stats, elapsed = exact_raw_variance_from_model(
                    model,
                    trace_samples=trace_samples,
                    cg_tol=cg_tol,
                )
                abs_err = abs(grad_var - exact.raw_variance_grad_exact)
                rel_err = abs_err / max(abs(exact.raw_variance_grad_exact), 1e-12)
                print(
                    f"  compute_gradients cg_tol={cg_tol:g}: raw grad_sigma_f^2={grad_var:.6e}  "
                    f"abs err={abs_err:.3e}  rel err={rel_err:.3e}  "
                    f"time={elapsed:.2f}s  mean_cg={stats.get('mean_cg_iters')}  trace_cg={stats.get('trace_cg_iters')}"
                )


if __name__ == "__main__":
    main()
