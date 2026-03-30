#!/usr/bin/env python3
"""
Evaluate alternative lengthscale gradient estimators.

We compare:
1. The current lengthscale gradient estimator used by EFGPND.
2. A new feature-space derivative of the whole approximate objective:

       dL/dtheta = 0.5 [ tr(A^{-1} A_theta)
                        - sigma^{-2}(2 Re(b_theta^* beta) - beta^* A_theta beta) ]

   where A = G + sigma^2 I, G = D(F^*F)D, b = D F^* y, and
   A_theta = E_theta G + G E_theta with E_theta = diag(0.5 * dlog(w^2)/dtheta).
3. An exact-scale-plus-PSD-residual split for the lengthscale trace term:

       d(w^2)/d ell = c_ell w^2 - q_ell,   q_ell >= 0

   with exact scale term c_ell (n - sigma^2 tr(K^{-1})) and residual
   feature-space trace built from H_res = D C Q C D.

Ground truth is always the exact approximate EFGP objective:
- small cases: checked against explicit dense K as well;
- full PRISM case: exact formed MxM feature-space system.
"""

from __future__ import annotations

import math
import os
import statistics as stats
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
from kernels.matern import Matern  # noqa: E402
from kernels.squared_exponential import SquaredExponential  # noqa: E402
from load_prism import load_prism_dataset_torch  # noqa: E402
from utils.kernels import get_xis  # noqa: E402


@dataclass
class Bundle:
    label: str
    kernel_name: str
    x: torch.Tensor
    y: torch.Tensor
    kernel: object
    dtype: torch.dtype
    cdtype: torch.dtype
    N: int
    d: int
    M: int
    mtot: int
    sigmasq: float
    variance: float
    lengthscale: float
    ws: torch.Tensor
    s: torch.Tensor
    d_l: torch.Tensor
    e_l: torch.Tensor
    C: torch.Tensor
    A: torch.Tensor
    G: torch.Tensor
    b: torch.Tensor
    beta: torch.Tensor
    A_theta: torch.Tensor
    trace_kinv_exact: float | None
    split_c: float | None
    q_l: torch.Tensor | None
    tr_cq: float | None
    split_trace_constant: float | None
    trace_exact: float
    quad_exact: float
    grad_exact: float
    dense_trace_exact: float | None
    dense_quad_exact: float | None
    dense_grad_exact: float | None
    fadj: object
    fwd: object
    toeplitz: object
    A_apply: object
    M_inv: object
    explicit_B_old: torch.Tensor | None


def load_usa_temp_subset(n: int) -> tuple[torch.Tensor, torch.Tensor]:
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


def build_explicit_c(toeplitz, M: int, cdtype: torch.dtype, batch_size: int = 64) -> torch.Tensor:
    eye = torch.eye(M, dtype=cdtype)
    cols = []
    for start in range(0, M, batch_size):
        stop = min(start + batch_size, M)
        cols.append(toeplitz(eye[start:stop]).T)
    return torch.cat(cols, dim=1)


def build_explicit_f(fwd, M: int, cdtype: torch.dtype, batch_size: int = 64) -> torch.Tensor:
    eye = torch.eye(M, dtype=cdtype)
    cols = []
    for start in range(0, M, batch_size):
        stop = min(start + batch_size, M)
        cols.append(fwd(eye[start:stop]).T)
    return torch.cat(cols, dim=1)


def make_kernel(kernel_name: str, d: int, *, lengthscale: float, variance: float):
    if kernel_name == "SE":
        kernel = SquaredExponential(dimension=d)
    elif kernel_name == "Matern":
        kernel = Matern(dimension=d)
    else:
        raise ValueError(f"unsupported kernel {kernel_name}")
    kernel.set_hyper("lengthscale", lengthscale)
    kernel.set_hyper("variance", variance)
    return kernel


def make_bundle(
    label: str,
    kernel_name: str,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    lengthscale: float,
    variance: float,
    sigmasq: float,
    eps: float,
    nufft_eps: float,
    build_dense_check: bool,
) -> Bundle:
    dtype = x.dtype
    cdtype = _cmplx(dtype)
    d = x.shape[1]
    N = x.shape[0]
    kernel = make_kernel(kernel_name, d, lengthscale=lengthscale, variance=variance)

    x0 = x.min(dim=0).values
    x1 = x.max(dim=0).values
    L = float((x1 - x0).max().item())
    xis_1d, h, mtot = get_xis(kernel_obj=kernel, eps=eps, L=L, use_integral=True, l2scaled=False)
    grids = torch.meshgrid(*(xis_1d for _ in range(d)), indexing="ij")
    xis = torch.stack(grids, dim=-1).view(-1, d)

    ws = torch.sqrt(kernel.spectral_density(xis).to(dtype=cdtype) * h**d)
    s = ws.abs().pow(2).real
    d_l = (h**d * kernel.spectral_grad(xis)[:, 0]).real
    e_l = 0.5 * d_l / s.clamp_min(1e-300)

    out_shape = (mtot,) * d
    nufft = NUFFT(x, torch.zeros(d, dtype=dtype), h, nufft_eps, cdtype=cdtype)
    fadj = lambda v: nufft.type1(v, out_shape=out_shape).reshape(v.shape[:-1] + (-1,))
    fwd = lambda fk: nufft.type2(fk, out_shape=out_shape)

    m_conv = (mtot - 1) // 2
    v_kernel = compute_convolution_vector_vectorized_dD(m_conv, x, h).to(dtype=cdtype)
    toeplitz = ToeplitzND(v_kernel, force_pow2=True)
    center = tuple(((torch.tensor(v_kernel.shape) - 1) // 2).tolist())
    diag_scale = v_kernel[center].real

    C = build_explicit_c(toeplitz, ws.numel(), cdtype)
    D = torch.diag(ws)
    G = D @ C @ D
    A = G + sigmasq * torch.eye(ws.numel(), dtype=cdtype)

    Fy = fadj(y.to(cdtype)).reshape(-1)
    b = ws * Fy
    beta = torch.linalg.solve(A, b)
    E = torch.diag(e_l.to(cdtype))
    A_theta = E @ G + G @ E
    b_theta = e_l.to(cdtype) * b

    trace_exact = float(torch.trace(torch.linalg.solve(A, A_theta)).real.item())
    quad_exact = float((2 * torch.vdot(b_theta, beta).real - torch.vdot(beta, A_theta @ beta).real).item() / sigmasq)
    grad_exact = 0.5 * (trace_exact - quad_exact)

    trace_kinv_exact = None
    split_c = None
    q_l = None
    tr_cq = None
    split_trace_constant = None
    dense_trace_exact = None
    dense_quad_exact = None
    dense_grad_exact = None
    explicit_B_old = None
    if build_dense_check:
        F = build_explicit_f(fwd, ws.numel(), cdtype)
        K = (F @ torch.diag(s.to(cdtype)) @ F.conj().T).real + sigmasq * torch.eye(N, dtype=dtype)
        dK = (F @ torch.diag(d_l.to(cdtype)) @ F.conj().T).real
        Kinv_dK = torch.linalg.solve(K, dK)
        alpha = torch.linalg.solve(K, y)
        dense_trace_exact = float(torch.trace(Kinv_dK).item())
        dense_quad_exact = float(torch.dot(alpha, dK @ alpha).item())
        dense_grad_exact = 0.5 * (dense_trace_exact - dense_quad_exact)
        trace_kinv_exact = float(torch.trace(torch.linalg.inv(K)).item())
        explicit_B_old = Kinv_dK.to(cdtype)
    elif ws.numel() <= 1024:
        A_inv = torch.linalg.inv(A)
        trace_kinv_exact = float((N / sigmasq - torch.trace(A_inv @ G).real / sigmasq).item())

    if kernel_name in {"SE", "Matern"}:
        split_c = d / lengthscale
        q_l = (split_c * s - d_l).real
        tr_cq = float((torch.diagonal(C).real * q_l).sum().item())
        if trace_kinv_exact is not None:
            split_trace_constant = split_c * (N - sigmasq * trace_kinv_exact) - tr_cq / sigmasq

    A_apply = create_A_mean(ws, toeplitz, torch.tensor(sigmasq, dtype=dtype), cdtype)
    M_inv = create_jacobi_precond(ws, torch.tensor(sigmasq, dtype=dtype), diag_scale=diag_scale)

    return Bundle(
        label=label,
        kernel_name=kernel_name,
        x=x,
        y=y,
        kernel=kernel,
        dtype=dtype,
        cdtype=cdtype,
        N=N,
        d=d,
        M=ws.numel(),
        mtot=mtot,
        sigmasq=sigmasq,
        variance=variance,
        lengthscale=lengthscale,
        ws=ws,
        s=s,
        d_l=d_l,
        e_l=e_l,
        C=C,
        A=A,
        G=G,
        b=b,
        beta=beta,
        A_theta=A_theta,
        trace_kinv_exact=trace_kinv_exact,
        split_c=split_c,
        q_l=q_l,
        tr_cq=tr_cq,
        split_trace_constant=split_trace_constant,
        trace_exact=trace_exact,
        quad_exact=quad_exact,
        grad_exact=grad_exact,
        dense_trace_exact=dense_trace_exact,
        dense_quad_exact=dense_quad_exact,
        dense_grad_exact=dense_grad_exact,
        fadj=fadj,
        fwd=fwd,
        toeplitz=toeplitz,
        A_apply=A_apply,
        M_inv=M_inv,
        explicit_B_old=explicit_B_old,
    )


def make_rademacher(shape: tuple[int, ...], dtype: torch.dtype, cdtype: torch.dtype, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    z = torch.empty(shape, dtype=dtype)
    z.bernoulli_(0.5)
    z.mul_(2).sub_(1)
    return z.to(cdtype)


def G_apply(bundle: Bundle, V: torch.Tensor) -> torch.Tensor:
    return (bundle.ws.unsqueeze(0) * bundle.toeplitz(bundle.ws.unsqueeze(0) * V).reshape(V.shape[0], -1)).to(bundle.cdtype)


def Hres_apply(bundle: Bundle, V: torch.Tensor) -> torch.Tensor:
    if bundle.q_l is None:
        raise ValueError("Residual split not available for this bundle.")
    q = bundle.q_l.to(bundle.cdtype)
    DV = bundle.ws.unsqueeze(0) * V
    CDV = bundle.toeplitz(DV).reshape(V.shape[0], -1).to(bundle.cdtype)
    QCDV = q.unsqueeze(0) * CDV
    CQCDV = bundle.toeplitz(QCDV).reshape(V.shape[0], -1).to(bundle.cdtype)
    return bundle.ws.unsqueeze(0) * CQCDV


def sample_old_grad(bundle: Bundle, num_samples: int, seed: int, batch_size: int = 1) -> torch.Tensor:
    grads = []
    d_l_c = bundle.d_l.to(bundle.cdtype)
    if bundle.explicit_B_old is not None:
        Z = make_rademacher((num_samples, bundle.N), bundle.dtype, bundle.cdtype, seed).real.to(bundle.dtype)
        term1 = ((Z @ bundle.explicit_B_old.real.to(bundle.dtype)) * Z).sum(dim=1)
        return 0.5 * (term1 - bundle.quad_exact)

    for offset in range(0, num_samples, batch_size):
        block = min(batch_size, num_samples - offset)
        Z = make_rademacher((block, bundle.N), bundle.dtype, bundle.cdtype, seed + offset)
        fadjZ = bundle.fadj(Z).reshape(block, -1)
        DiFZ = d_l_c.unsqueeze(0) * fadjZ
        rhs_data = bundle.fwd(DiFZ).reshape(block, -1)
        B_old = (bundle.ws.unsqueeze(0) * bundle.toeplitz(DiFZ).reshape(block, -1)).to(bundle.cdtype)
        Beta_old = torch.linalg.solve(bundle.A, B_old.T).T
        mean_part = bundle.fwd(bundle.ws.unsqueeze(0) * Beta_old).reshape(block, -1)
        Alpha_old = (rhs_data - mean_part) / bundle.sigmasq
        term1 = (Z.conj() * Alpha_old).sum(dim=1).real
        grads.append(0.5 * (term1 - bundle.quad_exact))
    return torch.cat(grads, dim=0)


def sample_new_grad(bundle: Bundle, num_samples: int, seed: int) -> torch.Tensor:
    V = make_rademacher((num_samples, bundle.M), bundle.dtype, bundle.cdtype, seed)
    Beta = torch.linalg.solve(bundle.A, bundle.A_theta @ V.T).T
    trace_vals = (V.conj() * Beta).sum(dim=1).real
    return 0.5 * (trace_vals - bundle.quad_exact)


def sample_residual_split_grad(bundle: Bundle, num_samples: int, seed: int) -> torch.Tensor:
    if bundle.split_trace_constant is None:
        raise ValueError("Residual split requires exact tr(K^{-1}) for this bundle.")
    V = make_rademacher((num_samples, bundle.M), bundle.dtype, bundle.cdtype, seed)
    HresV = Hres_apply(bundle, V)
    Beta = torch.linalg.solve(bundle.A, HresV.T).T
    trace_vals = bundle.split_trace_constant + (V.conj() * Beta).sum(dim=1).real / bundle.sigmasq
    return 0.5 * (trace_vals - bundle.quad_exact)


def bench_old_lengthscale(bundle: Bundle, seed: int, cg_tol: float) -> tuple[float, float, float, float]:
    t0 = time.perf_counter()
    Z = make_rademacher((1, bundle.N), bundle.dtype, bundle.cdtype, seed)
    fadjZ = bundle.fadj(Z).reshape(1, -1)
    DiFZ = bundle.d_l.to(bundle.cdtype).unsqueeze(0) * fadjZ
    rhs_data = bundle.fwd(DiFZ).reshape(1, -1)
    B_old = (bundle.ws.unsqueeze(0) * bundle.toeplitz(DiFZ).reshape(1, -1)).to(bundle.cdtype)
    rhs_sec = time.perf_counter() - t0

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
    mean_part = bundle.fwd(bundle.ws.unsqueeze(0) * Beta_old).reshape(1, -1)
    Alpha_old = (rhs_data - mean_part) / bundle.sigmasq
    grad = float((0.5 * ((Z.conj() * Alpha_old).sum(dim=1).real - bundle.quad_exact)).item())
    post_sec = time.perf_counter() - t1
    return rhs_sec + stats.solve_time_sec + post_sec, stats.iters_median, stats.rel_res_median, grad


def bench_new_lengthscale(bundle: Bundle, seed: int, cg_tol: float) -> tuple[float, float, float, float]:
    e = bundle.e_l.to(bundle.cdtype)
    t0 = time.perf_counter()
    V = make_rademacher((1, bundle.M), bundle.dtype, bundle.cdtype, seed)
    GV = G_apply(bundle, V)
    EV = e.unsqueeze(0) * V
    AthetaV = e.unsqueeze(0) * GV + G_apply(bundle, EV)

    Gbeta = bundle.b - bundle.sigmasq * bundle.beta
    Ge_beta = G_apply(bundle, (e * bundle.beta).unsqueeze(0)).reshape(-1)
    Atheta_beta = e * Gbeta + Ge_beta
    quad = (
        2 * torch.vdot(e * bundle.b, bundle.beta).real - torch.vdot(bundle.beta, Atheta_beta).real
    ) / bundle.sigmasq
    rhs_sec = time.perf_counter() - t0

    solver = RecordingConjugateGradients(
        bundle.A_apply,
        AthetaV,
        torch.zeros_like(AthetaV),
        tol=cg_tol,
        early_stopping=True,
        M_inv_apply=bundle.M_inv,
    )
    Beta_new, stats = solver.solve()

    t1 = time.perf_counter()
    grad = float((0.5 * ((V.conj() * Beta_new).sum(dim=1).real - quad)).item())
    post_sec = time.perf_counter() - t1
    return rhs_sec + stats.solve_time_sec + post_sec, stats.iters_median, stats.rel_res_median, grad


def bench_residual_split_lengthscale(bundle: Bundle, seed: int, cg_tol: float) -> tuple[float, float, float, float]:
    if bundle.split_trace_constant is None:
        raise ValueError("Residual split requires exact tr(K^{-1}) for this bundle.")
    t0 = time.perf_counter()
    V = make_rademacher((1, bundle.M), bundle.dtype, bundle.cdtype, seed)
    HresV = Hres_apply(bundle, V)
    rhs_sec = time.perf_counter() - t0

    solver = RecordingConjugateGradients(
        bundle.A_apply,
        HresV,
        torch.zeros_like(HresV),
        tol=cg_tol,
        early_stopping=True,
        M_inv_apply=bundle.M_inv,
    )
    Beta_res, stats = solver.solve()

    t1 = time.perf_counter()
    trace_val = bundle.split_trace_constant + float((V.conj() * Beta_res).sum(dim=1).real.item() / bundle.sigmasq)
    grad = 0.5 * (trace_val - bundle.quad_exact)
    post_sec = time.perf_counter() - t1
    return rhs_sec + stats.solve_time_sec + post_sec, stats.iters_median, stats.rel_res_median, grad


def summarize_samples(name: str, values: torch.Tensor, exact_grad: float) -> list[str]:
    vals = values.tolist()
    mean_grad = stats.mean(vals)
    std_grad = stats.pstdev(vals)
    bias_grad = mean_grad - exact_grad
    rel_bias_grad = abs(bias_grad) / max(abs(exact_grad), 1e-12)
    rel_std_grad = std_grad / max(abs(exact_grad), 1e-12)
    rmse_grad = math.sqrt(stats.mean([(g - exact_grad) ** 2 for g in vals]))
    return [
        f"{name}:",
        f"  grad mean/std = {mean_grad:.6e} / {std_grad:.6e}  exact = {exact_grad:.6e}",
        f"  grad bias     = {bias_grad:.6e}  rel_bias = {rel_bias_grad:.3e}",
        f"  grad rmse     = {rmse_grad:.6e}  rel_std  = {rel_std_grad:.3e}",
    ]


def report_bundle(bundle: Bundle, *, num_samples: int, seed: int, old_batch_size: int, cg_tol: float) -> None:
    print(f"\n== {bundle.label} ==")
    print(
        f"kernel={bundle.kernel_name}  N={bundle.N}  mtot={bundle.mtot}  M={bundle.M}  "
        f"ell={bundle.lengthscale:.5g}  sigma_f^2={bundle.variance:.5g}  sigma_n^2={bundle.sigmasq:.5g}"
    )
    print(
        f"exact feature grad={bundle.grad_exact:.6e}  trace={bundle.trace_exact:.6e}  quad={bundle.quad_exact:.6e}"
    )
    if bundle.dense_grad_exact is not None:
        print(
            f"dense-K exact grad={bundle.dense_grad_exact:.6e}  "
            f"abs diff={abs(bundle.grad_exact - bundle.dense_grad_exact):.3e}"
        )

    t0 = time.perf_counter()
    old_vals = sample_old_grad(bundle, num_samples=num_samples, seed=seed, batch_size=old_batch_size)
    t_old = time.perf_counter() - t0

    t0 = time.perf_counter()
    new_vals = sample_new_grad(bundle, num_samples=num_samples, seed=seed)
    t_new = time.perf_counter() - t0

    residual_vals = None
    t_residual = None
    if bundle.split_trace_constant is not None:
        t0 = time.perf_counter()
        residual_vals = sample_residual_split_grad(bundle, num_samples=num_samples, seed=seed)
        t_residual = time.perf_counter() - t0

    old_cg_sec, old_cg_iters, old_cg_relres, old_cg_grad = bench_old_lengthscale(bundle, seed=seed, cg_tol=cg_tol)
    new_cg_sec, new_cg_iters, new_cg_relres, new_cg_grad = bench_new_lengthscale(bundle, seed=seed, cg_tol=cg_tol)
    residual_cg = None
    if bundle.split_trace_constant is not None:
        residual_cg = bench_residual_split_lengthscale(bundle, seed=seed, cg_tol=cg_tol)

    for line in summarize_samples("old", old_vals, bundle.grad_exact):
        print(line)
    print(f"  exact-study wall = {t_old:.2f}s")
    print(f"  fast J=1 wall/iters/res = {old_cg_sec:.2f}s / {old_cg_iters:.0f} / {old_cg_relres:.3e}")
    print(
        f"  fast J=1 grad = {old_cg_grad:.6e}  "
        f"abs err = {abs(old_cg_grad - bundle.grad_exact):.3e}  "
        f"rel err = {abs(old_cg_grad - bundle.grad_exact) / max(abs(bundle.grad_exact), 1e-12):.3e}"
    )
    for line in summarize_samples("whole-objective feature-space", new_vals, bundle.grad_exact):
        print(line)
    print(f"  exact-study wall = {t_new:.2f}s")
    print(f"  fast J=1 wall/iters/res = {new_cg_sec:.2f}s / {new_cg_iters:.0f} / {new_cg_relres:.3e}")
    print(
        f"  fast J=1 grad = {new_cg_grad:.6e}  "
        f"abs err = {abs(new_cg_grad - bundle.grad_exact):.3e}  "
        f"rel err = {abs(new_cg_grad - bundle.grad_exact) / max(abs(bundle.grad_exact), 1e-12):.3e}"
    )
    if residual_vals is not None and residual_cg is not None and t_residual is not None:
        residual_cg_sec, residual_cg_iters, residual_cg_relres, residual_cg_grad = residual_cg
        for line in summarize_samples("exact-scale + PSD residual", residual_vals, bundle.grad_exact):
            print(line)
        print(f"  exact-study wall = {t_residual:.2f}s")
        print(
            f"  fast J=1 wall/iters/res = {residual_cg_sec:.2f}s / "
            f"{residual_cg_iters:.0f} / {residual_cg_relres:.3e}"
        )
        print(
            f"  fast J=1 grad = {residual_cg_grad:.6e}  "
            f"abs err = {abs(residual_cg_grad - bundle.grad_exact):.3e}  "
            f"rel err = {abs(residual_cg_grad - bundle.grad_exact) / max(abs(bundle.grad_exact), 1e-12):.3e}"
        )


def main() -> None:
    x_small, y_small = load_usa_temp_subset(192)
    report_bundle(
        make_bundle(
            "small dense check (SE)",
            "SE",
            x_small,
            y_small,
            lengthscale=0.09,
            variance=4.0,
            sigmasq=0.03,
            eps=1e-4,
            nufft_eps=1e-6,
            build_dense_check=True,
        ),
        num_samples=1000,
        seed=1234,
        old_batch_size=1000,
        cg_tol=1e-5,
    )

    report_bundle(
        make_bundle(
            "small dense check (Matern)",
            "Matern",
            x_small,
            y_small,
            lengthscale=0.12,
            variance=2.5,
            sigmasq=0.04,
            eps=1e-4,
            nufft_eps=1e-6,
            build_dense_check=True,
        ),
        num_samples=1000,
        seed=4321,
        old_batch_size=1000,
        cg_tol=1e-5,
    )

    x_prism, y_prism = load_prism_standardized()
    report_bundle(
        make_bundle(
            "full PRISM exact MxM check",
            "SE",
            x_prism,
            y_prism,
            lengthscale=0.09256,
            variance=3.878,
            sigmasq=0.05202,
            eps=1e-4,
            nufft_eps=1e-5,
            build_dense_check=False,
        ),
        num_samples=20,
        seed=0,
        old_batch_size=1,
        cg_tol=1e-5,
    )


if __name__ == "__main__":
    main()
