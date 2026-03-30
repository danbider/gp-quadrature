#!/usr/bin/env python3
"""
Evaluate lengthscale trace estimators against exact ground truth.

We compare:
1. The current old data-space Hutchinson estimator.
2. The full feature-space rewrite that was algebraically correct but unstable.
3. The new split idea: exact amplitude-like term plus a PSD residual estimator.

Ground truth is always the exact approximate EFGP objective:
- small cases: exact dense checks are available
- full PRISM case: exact MxM feature-space systems are formed
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

from efgpnd import (  # noqa: E402
    NUFFT,
    ToeplitzND,
    _cmplx,
    compute_convolution_vector_vectorized_dD,
    create_A_mean,
    create_jacobi_precond,
)
from diagnose_efgpnd_learning_curve import RecordingConjugateGradients  # noqa: E402
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
    ws: torch.Tensor
    s: torch.Tensor
    d_l: torch.Tensor
    c_zero: float
    c_psd: float
    q_psd: torch.Tensor
    q_min: float
    q_max: float
    C: torch.Tensor
    A: torch.Tensor
    G: torch.Tensor
    H_full: torch.Tensor
    H_psd: torch.Tensor
    trace_full_const: float
    trace_psd_const: float
    noise_trace_exact: float
    term1_exact: float
    term1_split_exact: float
    term2_exact: float
    grad_exact: float
    fadj: object
    fwd: object
    toeplitz: object
    A_apply: object
    M_inv: object
    diag_scale: float
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
    build_explicit_f_matrix: bool,
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
    zero_idx = torch.argmin(torch.sum(torch.abs(xis), dim=1)).item()
    ratio = d_l / s.clamp_min(1e-300)
    c_zero = float(ratio[zero_idx].item())
    c_psd = float(ratio.max().item())
    q_psd = c_psd * s - d_l
    q_min = float(q_psd.min().item())
    q_max = float(q_psd.max().item())

    out_shape = (mtot,) * d
    nufft = NUFFT(x, torch.zeros(d, dtype=dtype), h, nufft_eps, cdtype=cdtype)
    fadj = lambda v: nufft.type1(v, out_shape=out_shape).reshape(v.shape[:-1] + (-1,))
    fwd = lambda fk: nufft.type2(fk, out_shape=out_shape)

    m_conv = (mtot - 1) // 2
    v_kernel = compute_convolution_vector_vectorized_dD(m_conv, x, h).to(dtype=cdtype)
    toeplitz = ToeplitzND(v_kernel, force_pow2=True)
    center = tuple(((torch.tensor(v_kernel.shape) - 1) // 2).tolist())
    diag_scale = float(v_kernel[center].real.item())
    C = build_explicit_c(toeplitz, ws.numel(), cdtype)
    A_apply = create_A_mean(ws, toeplitz, torch.tensor(sigmasq, dtype=dtype), cdtype)
    M_inv = create_jacobi_precond(ws, torch.tensor(sigmasq, dtype=dtype), diag_scale=v_kernel[center].real)

    D = torch.diag(ws)
    G = D @ C @ D
    A = G + sigmasq * torch.eye(ws.numel(), dtype=cdtype)
    S_full = torch.diag(d_l.to(cdtype))
    Q_psd = torch.diag(q_psd.to(cdtype))
    H_full = D @ C @ S_full @ C @ D
    H_psd = D @ C @ Q_psd @ C @ D

    Ainv_G = torch.linalg.solve(A, G)
    noise_trace_exact = float((N / sigmasq - torch.trace(Ainv_G).real.item() / sigmasq))
    trace_full_const = float((torch.diagonal(C).real * d_l).sum().item() / sigmasq)
    term1_exact = float((trace_full_const - torch.trace(torch.linalg.solve(A, H_full)).real.item() / sigmasq))

    trace_psd_const = float((torch.diagonal(C).real * q_psd).sum().item() / sigmasq)
    term1_split_exact = float(
        c_psd * (N - sigmasq * noise_trace_exact)
        - trace_psd_const
        + torch.trace(torch.linalg.solve(A, H_psd)).real.item() / sigmasq
    )

    Fy = fadj(y.to(cdtype)).reshape(-1)
    beta_raw = torch.linalg.solve(A, ws * Fy)
    Dbeta = ws * beta_raw
    fadj_alpha = (Fy - C @ Dbeta) / sigmasq
    term2_exact = float(torch.vdot(fadj_alpha, d_l.to(cdtype) * fadj_alpha).real.item())
    grad_exact = 0.5 * (term1_exact - term2_exact)

    explicit_B_old = None
    if build_explicit_f_matrix:
        F = build_explicit_f(fwd, ws.numel(), cdtype)
        K = (F @ torch.diag(s.to(cdtype)) @ F.conj().T).real + sigmasq * torch.eye(N, dtype=dtype)
        dK = (F @ torch.diag(d_l.to(cdtype)) @ F.conj().T).real
        explicit_B_old = torch.linalg.solve(K, dK)

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
        ws=ws,
        s=s,
        d_l=d_l,
        c_zero=c_zero,
        c_psd=c_psd,
        q_psd=q_psd,
        q_min=q_min,
        q_max=q_max,
        C=C,
        A=A,
        G=G,
        H_full=H_full,
        H_psd=H_psd,
        trace_full_const=trace_full_const,
        trace_psd_const=trace_psd_const,
        noise_trace_exact=noise_trace_exact,
        term1_exact=term1_exact,
        term1_split_exact=term1_split_exact,
        term2_exact=term2_exact,
        grad_exact=grad_exact,
        fadj=fadj,
        fwd=fwd,
        toeplitz=toeplitz,
        A_apply=A_apply,
        M_inv=M_inv,
        diag_scale=diag_scale,
        explicit_B_old=explicit_B_old,
    )


def make_rademacher(shape: tuple[int, ...], dtype: torch.dtype, cdtype: torch.dtype, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    z = torch.empty(shape, dtype=dtype)
    z.bernoulli_(0.5)
    z.mul_(2).sub_(1)
    return z.to(cdtype)


def sample_old_term1(bundle: Bundle, num_samples: int, seed: int, batch_size: int = 1) -> torch.Tensor:
    if bundle.explicit_B_old is not None:
        Z = make_rademacher((num_samples, bundle.N), bundle.dtype, bundle.cdtype, seed)
        return ((Z @ bundle.explicit_B_old.to(bundle.cdtype)) * Z).sum(dim=1).real

    outputs = []
    d_l_c = bundle.d_l.to(bundle.cdtype)
    for offset in range(0, num_samples, batch_size):
        block = min(batch_size, num_samples - offset)
        Z = make_rademacher((block, bundle.N), bundle.dtype, bundle.cdtype, seed + offset)
        fadjZ = bundle.fadj(Z).reshape(block, -1)
        DiFZ = d_l_c.unsqueeze(0) * fadjZ
        rhs_data = bundle.fwd(DiFZ).reshape(block, -1)
        B_old = ((bundle.C @ DiFZ.T).T * bundle.ws.unsqueeze(0)).to(bundle.cdtype)
        Beta_old = torch.linalg.solve(bundle.A, B_old.T).T
        mean_part = bundle.fwd(bundle.ws.unsqueeze(0) * Beta_old).reshape(block, -1)
        Alpha_old = (rhs_data - mean_part) / bundle.sigmasq
        outputs.append((Z.conj() * Alpha_old).sum(dim=1).real.cpu())
    return torch.cat(outputs, dim=0)


def sample_full_rewrite_term1(bundle: Bundle, num_samples: int, seed: int) -> torch.Tensor:
    V = make_rademacher((num_samples, bundle.M), bundle.dtype, bundle.cdtype, seed)
    Beta = torch.linalg.solve(bundle.A, bundle.H_full @ V.T).T
    return bundle.trace_full_const - ((V.conj() * Beta).sum(dim=1).real / bundle.sigmasq)


def sample_split_term1(bundle: Bundle, num_samples: int, seed: int) -> torch.Tensor:
    V = make_rademacher((num_samples, bundle.M), bundle.dtype, bundle.cdtype, seed)
    Beta = torch.linalg.solve(bundle.A, bundle.H_psd @ V.T).T
    exact_scale = bundle.c_psd * (bundle.N - bundle.sigmasq * bundle.noise_trace_exact) - bundle.trace_psd_const
    return exact_scale + ((V.conj() * Beta).sum(dim=1).real / bundle.sigmasq)


def bench_old_cg(bundle: Bundle, seed: int, cg_tol: float) -> tuple[float, float, float]:
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
    _ = (Z.conj() * Alpha_old).sum(dim=1).real
    post_sec = time.perf_counter() - t1
    return rhs_sec + stats.solve_time_sec + post_sec, stats.iters_median, stats.rel_res_median


def bench_feature_cg(bundle: Bundle, seed: int, cg_tol: float, *, use_psd_split: bool) -> tuple[float, float, float]:
    t0 = time.perf_counter()
    V = make_rademacher((1, bundle.M), bundle.dtype, bundle.cdtype, seed)
    CDV = bundle.toeplitz(bundle.ws.unsqueeze(0) * V).reshape(1, -1)
    weights = bundle.q_psd if use_psd_split else bundle.d_l
    B = (bundle.ws.unsqueeze(0) * bundle.toeplitz(weights.to(bundle.cdtype).unsqueeze(0) * CDV).reshape(1, -1)).to(bundle.cdtype)
    rhs_sec = time.perf_counter() - t0

    solver = RecordingConjugateGradients(
        bundle.A_apply,
        B,
        torch.zeros_like(B),
        tol=cg_tol,
        early_stopping=True,
        M_inv_apply=bundle.M_inv,
    )
    Beta, stats = solver.solve()

    t1 = time.perf_counter()
    if use_psd_split:
        exact_scale = bundle.c_psd * (bundle.N - bundle.sigmasq * bundle.noise_trace_exact) - bundle.trace_psd_const
        _ = exact_scale + ((V.conj() * Beta).sum(dim=1).real / bundle.sigmasq)
    else:
        _ = bundle.trace_full_const - ((V.conj() * Beta).sum(dim=1).real / bundle.sigmasq)
    post_sec = time.perf_counter() - t1
    return rhs_sec + stats.solve_time_sec + post_sec, stats.iters_median, stats.rel_res_median


def summarize_samples(name: str, values: torch.Tensor, exact_term1: float, exact_grad: float, term2_exact: float) -> list[str]:
    vals = values.tolist()
    grads = [0.5 * (v - term2_exact) for v in vals]
    mean_term1 = stats.mean(vals)
    std_term1 = stats.pstdev(vals)
    mean_grad = stats.mean(grads)
    std_grad = stats.pstdev(grads)
    bias_grad = mean_grad - exact_grad
    rel_bias_grad = abs(bias_grad) / max(abs(exact_grad), 1e-12)
    rel_std_grad = std_grad / max(abs(exact_grad), 1e-12)
    rmse_grad = math.sqrt(stats.mean([(g - exact_grad) ** 2 for g in grads]))
    return [
        f"{name}:",
        f"  term1 mean/std = {mean_term1:.6e} / {std_term1:.6e}  exact = {exact_term1:.6e}",
        f"  grad  mean/std = {mean_grad:.6e} / {std_grad:.6e}  exact = {exact_grad:.6e}",
        f"  grad  bias     = {bias_grad:.6e}  rel_bias = {rel_bias_grad:.3e}",
        f"  grad  rmse     = {rmse_grad:.6e}  rel_std  = {rel_std_grad:.3e}",
    ]


def report_bundle(bundle: Bundle, *, num_samples: int, seed: int, old_batch_size: int, cg_tol: float) -> None:
    print(f"\n== {bundle.label} ==")
    print(
        f"kernel={bundle.kernel_name}  N={bundle.N}  mtot={bundle.mtot}  M={bundle.M}  "
        f"ell={bundle.kernel.get_hyper('lengthscale'):.5g}  sigma_f^2={bundle.variance:.5g}  sigma_n^2={bundle.sigmasq:.5g}"
    )
    print(
        f"split coeffs: c_zero={bundle.c_zero:.6e}  c_psd={bundle.c_psd:.6e}  "
        f"q_min={bundle.q_min:.6e}  q_max={bundle.q_max:.6e}"
    )
    print(
        f"exact checks: full trace={bundle.term1_exact:.6e}  split trace={bundle.term1_split_exact:.6e}  "
        f"abs diff={abs(bundle.term1_exact - bundle.term1_split_exact):.3e}"
    )
    print(f"exact term2={bundle.term2_exact:.6e}  exact grad={bundle.grad_exact:.6e}")

    t0 = time.perf_counter()
    old_vals = sample_old_term1(bundle, num_samples=num_samples, seed=seed, batch_size=old_batch_size)
    t_old = time.perf_counter() - t0

    t0 = time.perf_counter()
    full_vals = sample_full_rewrite_term1(bundle, num_samples=num_samples, seed=seed)
    t_full = time.perf_counter() - t0

    t0 = time.perf_counter()
    split_vals = sample_split_term1(bundle, num_samples=num_samples, seed=seed)
    t_split = time.perf_counter() - t0

    old_cg_sec, old_cg_iters, old_cg_relres = bench_old_cg(bundle, seed=seed, cg_tol=cg_tol)
    full_cg_sec, full_cg_iters, full_cg_relres = bench_feature_cg(bundle, seed=seed, cg_tol=cg_tol, use_psd_split=False)
    split_cg_sec, split_cg_iters, split_cg_relres = bench_feature_cg(bundle, seed=seed, cg_tol=cg_tol, use_psd_split=True)

    for line in summarize_samples("old", old_vals, bundle.term1_exact, bundle.grad_exact, bundle.term2_exact):
        print(line)
    print(f"  exact-study wall = {t_old:.2f}s")
    print(f"  CG J=1 wall/iters/res = {old_cg_sec:.2f}s / {old_cg_iters:.0f} / {old_cg_relres:.3e}")
    for line in summarize_samples("full rewrite", full_vals, bundle.term1_exact, bundle.grad_exact, bundle.term2_exact):
        print(line)
    print(f"  exact-study wall = {t_full:.2f}s")
    print(f"  CG J=1 wall/iters/res = {full_cg_sec:.2f}s / {full_cg_iters:.0f} / {full_cg_relres:.3e}")
    for line in summarize_samples("split PSD residual", split_vals, bundle.term1_exact, bundle.grad_exact, bundle.term2_exact):
        print(line)
    print(f"  exact-study wall = {t_split:.2f}s")
    print(f"  CG J=1 wall/iters/res = {split_cg_sec:.2f}s / {split_cg_iters:.0f} / {split_cg_relres:.3e}")


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
            build_explicit_f_matrix=True,
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
            build_explicit_f_matrix=True,
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
            build_explicit_f_matrix=False,
        ),
        num_samples=20,
        seed=0,
        old_batch_size=1,
        cg_tol=1e-5,
    )


if __name__ == "__main__":
    main()
