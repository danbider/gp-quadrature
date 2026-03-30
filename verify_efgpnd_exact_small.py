#!/usr/bin/env python3
"""
Direct exact checks for EFGPND gradients on small real-data subsets.

This validates the current `EFGPND.compute_gradients()` implementation against
the exact dense gradient of the same approximate objective, using the same
Rademacher probes for the stochastic trace terms.
"""

from __future__ import annotations

import math
import os
import sys
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

from efgpnd import EFGPND, NUFFT, ToeplitzND, _cmplx, compute_convolution_vector_vectorized_dD
from kernels.squared_exponential import SquaredExponential
from utils.kernels import get_xis


@dataclass
class ExactBundle:
    x: torch.Tensor
    y: torch.Tensor
    kernel: SquaredExponential
    sigmasq: torch.Tensor
    eps: float
    nufft_eps: float
    dtype: torch.dtype
    cdtype: torch.dtype
    N: int
    M: int
    mtot: int
    ws: torch.Tensor
    Dprime: torch.Tensor
    c0: torch.Tensor
    F: torch.Tensor
    C: torch.Tensor
    A: torch.Tensor
    G: torch.Tensor


def load_usa_temp_subset(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.load(REPO_DIR / "data" / "usa_temp_data.pt")
    x = data["x"].to(dtype=torch.float64)[:n]
    y = data["y"].to(dtype=torch.float64)[:n]
    x = (x - x.min(dim=0).values) / (x.max(dim=0).values - x.min(dim=0).values)
    y = (y - y.mean()) / y.std()
    return x, y


def build_explicit_f_matrix(fwd, M: int, cdtype: torch.dtype, batch_size: int = 64) -> torch.Tensor:
    eye = torch.eye(M, dtype=cdtype)
    cols = []
    for start in range(0, M, batch_size):
        stop = min(start + batch_size, M)
        cols.append(fwd(eye[start:stop]).T)
    return torch.cat(cols, dim=1)


def build_explicit_c_matrix(toeplitz, M: int, cdtype: torch.dtype, batch_size: int = 64) -> torch.Tensor:
    eye = torch.eye(M, dtype=cdtype)
    cols = []
    for start in range(0, M, batch_size):
        stop = min(start + batch_size, M)
        cols.append(toeplitz(eye[start:stop]).T)
    return torch.cat(cols, dim=1)


def make_bundle(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    lengthscale: float,
    variance: float,
    sigmasq: float,
    eps: float,
    nufft_eps: float,
) -> ExactBundle:
    dtype = x.dtype
    cdtype = _cmplx(dtype)
    d = x.shape[1]
    N = x.shape[0]

    kernel = SquaredExponential(dimension=d)
    kernel.set_hyper("lengthscale", lengthscale)
    kernel.set_hyper("variance", variance)

    x0 = x.min(dim=0).values
    x1 = x.max(dim=0).values
    L = float((x1 - x0).max().item())
    xis_1d, h, mtot = get_xis(kernel_obj=kernel, eps=eps, L=L, use_integral=True, l2scaled=False)
    grids = torch.meshgrid(*(xis_1d for _ in range(d)), indexing="ij")
    xis = torch.stack(grids, dim=-1).view(-1, d)
    ws = torch.sqrt(kernel.spectral_density(xis).to(dtype=cdtype) * h**d)
    Dprime = (h**d * kernel.spectral_grad(xis)).to(cdtype)

    out_shape = (mtot,) * d
    nufft = NUFFT(x, torch.zeros(d, dtype=dtype), h, nufft_eps, cdtype=cdtype)
    fwd = lambda fk: nufft.type2(fk, out_shape=out_shape)

    m_conv = (mtot - 1) // 2
    v_kernel = compute_convolution_vector_vectorized_dD(m_conv, x, h).to(dtype=cdtype)
    toeplitz = ToeplitzND(v_kernel, force_pow2=True)
    center = tuple(((torch.tensor(v_kernel.shape) - 1) // 2).tolist())
    c0 = v_kernel[center].real

    M = ws.numel()
    F = build_explicit_f_matrix(fwd, M, cdtype)
    C = build_explicit_c_matrix(toeplitz, M, cdtype)
    D = torch.diag(ws)
    G = D @ C @ D
    sigmasq_tensor = torch.tensor(sigmasq, dtype=dtype)
    A = G + sigmasq_tensor * torch.eye(M, dtype=cdtype)

    return ExactBundle(
        x=x,
        y=y,
        kernel=kernel,
        sigmasq=sigmasq_tensor,
        eps=eps,
        nufft_eps=nufft_eps,
        dtype=dtype,
        cdtype=cdtype,
        N=N,
        M=M,
        mtot=mtot,
        ws=ws,
        Dprime=Dprime,
        c0=c0,
        F=F,
        C=C,
        A=A,
        G=G,
    )


def exact_raw_gradient(bundle: ExactBundle, *, trace_samples: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = bundle.dtype
    cdtype = bundle.cdtype
    N = bundle.N
    variance = torch.tensor(bundle.kernel.get_hyper("variance"), dtype=dtype)
    sigmasq = bundle.sigmasq

    K = (bundle.F @ torch.diag(bundle.ws.abs().pow(2).to(cdtype)) @ bundle.F.conj().T).real
    Kn = K + sigmasq * torch.eye(N, dtype=dtype)
    alpha = torch.linalg.solve(Kn, bundle.y)
    alpha_c = alpha.to(cdtype)

    dK_length = (bundle.F @ torch.diag(bundle.Dprime[:, 0]) @ bundle.F.conj().T).real
    Kinv_dK_length = torch.linalg.solve(Kn, dK_length)

    term2_length = torch.vdot(alpha_c, (dK_length @ alpha).to(cdtype)).real
    y_alpha = torch.dot(bundle.y, alpha).real
    alpha_norm = torch.dot(alpha, alpha).real
    term2_variance = (y_alpha - sigmasq * alpha_norm) / variance
    term2_noise = alpha_norm

    torch.manual_seed(seed)
    Z = torch.empty((trace_samples, N), dtype=dtype)
    Z.bernoulli_(0.5)
    Z.mul_(2).sub_(1)
    term1_length = ((Z @ Kinv_dK_length) * Z).sum(dim=1).mean()

    V = torch.empty((trace_samples, bundle.M), dtype=dtype)
    V.bernoulli_(0.5)
    V.mul_(2).sub_(1)
    V = V.to(cdtype)
    Beta_noise = torch.linalg.solve(bundle.A, bundle.G @ V.T).T
    term1_noise = torch.tensor(N, dtype=dtype) / sigmasq - ((V.conj() * Beta_noise).sum(dim=1).real / sigmasq).mean()
    term1_variance = (torch.tensor(N, dtype=dtype) - sigmasq * term1_noise) / variance

    grad_pos = 0.5 * torch.stack(
        [term1_length - term2_length, term1_variance - term2_variance, term1_noise - term2_noise]
    )
    pos = torch.tensor(
        [bundle.kernel.get_hyper("lengthscale"), bundle.kernel.get_hyper("variance"), float(sigmasq.item())],
        dtype=dtype,
    )
    grad_raw = grad_pos * pos
    return grad_pos, grad_raw


def component_summary(label: str, got: torch.Tensor, ref: torch.Tensor) -> list[str]:
    names = ["lengthscale", "variance", "sigmasq"]
    lines = [label]
    for i, name in enumerate(names):
        abs_err = float((got[i] - ref[i]).abs().item())
        rel_err = float(((got[i] - ref[i]).abs() / ref[i].abs().clamp_min(1e-12)).item())
        lines.append(
            f"  {name:<11} got={float(got[i].item()): .6e}  ref={float(ref[i].item()): .6e}  "
            f"abs={abs_err:.3e}  rel={rel_err:.3e}"
        )
    total_rel = float((torch.linalg.norm(got - ref) / torch.linalg.norm(ref).clamp_min(1e-12)).item())
    lines.append(f"  total rel={total_rel:.3e}")
    return lines


def run_case(
    *,
    label: str,
    x: torch.Tensor,
    y: torch.Tensor,
    lengthscale: float,
    variance: float,
    sigmasq: float,
    eps: float,
    trace_samples: int,
    cg_tol: float,
    seed: int,
) -> None:
    print(f"\n== {label} ==")
    bundle = make_bundle(
        x,
        y,
        lengthscale=lengthscale,
        variance=variance,
        sigmasq=sigmasq,
        eps=eps,
        nufft_eps=1e-6,
    )

    kernel = SquaredExponential(dimension=x.shape[1])
    kernel.set_hyper("lengthscale", lengthscale)
    kernel.set_hyper("variance", variance)
    model = EFGPND(
        x,
        y,
        kernel=kernel,
        eps=eps,
        sigmasq=sigmasq,
        estimate_params=False,
        opts={
            "mean_cg_warm_start": False,
            "mean_cg_preconditioner": True,
            "trace_cg_preconditioner": True,
        },
    )

    torch.manual_seed(seed)
    grad_raw = model.compute_gradients(
        trace_samples=trace_samples,
        cg_tol=cg_tol,
        apply_gradients=False,
    )
    grad_pos = grad_raw / model._gp_params.pos.detach()

    exact_pos, exact_raw = exact_raw_gradient(bundle, trace_samples=trace_samples, seed=seed)

    print(f"N={bundle.N}, mtot={bundle.mtot}, M={bundle.M}, cg_tol={cg_tol}, trace_samples={trace_samples}")
    print(f"logged stats: {model.last_gradient_stats}")
    for line in component_summary("raw gradient", grad_raw.detach().cpu(), exact_raw.cpu()):
        print(line)
    for line in component_summary("positive gradient", grad_pos.detach().cpu(), exact_pos.cpu()):
        print(line)


def main() -> None:
    x96, y96 = load_usa_temp_subset(96)
    x192, y192 = load_usa_temp_subset(192)

    run_case(
        label="moderate regime, tight CG",
        x=x96,
        y=y96,
        lengthscale=0.14,
        variance=1.9,
        sigmasq=0.07,
        eps=1e-4,
        trace_samples=64,
        cg_tol=1e-8,
        seed=1234,
    )

    run_case(
        label="harder regime, tight CG",
        x=x192,
        y=y192,
        lengthscale=0.09,
        variance=4.0,
        sigmasq=0.03,
        eps=1e-4,
        trace_samples=64,
        cg_tol=1e-8,
        seed=4321,
    )

    run_case(
        label="harder regime, looser CG",
        x=x192,
        y=y192,
        lengthscale=0.09,
        variance=4.0,
        sigmasq=0.03,
        eps=1e-4,
        trace_samples=64,
        cg_tol=1e-5,
        seed=4321,
    )


if __name__ == "__main__":
    main()
