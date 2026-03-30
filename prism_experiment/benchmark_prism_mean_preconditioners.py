#!/usr/bin/env python3
"""
Benchmark mean-solve preconditioners on full PRISM in the sad regime.

We compare:

- none
- Jacobi:       (N |w|^2 + sigma^2 I)^{-1}
- Toeplitz-circ scalar:
    (sigma^2 I + alpha C_circ)^{-1}
- Toeplitz-circ sandwich:
    D^{-1} (C_circ + tau I)^{-1} D^{-1}

where C_circ is a wrapped circulant approximation to the Toeplitz operator
F^* F on the tensor grid.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
from torch.fft import fftn, ifftn

torch.set_num_threads(1)

THIS_DIR = Path(__file__).resolve().parent
REPO_DIR = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
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
from load_prism import load_prism_dataset_torch  # noqa: E402
from utils.kernels import get_xis  # noqa: E402


@dataclass
class State:
    name: str
    lengthscale: float
    variance: float
    sigmasq: float
    mtot: int
    M: int
    ws: torch.Tensor
    v_kernel: torch.Tensor
    rhs: torch.Tensor
    A_apply: Callable[[torch.Tensor], torch.Tensor]
    diag_scale: float


def load_standardized_prism() -> tuple[torch.Tensor, torch.Tensor]:
    x, y = load_prism_dataset_torch("prism_tmean_us_30s_2020_avg_30y")
    x = x.to(dtype=torch.float64)
    y = y.to(dtype=torch.float64)
    x = (x - x.min(dim=0).values) / (x.max(dim=0).values - x.min(dim=0).values)
    y = (y - y.mean()) / y.std()
    return x, y


def build_state(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    name: str,
    lengthscale: float,
    variance: float,
    sigmasq: float,
    eps: float,
    nufft_eps: float,
) -> State:
    kernel = SquaredExponential(dimension=2)
    kernel.set_hyper("lengthscale", lengthscale)
    kernel.set_hyper("variance", variance)

    dtype = x.dtype
    cdtype = _cmplx(dtype)
    d = x.shape[1]
    L = float((x.max(dim=0).values - x.min(dim=0).values).max().item())

    xis_1d, h, mtot = get_xis(kernel, eps=eps, L=L, use_integral=True, l2scaled=False)
    xis_1d = xis_1d.to(dtype=dtype)
    xis = torch.stack(torch.meshgrid(*(xis_1d for _ in range(d)), indexing="ij"), dim=-1).reshape(-1, d)
    ws = torch.sqrt(kernel.spectral_density(xis).to(dtype=cdtype) * h**d)

    OUT = (mtot,) * d
    nufft = NUFFT(x, torch.zeros(d, dtype=dtype), h, nufft_eps, cdtype=cdtype)
    fadj = lambda v: nufft.type1(v, out_shape=OUT).reshape(-1)

    m_conv = (mtot - 1) // 2
    v_kernel = compute_convolution_vector_vectorized_dD(m_conv, x, h).to(dtype=cdtype)
    toeplitz = ToeplitzND(v_kernel, force_pow2=True)
    A_apply = create_A_mean(ws, toeplitz, torch.tensor(sigmasq, dtype=dtype), cdtype)
    rhs = ws * fadj(y).reshape(-1)

    center = tuple(((torch.tensor(v_kernel.shape) - 1) // 2).tolist())
    diag_scale = float(v_kernel[center].real.item())
    return State(
        name=name,
        lengthscale=lengthscale,
        variance=variance,
        sigmasq=sigmasq,
        mtot=mtot,
        M=ws.numel(),
        ws=ws,
        v_kernel=v_kernel,
        rhs=rhs,
        A_apply=A_apply,
        diag_scale=diag_scale,
    )


def wrap_to_circulant_kernel(v_kernel: torch.Tensor) -> torch.Tensor:
    ns = [(L + 1) // 2 for L in v_kernel.shape]
    center = [n - 1 for n in ns]
    circ = torch.zeros(ns, dtype=v_kernel.dtype, device=v_kernel.device)
    for idx in torch.cartesian_prod(*[torch.arange(L, device=v_kernel.device) for L in v_kernel.shape]):
        circ_idx = tuple(int((idx[d].item() - center[d]) % ns[d]) for d in range(len(ns)))
        circ[circ_idx] += v_kernel[tuple(int(i.item()) for i in idx)]
    return circ


def make_circulant_inverse(circ_kernel: torch.Tensor, denom: torch.Tensor):
    fft_dims = tuple(range(-circ_kernel.ndim, 0))

    def M_inv(v: torch.Tensor) -> torch.Tensor:
        if v.ndim == 1:
            x = v.reshape(circ_kernel.shape)
            xhat = fftn(x, dim=fft_dims)
            y = ifftn(xhat / denom, dim=fft_dims)
            return y.reshape(-1)
        x = v.reshape(v.shape[0], *circ_kernel.shape)
        xhat = fftn(x, dim=tuple(range(-circ_kernel.ndim, 0)))
        y = ifftn(xhat / denom.unsqueeze(0), dim=tuple(range(-circ_kernel.ndim, 0)))
        return y.reshape(v.shape)

    return M_inv


def make_preconditioners(state: State) -> Dict[str, Callable[[torch.Tensor], torch.Tensor] | None]:
    preconds: Dict[str, Callable[[torch.Tensor], torch.Tensor] | None] = {"none": None}

    abs2 = state.ws.abs().pow(2).real
    jacobi_diag = state.diag_scale * abs2 + state.sigmasq
    preconds["jacobi_Nws2"] = lambda v: v / jacobi_diag

    circ_kernel = wrap_to_circulant_kernel(state.v_kernel)
    circ_eigs = fftn(circ_kernel, dim=tuple(range(circ_kernel.ndim))).real
    eig_floor = 1e-10 * max(float(circ_eigs.abs().max().item()), 1.0)
    circ_eigs = circ_eigs.clamp_min(eig_floor)

    alpha_mean = float(abs2.mean().item())
    alpha_max = float(abs2.max().item())
    denom_mean = (state.sigmasq + alpha_mean * circ_eigs).to(dtype=state.v_kernel.dtype)
    denom_max = (state.sigmasq + alpha_max * circ_eigs).to(dtype=state.v_kernel.dtype)
    preconds["circ_scalar_meanws2"] = make_circulant_inverse(circ_kernel, denom_mean)
    preconds["circ_scalar_maxws2"] = make_circulant_inverse(circ_kernel, denom_max)

    w_abs = state.ws.abs().real
    w_floor = torch.quantile(w_abs, 0.05).clamp_min(1e-8)
    w_safe = w_abs.clamp_min(w_floor).to(dtype=state.v_kernel.dtype)
    tau_med = float((state.sigmasq / (w_abs.clamp_min(w_floor) ** 2)).median().item())
    tau_geom = float(torch.exp(torch.log(state.sigmasq / (w_abs.clamp_min(w_floor) ** 2)).mean()).item())

    denom_sand_med = (circ_eigs + tau_med).to(dtype=state.v_kernel.dtype)
    denom_sand_geom = (circ_eigs + tau_geom).to(dtype=state.v_kernel.dtype)

    circ_inv_med = make_circulant_inverse(circ_kernel, denom_sand_med)
    circ_inv_geom = make_circulant_inverse(circ_kernel, denom_sand_geom)

    preconds["circ_sandwich_med"] = lambda v: circ_inv_med(v / w_safe) / w_safe
    preconds["circ_sandwich_geom"] = lambda v: circ_inv_geom(v / w_safe) / w_safe
    return preconds


def benchmark_state(state: State, *, cg_tol: float) -> list[Dict[str, float]]:
    rows = []
    for name, m_inv in make_preconditioners(state).items():
        solver = RecordingConjugateGradients(
            state.A_apply,
            state.rhs,
            torch.zeros_like(state.rhs),
            tol=cg_tol,
            early_stopping=True,
            M_inv_apply=m_inv,
        )
        soln, stats = solver.solve()
        rel_res = float((torch.linalg.norm(state.rhs - state.A_apply(soln)) / torch.linalg.norm(state.rhs)).item())
        rows.append({
            "state": state.name,
            "preconditioner": name,
            "iters": int(stats.iters_completed),
            "rel_res": rel_res,
            "cap_hit": int(stats.iters_completed == 2 * state.M),
        })
    return rows


def main() -> None:
    eps = 1e-4
    nufft_eps = 1e-5
    cg_tol = 1e-5
    x, y = load_standardized_prism()

    states = [
        build_state(
            x, y,
            name="iter40",
            lengthscale=0.09256,
            variance=3.878,
            sigmasq=0.05202,
            eps=eps,
            nufft_eps=nufft_eps,
        ),
        build_state(
            x, y,
            name="final",
            lengthscale=0.07518,
            variance=5.258,
            sigmasq=0.05606,
            eps=eps,
            nufft_eps=nufft_eps,
        ),
    ]

    print(f"N={x.shape[0]}")
    for state in states:
        print(
            f"\nState {state.name}: ell={state.lengthscale}, sigma_f^2={state.variance}, "
            f"sigma_n^2={state.sigmasq}, mtot={state.mtot}, M={state.M}"
        )
        print("preconditioner         iters    rel_res      cap_hit")
        rows = benchmark_state(state, cg_tol=cg_tol)
        for row in rows:
            print(f"{row['preconditioner']:<22} {row['iters']:>6}   {row['rel_res']:.3e}   {bool(row['cap_hit'])}")


if __name__ == "__main__":
    main()
