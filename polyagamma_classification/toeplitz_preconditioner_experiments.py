from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as nnf
from torch.distributions import NegativeBinomial
from torch.fft import fftn, ifftn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from cg import ConjugateGradients
from kernels import SquaredExponential
from vanilla_gp_sampling import sample_gp_spectral_approx

from pg_classifier import (
    _PGNegativeBinomialLikelihood,
    _VariationalState,
    _build_spectral_state,
    _build_weighted_toeplitz,
    _run_estep,
    _sample_rademacher,
)


def _make_state(
    *,
    n_train: int,
    lengthscale: float,
    variance: float,
    total_count: float,
    nufft_eps: float,
    spectral_eps: float,
    trunc_eps: float,
    cg_tol: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, object, _VariationalState, torch.Tensor]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    d = 2
    X = torch.rand(n_train, d) * 2.0 - 1.0
    f = sample_gp_spectral_approx(
        X,
        num_samples=1,
        length_scale=lengthscale,
        variance=variance,
        spectral_eps=spectral_eps,
        trunc_eps=trunc_eps,
        nufft_eps=nufft_eps,
        seed=12,
    )
    y = NegativeBinomial(
        total_count=torch.tensor(total_count, dtype=torch.float64),
        logits=f,
    ).sample().to(torch.float64)

    kernel = SquaredExponential(
        dimension=d,
        init_lengthscale=lengthscale,
        init_variance=variance,
    )
    spectral = _build_spectral_state(
        X,
        kernel,
        spectral_eps=spectral_eps,
        trunc_eps=trunc_eps,
        nufft_eps=nufft_eps,
        rdtype=torch.float64,
        cdtype=torch.complex128,
        device=torch.device("cpu"),
    )
    likelihood = _PGNegativeBinomialLikelihood(total_count=total_count)
    kappa = likelihood.kappa(y)
    pg_b = likelihood.pg_b(y)
    q = _VariationalState(delta=(0.25 * pg_b).clone())
    q, _ = _run_estep(
        y,
        kappa,
        pg_b,
        likelihood,
        q,
        spectral,
        max_iters=3,
        rho0=0.7,
        gamma=1e-3,
        tol=1e-4,
        n_probes=1,
        cg_tol=cg_tol,
        reuse_probes=True,
        use_exact_weighted_toeplitz_operator=True,
        seed=0,
        verbose=0,
    )
    return X, y, spectral, q, kappa


def _make_circulant_inverse(toeplitz, shift: float):
    fft_kernel = toeplitz.fft_kernel
    denom = shift + fft_kernel
    ns = tuple(toeplitz.ns)
    fft_shape = tuple(toeplitz.fft_shape)
    d = len(ns)
    starts = toeplitz.starts
    ends = toeplitz.ends
    pad = []
    for n, f in zip(reversed(ns), reversed(fft_shape)):
        pad += [0, f - n]

    def apply(v: torch.Tensor) -> torch.Tensor:
        orig_flat = False
        if v.shape[-1] == toeplitz.size:
            orig_flat = True
            batch_shape = v.shape[:-1]
            x = v.reshape(*batch_shape, *ns)
        else:
            x = v
            batch_shape = x.shape[:-d]

        x = x.to(dtype=fft_kernel.dtype)
        x_pad = nnf.pad(x, pad)
        x_fft = fftn(x_pad, dim=tuple(range(-d, 0)))
        y_fft = x_fft / denom
        y = ifftn(y_fft, dim=tuple(range(-d, 0)))

        slices = [slice(None)] * (y.ndim - d)
        for st, en in zip(starts, ends):
            slices.append(slice(st, en))
        y = y[tuple(slices)]

        if orig_flat:
            y = y.reshape(*batch_shape, toeplitz.size)
        return y

    return apply


def _make_deflation_preconditioner(
    *,
    A_apply,
    m: int,
    k: int,
    bulk_scale: float,
    seed: int,
    power_iters: int,
):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    Q = torch.randn((m, k), generator=gen, dtype=torch.complex128)
    Q, _ = torch.linalg.qr(Q, mode="reduced")

    for _ in range(power_iters):
        AQ = A_apply(Q.T).T
        Q, _ = torch.linalg.qr(AQ, mode="reduced")

    AQ = A_apply(Q.T).T
    T = Q.conj().T @ AQ
    evals, evecs = torch.linalg.eigh(T)
    order = torch.argsort(evals.real, descending=True)
    evals = evals[order].real.clamp_min(1e-8)
    U = Q @ evecs[:, order]
    UH = U.conj().T

    def M_inv(v: torch.Tensor) -> torch.Tensor:
        vector_input = v.dim() == 1
        rhs = v[:, None] if vector_input else v.T
        coeff = UH @ rhs
        low = U @ (coeff / evals[:, None])
        rem = rhs - U @ coeff
        out = low + rem / bulk_scale
        return out[:, 0] if vector_input else out.T

    return M_inv


def _run_single_case(
    *,
    n_train: int,
    lengthscale: float,
    variance: float,
    total_count: float,
    nufft_eps: float,
    spectral_eps: float,
    trunc_eps: float,
    cg_tol: float,
    seed: int,
    power_iters: int,
) -> list[dict[str, float]]:
    _, _, spectral, q, kappa = _make_state(
        n_train=n_train,
        lengthscale=lengthscale,
        variance=variance,
        total_count=total_count,
        nufft_eps=nufft_eps,
        spectral_eps=spectral_eps,
        trunc_eps=trunc_eps,
        cg_tol=cg_tol,
        seed=seed,
    )

    D = spectral.ws
    weighted_toeplitz = _build_weighted_toeplitz(q.delta, spectral)

    def A_apply(u: torch.Tensor) -> torch.Tensor:
        if u.dim() == 1:
            t = D * u
            return u + D * weighted_toeplitz(t)
        t = u * D[None, :]
        return u + D[None, :] * weighted_toeplitz(t)

    rhs_data = torch.stack(
        [
            kappa,
            _sample_rademacher(
                (q.delta.numel(),),
                device=torch.device("cpu"),
                dtype=torch.float64,
                seed=0,
            ),
        ],
        dim=0,
    )
    rhs = D * spectral.fadj_batched(rhs_data.to(dtype=D.dtype))

    ws2 = (D.abs() ** 2).real
    diag = 1.0 + q.delta.sum().real * ws2

    jacobi = lambda v: v / diag

    circ_scale = float(ws2.mean().item())
    circulant = _make_circulant_inverse(weighted_toeplitz, 1.0 / circ_scale)

    def circ_meanws2(v: torch.Tensor) -> torch.Tensor:
        return circulant(v) / circ_scale

    bulk_scale = float(diag.mean().item())

    strategies: list[tuple[str, object, float]] = [
        ("none", None, 0.0),
        ("jacobi", jacobi, 0.0),
        ("circ_meanws2", circ_meanws2, 0.0),
    ]

    for k in (8, 16, 32):
        t0 = time.time()
        Minv = _make_deflation_preconditioner(
            A_apply=A_apply,
            m=D.numel(),
            k=k,
            bulk_scale=bulk_scale,
            seed=seed,
            power_iters=power_iters,
        )
        setup_sec = time.time() - t0
        strategies.append((f"deflate{k}", Minv, setup_sec))

    rows: list[dict[str, float]] = []
    for name, Minv, setup_sec in strategies:
        t0 = time.time()
        cg = ConjugateGradients(
            A_apply,
            rhs,
            x0=torch.zeros_like(rhs),
            tol=cg_tol,
            max_iter=2000,
            early_stopping=True,
            M_inv_apply=Minv,
        )
        sol = cg.solve()
        solve_sec = time.time() - t0
        residual = rhs - A_apply(sol)
        rel_res = (
            torch.linalg.norm(residual, dim=1).real
            / torch.linalg.norm(rhs, dim=1).real
        ).max().item()
        rows.append(
            {
                "n": float(n_train),
                "m": float(D.numel()),
                "strategy": name,
                "setup_sec": setup_sec,
                "solve_sec": solve_sec,
                "total_sec": setup_sec + solve_sec,
                "iters": float(cg.iters_completed),
                "max_rel_residual": rel_res,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-values", type=int, nargs="+", default=[50_000, 500_000])
    parser.add_argument("--lengthscale", type=float, default=0.2)
    parser.add_argument("--variance", type=float, default=1.0)
    parser.add_argument("--total-count", type=float, default=3.0)
    parser.add_argument("--nufft-eps", type=float, default=1e-4)
    parser.add_argument("--spectral-eps", type=float, default=1e-4)
    parser.add_argument("--trunc-eps", type=float, default=1e-4)
    parser.add_argument("--cg-tol", type=float, default=1e-6)
    parser.add_argument("--power-iters", type=int, default=12)
    parser.add_argument("--seed", type=int, default=760)
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    rows: list[dict[str, float]] = []
    for n in args.n_values:
        rows.extend(
            _run_single_case(
                n_train=n,
                lengthscale=args.lengthscale,
                variance=args.variance,
                total_count=args.total_count,
                nufft_eps=args.nufft_eps,
                spectral_eps=args.spectral_eps,
                trunc_eps=args.trunc_eps,
                cg_tol=args.cg_tol,
                seed=args.seed,
                power_iters=args.power_iters,
            )
        )

    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
