from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from efgpnd import NUFFT, ToeplitzND, compute_convolution_vector_vectorized_dD

from fixed_hyperparam_preconditioning import (
    _delta_summary,
    _feature_operator_parts,
    _generate_nb_data,
    _solve_with_inner_cg,
)
from pg_classifier import (
    ConjugateGradients,
    _PGNegativeBinomialLikelihood,
    _VariationalState,
    _build_spectral_state,
    _make_kernel,
    _run_estep,
    _sample_rademacher,
)


@dataclass(frozen=True)
class Strategy:
    name: str
    kind: str
    num_bins: int = 1
    precond_tol: float = 1e-2
    precond_max_iter: int = 200


@dataclass
class Record:
    n: int
    feature_dim: int
    mtot: int
    solve_kind: str
    strategy: str
    delta_cv: float
    delta_rel_l2: float
    batch_size: int
    outer_cg_iters: int
    outer_cg_ms: float
    total_ms: float
    precond_setup_ms: float
    precond_ms: float
    precond_calls: int
    precond_iters_total: int
    rel_res: float
    num_bins: int
    lengthscale: float
    variance: float


def _materialize_toeplitz_matrix(
    toeplitz: ToeplitzND,
    *,
    size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    basis = torch.eye(size, dtype=dtype, device=device)
    # `toeplitz(basis)` returns the images of the basis vectors stacked as rows,
    # so those rows are the columns of the linear operator.
    T = toeplitz(basis).T.contiguous()
    return 0.5 * (T + T.conj().T)


def _dense_cholesky_apply(chol: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if v.dim() == 1:
        return torch.cholesky_solve(v[:, None], chol).squeeze(1)
    return torch.cholesky_solve(v.T.contiguous(), chol).T


def _build_scalar_toeplitz_cholesky_preconditioner(
    spectral,
    delta: torch.Tensor,
    Ds: torch.Tensor,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], float]:
    size = spectral.xis.shape[0]
    eye = torch.eye(size, dtype=spectral.ws.dtype, device=Ds.device)
    t0 = time.perf_counter()
    T = _materialize_toeplitz_matrix(
        spectral.toeplitz,
        size=size,
        dtype=spectral.ws.dtype,
        device=Ds.device,
    )
    wbar = delta.to(dtype=spectral.ws.dtype).real.mean()
    M = eye + wbar * (Ds[:, None] * T * Ds[None, :])
    M = 0.5 * (M + M.conj().T)
    chol = torch.linalg.cholesky(M + 1e-10 * eye)
    setup_ms = 1000.0 * (time.perf_counter() - t0)
    return lambda v: _dense_cholesky_apply(chol, v), setup_ms


def _bin_indices_by_delta(delta: torch.Tensor, num_bins: int) -> list[torch.Tensor]:
    perm = torch.argsort(delta.real)
    bins: list[torch.Tensor] = []
    n = perm.numel()
    for b in range(num_bins):
        start = (b * n) // num_bins
        end = ((b + 1) * n) // num_bins
        if end > start:
            bins.append(perm[start:end])
    return bins


def _build_binned_toeplitz_cholesky_preconditioner(
    X: torch.Tensor,
    spectral,
    delta: torch.Tensor,
    Ds: torch.Tensor,
    *,
    num_bins: int,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], float]:
    size = spectral.xis.shape[0]
    eye = torch.eye(size, dtype=spectral.ws.dtype, device=Ds.device)
    m_conv = (spectral.mtot - 1) // 2
    t0 = time.perf_counter()
    M = eye.clone()
    for idx in _bin_indices_by_delta(delta, num_bins):
        if idx.numel() == 0:
            continue
        X_bin = X[idx]
        weight = delta[idx].to(dtype=spectral.ws.dtype).real.mean().to(dtype=spectral.ws.dtype)
        v_kernel = compute_convolution_vector_vectorized_dD(m_conv, X_bin, spectral.h).to(dtype=spectral.ws.dtype)
        toeplitz_bin = ToeplitzND(v_kernel, force_pow2=True)
        T_bin = _materialize_toeplitz_matrix(
            toeplitz_bin,
            size=size,
            dtype=spectral.ws.dtype,
            device=Ds.device,
        )
        M = M + weight * (Ds[:, None] * T_bin * Ds[None, :])
    M = 0.5 * (M + M.conj().T)
    chol = torch.linalg.cholesky(M + 1e-10 * eye)
    setup_ms = 1000.0 * (time.perf_counter() - t0)
    return lambda v: _dense_cholesky_apply(chol, v), setup_ms


def _compute_weighted_convolution_vector(
    x: torch.Tensor,
    weights: torch.Tensor,
    *,
    m_conv: int,
    h: float,
) -> torch.Tensor:
    if x.ndim == 1:
        x = x[:, None]
    d = x.shape[1]
    dtype_real = x.dtype
    dtype_cmplx = weights.dtype
    xcen = torch.zeros(d, device=x.device, dtype=dtype_real)
    nufft_op = NUFFT(x, xcen, h, eps=6e-8, cdtype=dtype_cmplx, device=x.device)
    return nufft_op.type1(weights.to(dtype=dtype_cmplx), out_shape=tuple([4 * m_conv + 1] * d))


def _build_exact_weighted_toeplitz_cholesky_preconditioner(
    X: torch.Tensor,
    spectral,
    delta: torch.Tensor,
    Ds: torch.Tensor,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], float]:
    size = spectral.xis.shape[0]
    eye = torch.eye(size, dtype=spectral.ws.dtype, device=Ds.device)
    m_conv = (spectral.mtot - 1) // 2
    t0 = time.perf_counter()
    v_weighted = _compute_weighted_convolution_vector(
        X,
        delta.to(dtype=spectral.ws.dtype),
        m_conv=m_conv,
        h=spectral.h,
    )
    toeplitz_weighted = ToeplitzND(v_weighted, force_pow2=True)
    T_weighted = _materialize_toeplitz_matrix(
        toeplitz_weighted,
        size=size,
        dtype=spectral.ws.dtype,
        device=Ds.device,
    )
    M = eye + (Ds[:, None] * T_weighted * Ds[None, :])
    M = 0.5 * (M + M.conj().T)
    chol = torch.linalg.cholesky(M + 1e-10 * eye)
    setup_ms = 1000.0 * (time.perf_counter() - t0)
    return lambda v: _dense_cholesky_apply(chol, v), setup_ms


def _default_strategies() -> list[Strategy]:
    return [
        Strategy(name="none", kind="none"),
        Strategy(name="toeplitz_pcg", kind="toeplitz_pcg", precond_tol=1e-2, precond_max_iter=200),
        Strategy(name="toeplitz_chol", kind="toeplitz_chol"),
        Strategy(name="exact_weighted_toeplitz_chol", kind="exact_weighted_toeplitz_chol"),
        Strategy(name="binned_toeplitz_2_chol", kind="binned_toeplitz_chol", num_bins=2),
        Strategy(name="binned_toeplitz_4_chol", kind="binned_toeplitz_chol", num_bins=4),
        Strategy(name="binned_toeplitz_8_chol", kind="binned_toeplitz_chol", num_bins=8),
    ]


def _benchmark_solve(
    *,
    solve_kind: str,
    spectral,
    X: torch.Tensor,
    delta: torch.Tensor,
    rhs: torch.Tensor,
    strategy: Strategy,
    cg_tol: float,
    delta_cv: float,
    delta_rel_l2: float,
    lengthscale: float,
    variance: float,
) -> Record:
    Ds, apply_A, apply_A_toeplitz, _ = _feature_operator_parts(spectral, delta)
    rhs = rhs.to(dtype=spectral.ws.dtype)
    precond_setup_ms = 0.0
    precond_ms = 0.0
    precond_calls = 0
    precond_iters_total = 0
    total_t0 = time.perf_counter()

    M_inv_apply = None
    if strategy.kind == "toeplitz_pcg":

        def toeplitz_precond(v: torch.Tensor) -> torch.Tensor:
            nonlocal precond_ms, precond_calls, precond_iters_total
            precond_calls += 1
            t0 = time.perf_counter()
            soln, inner_iters = _solve_with_inner_cg(
                apply_A_toeplitz,
                v,
                tol=strategy.precond_tol,
                max_iter=strategy.precond_max_iter,
            )
            precond_ms += 1000.0 * (time.perf_counter() - t0)
            precond_iters_total += inner_iters
            return soln

        M_inv_apply = toeplitz_precond
    elif strategy.kind == "toeplitz_chol":
        M_inv_apply, precond_setup_ms = _build_scalar_toeplitz_cholesky_preconditioner(spectral, delta, Ds)
    elif strategy.kind == "exact_weighted_toeplitz_chol":
        M_inv_apply, precond_setup_ms = _build_exact_weighted_toeplitz_cholesky_preconditioner(
            X,
            spectral,
            delta,
            Ds,
        )
    elif strategy.kind == "binned_toeplitz_chol":
        M_inv_apply, precond_setup_ms = _build_binned_toeplitz_cholesky_preconditioner(
            X,
            spectral,
            delta,
            Ds,
            num_bins=strategy.num_bins,
        )

    outer_t0 = time.perf_counter()
    cg = ConjugateGradients(
        apply_A,
        rhs,
        x0=torch.zeros_like(rhs),
        tol=cg_tol,
        max_iter=2000,
        early_stopping=True,
        M_inv_apply=M_inv_apply,
    )
    soln = cg.solve()
    outer_cg_ms = 1000.0 * (time.perf_counter() - outer_t0)
    outer_cg_iters = int(cg.iters_completed)

    residual = rhs - apply_A(soln)
    rel_res = float(
        (
            torch.linalg.norm(residual.reshape(-1)).real
            / max(float(torch.linalg.norm(rhs.reshape(-1)).real.item()), 1e-16)
        ).item()
    )

    return Record(
        n=int(X.shape[0]),
        feature_dim=int(spectral.xis.shape[0]),
        mtot=int(spectral.mtot),
        solve_kind=solve_kind,
        strategy=strategy.name,
        delta_cv=delta_cv,
        delta_rel_l2=delta_rel_l2,
        batch_size=int(rhs.shape[0]) if rhs.dim() == 2 else 1,
        outer_cg_iters=outer_cg_iters,
        outer_cg_ms=outer_cg_ms,
        total_ms=1000.0 * (time.perf_counter() - total_t0),
        precond_setup_ms=precond_setup_ms,
        precond_ms=precond_ms,
        precond_calls=precond_calls,
        precond_iters_total=precond_iters_total,
        rel_res=rel_res,
        num_bins=int(strategy.num_bins),
        lengthscale=lengthscale,
        variance=variance,
    )


def run_search(
    *,
    n_values: list[int],
    d: int,
    n_e_probes: int,
    n_m_probes: int,
    burnin_iters: int,
    lengthscale: float,
    variance: float,
    total_count: float,
    spectral_eps: float,
    trunc_eps: float,
    nufft_eps: float,
    cg_tol: float,
    seed: int,
) -> list[Record]:
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cpu")
    rdtype = torch.float64
    cdtype = torch.complex128

    likelihood = _PGNegativeBinomialLikelihood(total_count=total_count)
    strategies = _default_strategies()
    records: list[Record] = []

    for idx, n in enumerate(n_values):
        data_seed = seed + 1000 * idx
        X, y = _generate_nb_data(
            n=n,
            d=d,
            seed=data_seed,
            lengthscale=lengthscale,
            variance=variance,
            total_count=total_count,
            spectral_eps=spectral_eps,
            trunc_eps=trunc_eps,
            nufft_eps=nufft_eps,
        )
        X = X.to(device=device, dtype=rdtype)
        y = y.to(device=device, dtype=rdtype)

        kernel = _make_kernel(
            "squared_exponential",
            dimension=d,
            lengthscale=lengthscale,
            variance=variance,
        )
        spectral = _build_spectral_state(
            X,
            kernel,
            spectral_eps=spectral_eps,
            trunc_eps=trunc_eps,
            nufft_eps=nufft_eps,
            rdtype=rdtype,
            cdtype=cdtype,
            device=device,
        )

        kappa = likelihood.kappa(y)
        pg_b = likelihood.pg_b(y)
        variational = _VariationalState(delta=0.25 * pg_b.clone())
        variational, _ = _run_estep(
            y,
            kappa,
            pg_b,
            likelihood,
            variational,
            spectral,
            max_iters=burnin_iters,
            rho0=0.7,
            gamma=1e-3,
            tol=1e-6,
            n_probes=n_e_probes,
            cg_tol=cg_tol,
            reuse_probes=True,
            use_toeplitz_warm_start=False,
            seed=data_seed + 17,
            verbose=0,
            use_toeplitz_preconditioner=True,
        )

        delta_cv, delta_rel_l2 = _delta_summary(variational.delta)

        e_probes = _sample_rademacher(
            (n_e_probes, n),
            device=device,
            dtype=rdtype,
            seed=data_seed + 23,
        ) if n_e_probes > 0 else torch.empty((0, n), device=device, dtype=rdtype)
        Z = torch.cat([kappa[None, :], e_probes], dim=0)
        Delta = variational.delta.to(dtype=spectral.ws.dtype).view(1, -1)
        z_feat = spectral.fadj_batched(Z.to(dtype=spectral.ws.dtype))
        Kz = spectral.fwd_batched(spectral.ws2 * z_feat)
        rhs_e = spectral.ws * spectral.fadj_batched(Delta * Kz)

        m_probes = _sample_rademacher(
            (n_m_probes, n),
            device=device,
            dtype=rdtype,
            seed=data_seed + 29,
        ).to(dtype=spectral.ws.dtype)
        Q_block = spectral.fadj_batched(m_probes) if n_m_probes > 0 else torch.empty(
            (0, spectral.xis.shape[0]), device=device, dtype=cdtype
        )
        q_y = spectral.fadj_batched(kappa.to(dtype=spectral.ws.dtype).unsqueeze(0))
        rhs_m = torch.cat([Q_block, q_y], dim=0) * spectral.ws[None, :]

        for strategy in strategies:
            records.append(
                _benchmark_solve(
                    solve_kind="estep",
                    spectral=spectral,
                    X=X,
                    delta=variational.delta,
                    rhs=rhs_e,
                    strategy=strategy,
                    cg_tol=cg_tol,
                    delta_cv=delta_cv,
                    delta_rel_l2=delta_rel_l2,
                    lengthscale=lengthscale,
                    variance=variance,
                )
            )
            records.append(
                _benchmark_solve(
                    solve_kind="mstep",
                    spectral=spectral,
                    X=X,
                    delta=variational.delta,
                    rhs=rhs_m,
                    strategy=strategy,
                    cg_tol=cg_tol,
                    delta_cv=delta_cv,
                    delta_rel_l2=delta_rel_l2,
                    lengthscale=lengthscale,
                    variance=variance,
                )
            )

    return records


def _write_csv(path: Path, rows: list[Record]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> None:
    parser = argparse.ArgumentParser(description="Search structured Toeplitz-based preconditioners for fixed-hyperparameter training solves.")
    parser.add_argument("--n-values", nargs="+", type=int, default=[5000, 10000, 50000])
    parser.add_argument("--d", type=int, default=2)
    parser.add_argument("--n-e-probes", type=int, default=2)
    parser.add_argument("--n-m-probes", type=int, default=2)
    parser.add_argument("--burnin-iters", type=int, default=3)
    parser.add_argument("--lengthscale", type=float, default=0.35)
    parser.add_argument("--variance", type=float, default=1.0)
    parser.add_argument("--total-count", type=float, default=3.0)
    parser.add_argument("--spectral-eps", type=float, default=1e-4)
    parser.add_argument("--trunc-eps", type=float, default=1e-4)
    parser.add_argument("--nufft-eps", type=float, default=1e-7)
    parser.add_argument("--cg-tol", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=Path, default=Path("structured_preconditioning_search_results.csv"))
    args = parser.parse_args()

    rows = run_search(
        n_values=args.n_values,
        d=args.d,
        n_e_probes=args.n_e_probes,
        n_m_probes=args.n_m_probes,
        burnin_iters=args.burnin_iters,
        lengthscale=args.lengthscale,
        variance=args.variance,
        total_count=args.total_count,
        spectral_eps=args.spectral_eps,
        trunc_eps=args.trunc_eps,
        nufft_eps=args.nufft_eps,
        cg_tol=args.cg_tol,
        seed=args.seed,
    )
    _write_csv(args.csv, rows)
    print(f"wrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
