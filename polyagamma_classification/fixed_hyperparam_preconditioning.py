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
from torch.distributions import NegativeBinomial

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from vanilla_gp_sampling import sample_gp_spectral_approx

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
    preconditioner: str = "none"
    use_warm_start: bool = False
    warm_tol: float = 1e-3
    warm_max_iter: int = 5_000
    precond_tol: float = 1e-2
    precond_max_iter: int = 200
    sample_multiplier: float = 0.0
    max_sample_size: int | None = None


@dataclass
class SolveRecord:
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
    setup_ms: float
    warmstart_ms: float
    warmstart_iters: int
    precond_ms: float
    precond_setup_ms: float
    precond_calls: int
    precond_iters_total: int
    rel_res: float
    sample_size: int
    lengthscale: float
    variance: float


def _delta_summary(delta: torch.Tensor) -> tuple[float, float]:
    delta_real = delta.detach().to(dtype=torch.float64)
    mean = float(delta_real.mean().item())
    std = float(delta_real.std(unbiased=False).item())
    cv = std / max(mean, 1e-12)
    rel_l2 = float((torch.linalg.norm(delta_real - mean) / torch.linalg.norm(delta_real)).item())
    return cv, rel_l2


def _generate_nb_data(
    *,
    n: int,
    d: int,
    seed: int,
    lengthscale: float,
    variance: float,
    total_count: float,
    spectral_eps: float,
    trunc_eps: float,
    nufft_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    X = torch.rand(n, d, dtype=torch.float64) * 2.0 - 1.0
    latent = sample_gp_spectral_approx(
        X,
        num_samples=1,
        length_scale=lengthscale,
        variance=variance,
        spectral_eps=spectral_eps,
        trunc_eps=trunc_eps,
        nufft_eps=nufft_eps,
        seed=seed + 19,
    )
    y = NegativeBinomial(
        total_count=torch.tensor(total_count, dtype=torch.float64),
        logits=latent,
    ).sample()
    return X, y.to(dtype=torch.float64)


def _solve_with_inner_cg(
    A_apply: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    *,
    tol: float,
    max_iter: int,
) -> tuple[torch.Tensor, int]:
    inner = ConjugateGradients(
        A_apply,
        rhs,
        x0=torch.zeros_like(rhs),
        tol=tol,
        max_iter=max_iter,
        early_stopping=True,
    )
    soln = inner.solve()
    return soln, int(inner.iters_completed)


def _feature_operator_parts(spectral, delta: torch.Tensor):
    omega = delta.to(dtype=spectral.ws.dtype, device=delta.device).flatten()
    D2_real = spectral.ws2.real
    eps_d = max(float(D2_real.mean()) * 1e-14, 1e-14)
    Ds = torch.sqrt(torch.clamp(D2_real, min=eps_d)).to(dtype=spectral.ws.dtype)
    wbar = omega.real.mean().to(dtype=spectral.ws.dtype)
    diag_IpS = (1.0 + omega.real.sum() * D2_real).to(dtype=spectral.ws.dtype)

    def apply_omega(v: torch.Tensor) -> torch.Tensor:
        if v.dim() == 2:
            return omega[:, None] * v
        return omega * v

    def apply_A(Y: torch.Tensor) -> torch.Tensor:
        if Y.dim() == 1:
            t = Ds * Y
            u = spectral.fwd(t)
            u = apply_omega(u)
            v = spectral.fadj(u)
            return Y + Ds * v
        t = Y * Ds[None, :]
        u = spectral.fwd_batched(t).mT
        u = apply_omega(u)
        v = spectral.fadj_batched(u.T).T
        return Y + (Ds[:, None] * v).T

    def apply_A_toeplitz(Y: torch.Tensor) -> torch.Tensor:
        if Y.dim() == 1:
            t = Ds * Y
            return Y + wbar * (Ds * spectral.toeplitz(t))
        t = Y * Ds[None, :]
        rows = [spectral.toeplitz(t[b]) for b in range(t.shape[0])]
        Tt = torch.stack(rows, dim=0)
        return Y + wbar * (Ds[None, :] * Tt)

    return Ds, apply_A, apply_A_toeplitz, diag_IpS


def _build_sampled_dense_preconditioner(
    X: torch.Tensor,
    delta: torch.Tensor,
    spectral,
    *,
    sample_size: int,
    seed: int,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], float, int]:
    n = X.shape[0]
    m = spectral.xis.shape[0]
    sample_size = max(1, min(sample_size, n))
    if sample_size == n:
        idx = torch.arange(n, device=X.device)
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        idx = torch.randperm(n, generator=generator)[:sample_size].to(device=X.device)

    t0 = time.perf_counter()
    Xs = X[idx]
    delta_s = delta[idx].to(dtype=spectral.ws.dtype)
    phases = 2.0j * math.pi * (Xs @ spectral.xis.T)
    Psi_s = torch.exp(phases).to(dtype=spectral.ws.dtype) * spectral.ws[None, :]
    scale = float(n) / float(sample_size)
    eye = torch.eye(m, dtype=spectral.ws.dtype, device=X.device)
    A_sub = eye + scale * (Psi_s.conj().T @ (delta_s[:, None] * Psi_s))
    A_sub = 0.5 * (A_sub + A_sub.conj().T)
    chol = torch.linalg.cholesky(A_sub + 1e-10 * eye)
    setup_ms = 1000.0 * (time.perf_counter() - t0)

    def apply(v: torch.Tensor) -> torch.Tensor:
        if v.dim() == 1:
            return torch.cholesky_solve(v[:, None], chol).squeeze(1)
        return torch.cholesky_solve(v.T.contiguous(), chol).T

    return apply, setup_ms, sample_size


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
    seed: int,
) -> SolveRecord:
    Ds, apply_A, apply_A_toeplitz, diag_IpS = _feature_operator_parts(spectral, delta)
    rhs = rhs.to(dtype=spectral.ws.dtype)

    warmstart_ms = 0.0
    warmstart_iters = 0
    precond_ms = 0.0
    precond_setup_ms = 0.0
    precond_calls = 0
    precond_iters_total = 0
    sample_size = 0
    total_t0 = time.perf_counter()
    x0 = torch.zeros_like(rhs)

    if strategy.use_warm_start:
        warm_t0 = time.perf_counter()
        warm, warmstart_iters = _solve_with_inner_cg(
            apply_A_toeplitz,
            rhs,
            tol=strategy.warm_tol,
            max_iter=strategy.warm_max_iter,
        )
        warmstart_ms = 1000.0 * (time.perf_counter() - warm_t0)
        if warmstart_iters < int(0.9 * strategy.warm_max_iter):
            x0 = warm

    M_inv_apply = None
    if strategy.preconditioner == "toeplitz":

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
    elif strategy.preconditioner == "jacobi":

        def jacobi_precond(v: torch.Tensor) -> torch.Tensor:
            if v.dim() == 1:
                return v / diag_IpS
            return v / diag_IpS[None, :]

        M_inv_apply = jacobi_precond
    elif strategy.preconditioner == "sampled_dense":
        target = int(math.ceil(strategy.sample_multiplier * spectral.xis.shape[0]))
        if strategy.max_sample_size is not None:
            target = min(target, strategy.max_sample_size)
        target = max(target, spectral.xis.shape[0])
        M_inv_apply, precond_setup_ms, sample_size = _build_sampled_dense_preconditioner(
            X,
            delta,
            spectral,
            sample_size=target,
            seed=seed,
        )

    outer_t0 = time.perf_counter()
    cg = ConjugateGradients(
        apply_A,
        rhs,
        x0=x0,
        tol=cg_tol,
        max_iter=2_000,
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

    return SolveRecord(
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
        setup_ms=0.0,
        warmstart_ms=warmstart_ms,
        warmstart_iters=warmstart_iters,
        precond_ms=precond_ms,
        precond_setup_ms=precond_setup_ms,
        precond_calls=precond_calls,
        precond_iters_total=precond_iters_total,
        rel_res=rel_res,
        sample_size=sample_size,
        lengthscale=lengthscale,
        variance=variance,
    )


def _default_strategies() -> list[Strategy]:
    return [
        Strategy(name="none"),
        Strategy(name="jacobi", preconditioner="jacobi"),
        Strategy(name="toeplitz_pcg", preconditioner="toeplitz", precond_tol=1e-2, precond_max_iter=200),
        Strategy(name="sampled_dense_2m", preconditioner="sampled_dense", sample_multiplier=2.0),
        Strategy(name="sampled_dense_4m", preconditioner="sampled_dense", sample_multiplier=4.0),
    ]


def run_fixed_hyperparam_benchmark(
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
) -> list[SolveRecord]:
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cpu")
    rdtype = torch.float64
    cdtype = torch.complex128

    likelihood = _PGNegativeBinomialLikelihood(total_count=total_count)
    strategies = _default_strategies()
    records: list[SolveRecord] = []

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
        Q_all = torch.cat([Q_block, q_y], dim=0)
        rhs_m = Q_all * spectral.ws[None, :]

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
                    seed=data_seed + 101,
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
                    seed=data_seed + 202,
                )
            )

    return records


def _write_csv(path: Path, rows: list[SolveRecord]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark fixed-hyperparameter training solves under different preconditioners.")
    parser.add_argument("--n-values", nargs="+", type=int, default=[500, 1000, 2000, 5000, 10000])
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
    parser.add_argument("--csv", type=Path, default=Path("fixed_hyperparam_preconditioning_results.csv"))
    args = parser.parse_args()

    rows = run_fixed_hyperparam_benchmark(
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
