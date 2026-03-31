from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.distributions import NegativeBinomial
from torch.optim import Adam

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from vanilla_gp_sampling import sample_gp_spectral_approx

from pg_classifier import (
    ConjugateGradients,
    _PGNegativeBinomialLikelihood,
    _VariationalState,
    _build_spectral_state,
    _compute_mstep_gradient,
    _make_kernel,
    _run_estep,
    _sample_rademacher,
)


@dataclass(frozen=True)
class SolverStrategy:
    name: str
    use_warm_start: bool = False
    preconditioner: str = "none"
    warm_tol: float = 1e-2
    precond_tol: float = 1e-2
    warm_max_iter: int = 5000
    precond_max_iter: int = 500


@dataclass
class SolveStats:
    outer: int
    solve_kind: str
    strategy: str
    delta_cv: float
    delta_rel_l2: float
    batch_size: int
    outer_cg_iters: int
    outer_cg_ms: float
    total_ms: float
    fixed_ms: float
    warmstart_ms: float
    warmstart_iters: int
    precond_ms: float
    precond_calls: int
    precond_iters_total: int
    rel_res: float
    lengthscale: float
    variance: float


def _clone_variational_state(state: _VariationalState) -> _VariationalState:
    return _VariationalState(
        delta=state.delta.clone(),
        mean=None if state.mean is None else state.mean.clone(),
        sigma_diag=None if state.sigma_diag is None else state.sigma_diag.clone(),
        probes=None if state.probes is None else state.probes.clone(),
    )


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
    true_length_scale: float,
    true_variance: float,
    true_total_count: float,
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
        length_scale=true_length_scale,
        variance=true_variance,
        spectral_eps=spectral_eps,
        trunc_eps=trunc_eps,
        nufft_eps=nufft_eps,
        seed=seed + 19,
    )
    y = NegativeBinomial(
        total_count=torch.tensor(true_total_count, dtype=torch.float64),
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


def _benchmark_sigma_solve(
    spectral,
    delta: torch.Tensor,
    z: torch.Tensor,
    *,
    cg_tol: float,
    strategy: SolverStrategy,
    outer: int,
    delta_cv: float,
    delta_rel_l2: float,
    lengthscale: float,
    variance: float,
) -> SolveStats:
    delta_complex = delta.to(dtype=spectral.ws.dtype, device=delta.device)
    vector_input = z.dim() == 1
    if vector_input:
        z = z.unsqueeze(0)
    z = z.to(dtype=spectral.ws.dtype)

    total_t0 = time.perf_counter()
    fixed_ms = 0.0
    warmstart_ms = 0.0
    warmstart_iters = 0
    precond_ms = 0.0
    precond_calls = 0
    precond_iters_total = 0

    fixed_t0 = time.perf_counter()
    Delta = delta_complex.view(1, -1)
    z_feat = spectral.fadj_batched(z)
    Kz = spectral.fwd_batched(spectral.ws2 * z_feat)
    rhs = spectral.ws * spectral.fadj_batched(Delta * Kz)
    fixed_ms += 1000.0 * (time.perf_counter() - fixed_t0)

    def A_feat(u: torch.Tensor) -> torch.Tensor:
        psi_u = spectral.fwd_batched(spectral.ws * u)
        return u + spectral.ws * spectral.fadj_batched(Delta * psi_u)

    delta_bar = delta_complex.mean()

    def A_tilde(u: torch.Tensor) -> torch.Tensor:
        if u.dim() == 1:
            return u + delta_bar * (spectral.ws * spectral.toeplitz(spectral.ws * u))
        rows = [spectral.ws * spectral.toeplitz(spectral.ws * u[b]) for b in range(u.shape[0])]
        return u + delta_bar * torch.stack(rows, dim=0)

    x0 = torch.zeros_like(rhs)
    if strategy.use_warm_start:
        warm_t0 = time.perf_counter()
        warm, warmstart_iters = _solve_with_inner_cg(
            A_tilde,
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
                A_tilde,
                v,
                tol=strategy.precond_tol,
                max_iter=strategy.precond_max_iter,
            )
            precond_ms += 1000.0 * (time.perf_counter() - t0)
            precond_iters_total += inner_iters
            return soln

        M_inv_apply = toeplitz_precond

    outer_t0 = time.perf_counter()
    cg = ConjugateGradients(
        A_feat,
        rhs,
        x0=x0,
        tol=cg_tol,
        max_iter=2000,
        early_stopping=True,
        M_inv_apply=M_inv_apply,
    )
    x = cg.solve()
    outer_cg_ms = 1000.0 * (time.perf_counter() - outer_t0)
    outer_cg_iters = int(cg.iters_completed)

    fixed_t1 = time.perf_counter()
    out = spectral.fwd_batched(spectral.ws * x)
    result = (Kz - out).real
    fixed_ms += 1000.0 * (time.perf_counter() - fixed_t1)
    if vector_input:
        result = result.squeeze(0)

    residual = rhs - A_feat(x)
    rel_res = float(
        (
            torch.linalg.norm(residual.reshape(-1)).real
            / max(float(torch.linalg.norm(rhs.reshape(-1)).real.item()), 1e-16)
        ).item()
    )

    return SolveStats(
        outer=outer,
        solve_kind="estep",
        strategy=strategy.name,
        delta_cv=delta_cv,
        delta_rel_l2=delta_rel_l2,
        batch_size=int(z.shape[0]),
        outer_cg_iters=outer_cg_iters,
        outer_cg_ms=outer_cg_ms,
        total_ms=1000.0 * (time.perf_counter() - total_t0),
        fixed_ms=fixed_ms,
        warmstart_ms=warmstart_ms,
        warmstart_iters=warmstart_iters,
        precond_ms=precond_ms,
        precond_calls=precond_calls,
        precond_iters_total=precond_iters_total,
        rel_res=rel_res,
        lengthscale=lengthscale,
        variance=variance,
    )


def _benchmark_feature_space_solve(
    spectral,
    delta: torch.Tensor,
    q_all: torch.Tensor,
    *,
    cg_tol: float,
    strategy: SolverStrategy,
    outer: int,
    delta_cv: float,
    delta_rel_l2: float,
    lengthscale: float,
    variance: float,
) -> SolveStats:
    omega = delta.to(dtype=spectral.ws.dtype, device=delta.device).flatten()
    D2_real = spectral.ws2.real
    eps_d = max(float(D2_real.mean()) * 1e-14, 1e-14)
    Ds = torch.sqrt(torch.clamp(D2_real, min=eps_d)).to(dtype=spectral.ws.dtype)
    rhs = q_all * Ds[None, :]
    wbar = omega.real.mean().to(dtype=spectral.ws.dtype)

    fixed_ms = 0.0
    warmstart_ms = 0.0
    warmstart_iters = 0
    precond_ms = 0.0
    precond_calls = 0
    precond_iters_total = 0
    total_t0 = time.perf_counter()

    def apply_omega(v: torch.Tensor) -> torch.Tensor:
        if v.dim() == 2:
            return omega[:, None] * v
        return omega * v

    def apply_S(Y: torch.Tensor) -> torch.Tensor:
        if Y.dim() == 1:
            t = Ds * Y
            u = spectral.fwd(t)
            u = apply_omega(u)
            v = spectral.fadj(u)
            return Ds * v
        t = Y * Ds[None, :]
        u = spectral.fwd_batched(t).mT
        u = apply_omega(u)
        v = spectral.fadj_batched(u.T).T
        return (Ds[:, None] * v).T

    def apply_IpS(Y: torch.Tensor) -> torch.Tensor:
        return Y + apply_S(Y)

    def apply_IpS_toeplitz(Y: torch.Tensor) -> torch.Tensor:
        if Y.dim() == 1:
            t = Ds * Y
            return Y + wbar * (Ds * spectral.toeplitz(t))
        t = Y * Ds[None, :]
        rows = [spectral.toeplitz(t[b]) for b in range(t.shape[0])]
        Tt = torch.stack(rows, dim=0)
        return Y + wbar * (Ds[None, :] * Tt)

    diagB = (omega.real.sum() * torch.ones_like(D2_real)).to(dtype=D2_real.dtype)
    diag_IpS = (1.0 + diagB * D2_real).to(dtype=spectral.ws.dtype)

    x0 = torch.zeros_like(rhs)
    if strategy.use_warm_start:
        warm_t0 = time.perf_counter()
        warm, warmstart_iters = _solve_with_inner_cg(
            apply_IpS_toeplitz,
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
                apply_IpS_toeplitz,
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
            return v / diag_IpS if v.dim() == 1 else v / diag_IpS[None, :]

        M_inv_apply = jacobi_precond

    outer_t0 = time.perf_counter()
    cg = ConjugateGradients(
        apply_IpS,
        rhs,
        x0=x0,
        tol=cg_tol,
        max_iter=2000,
        early_stopping=True,
        M_inv_apply=M_inv_apply,
    )
    y = cg.solve()
    outer_cg_ms = 1000.0 * (time.perf_counter() - outer_t0)
    outer_cg_iters = int(cg.iters_completed)

    residual = rhs - apply_IpS(y)
    rel_res = float(
        (
            torch.linalg.norm(residual.reshape(-1)).real
            / max(float(torch.linalg.norm(rhs.reshape(-1)).real.item()), 1e-16)
        ).item()
    )

    return SolveStats(
        outer=outer,
        solve_kind="mstep",
        strategy=strategy.name,
        delta_cv=delta_cv,
        delta_rel_l2=delta_rel_l2,
        batch_size=int(q_all.shape[0]),
        outer_cg_iters=outer_cg_iters,
        outer_cg_ms=outer_cg_ms,
        total_ms=1000.0 * (time.perf_counter() - total_t0),
        fixed_ms=fixed_ms,
        warmstart_ms=warmstart_ms,
        warmstart_iters=warmstart_iters,
        precond_ms=precond_ms,
        precond_calls=precond_calls,
        precond_iters_total=precond_iters_total,
        rel_res=rel_res,
        lengthscale=lengthscale,
        variance=variance,
    )


def _default_estep_strategies() -> list[SolverStrategy]:
    return [
        SolverStrategy(name="none"),
        SolverStrategy(name="warm_start", use_warm_start=True, warm_tol=1e-3),
        SolverStrategy(
            name="toeplitz_pcg",
            preconditioner="toeplitz",
            precond_tol=5e-2,
            precond_max_iter=200,
        ),
        SolverStrategy(
            name="toeplitz_pcg_warm",
            use_warm_start=True,
            preconditioner="toeplitz",
            warm_tol=1e-3,
            precond_tol=5e-2,
            precond_max_iter=200,
        ),
    ]


def _default_mstep_strategies() -> list[SolverStrategy]:
    return [
        SolverStrategy(name="none"),
        SolverStrategy(name="warm_start", use_warm_start=True, warm_tol=1e-2),
        SolverStrategy(name="jacobi", preconditioner="jacobi"),
        SolverStrategy(name="jacobi_warm", use_warm_start=True, preconditioner="jacobi", warm_tol=1e-2),
        SolverStrategy(
            name="toeplitz_pcg",
            preconditioner="toeplitz",
            precond_tol=1e-2,
            precond_max_iter=200,
        ),
        SolverStrategy(
            name="toeplitz_pcg_warm",
            use_warm_start=True,
            preconditioner="toeplitz",
            warm_tol=1e-2,
            precond_tol=1e-2,
            precond_max_iter=200,
        ),
    ]


def run_benchmark(
    *,
    n: int,
    d: int,
    outer_iters: int,
    n_e_probes: int,
    n_m_probes: int,
    true_length_scale: float,
    true_variance: float,
    true_total_count: float,
    init_lengthscale: float,
    init_variance: float,
    lr: float,
    spectral_eps: float,
    trunc_eps: float,
    nufft_eps: float,
    cg_tol: float,
    seed: int,
) -> list[SolveStats]:
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cpu")
    rdtype = torch.float64
    cdtype = torch.complex128

    X, y = _generate_nb_data(
        n=n,
        d=d,
        seed=seed,
        true_length_scale=true_length_scale,
        true_variance=true_variance,
        true_total_count=true_total_count,
        spectral_eps=spectral_eps,
        trunc_eps=trunc_eps,
        nufft_eps=nufft_eps,
    )
    X = X.to(device=device, dtype=rdtype)
    y = y.to(device=device, dtype=rdtype)

    likelihood = _PGNegativeBinomialLikelihood(total_count=true_total_count)
    kappa = likelihood.kappa(y)
    pg_b = likelihood.pg_b(y)

    kernel = _make_kernel(
        "squared_exponential",
        dimension=d,
        lengthscale=init_lengthscale,
        variance=init_variance,
    )
    optimizer = Adam(kernel._gp_params_ref.parameters(), lr=lr, maximize=True)

    variational = _VariationalState(delta=0.25 * pg_b.clone())
    estep_strategies = _default_estep_strategies()
    mstep_strategies = _default_mstep_strategies()
    rows: list[SolveStats] = []

    for outer in range(outer_iters):
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

        delta_cv_e, delta_rel_l2_e = _delta_summary(variational.delta)
        e_probe_seed = seed + 1000 * outer + 17
        e_probes = _sample_rademacher(
            (n_e_probes, y.numel()),
            device=device,
            dtype=rdtype,
            seed=e_probe_seed,
        ) if n_e_probes > 0 else torch.empty((0, y.numel()), device=device, dtype=rdtype)
        Z = torch.cat([kappa[None, :], e_probes], dim=0)

        for strategy in estep_strategies:
            rows.append(
                _benchmark_sigma_solve(
                    spectral,
                    variational.delta,
                    Z,
                    cg_tol=cg_tol,
                    strategy=strategy,
                    outer=outer,
                    delta_cv=delta_cv_e,
                    delta_rel_l2=delta_rel_l2_e,
                    lengthscale=float(kernel.lengthscale),
                    variance=float(kernel.variance),
                )
            )

        warm_state = _clone_variational_state(variational)
        warm_state.probes = e_probes.clone() if n_e_probes > 0 else None
        warm_state, _ = _run_estep(
            y,
            kappa,
            pg_b,
            likelihood,
            warm_state,
            spectral,
            max_iters=1,
            rho0=0.7,
            gamma=1e-3,
            tol=1e-6,
            n_probes=n_e_probes,
            cg_tol=cg_tol,
            reuse_probes=True,
            use_toeplitz_warm_start=True,
            seed=seed + 1000 * outer,
            verbose=0,
        )

        delta_cv_m, delta_rel_l2_m = _delta_summary(warm_state.delta)
        m_probe_seed = seed + 10_000 + 1000 * outer
        m_probes = _sample_rademacher(
            (n_m_probes, kappa.numel()),
            device=device,
            dtype=rdtype,
            seed=m_probe_seed,
        ).to(dtype=spectral.ws.dtype)
        Q_block = spectral.fadj_batched(m_probes)
        q_y = spectral.fadj_batched(kappa.to(dtype=spectral.ws.dtype).unsqueeze(0))
        Q_all = torch.cat([Q_block, q_y], dim=0)

        for strategy in mstep_strategies:
            rows.append(
                _benchmark_feature_space_solve(
                    spectral,
                    warm_state.delta,
                    Q_all,
                    cg_tol=cg_tol,
                    strategy=strategy,
                    outer=outer,
                    delta_cv=delta_cv_m,
                    delta_rel_l2=delta_rel_l2_m,
                    lengthscale=float(kernel.lengthscale),
                    variance=float(kernel.variance),
                )
            )

        mstep_warm = _compute_mstep_gradient(
            kappa,
            warm_state.delta,
            spectral,
            n_probes=n_m_probes,
            cg_tol=cg_tol,
            use_toeplitz_warm_start=True,
            seed=seed + 1000 * outer,
        )

        grad = mstep_warm["grad"].real
        raw = kernel._gp_params_ref.raw
        raw.grad = torch.stack(
            [
                grad[0].to(dtype=raw.dtype, device=raw.device) * kernel.lengthscale,
                grad[1].to(dtype=raw.dtype, device=raw.device) * kernel.variance,
                torch.tensor(0.0, dtype=raw.dtype, device=raw.device),
            ]
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        variational = warm_state

    return rows


def _write_csv(rows: list[SolveStats], path: Path) -> None:
    fieldnames = list(SolveStats.__annotations__.keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _summarize(rows: list[SolveStats]) -> str:
    lines = [
        "Preconditioning benchmark on synthetic negative-binomial GP data",
        "",
    ]

    for solve_kind in ("estep", "mstep"):
        sub = [r for r in rows if r.solve_kind == solve_kind]
        by_strategy = {}
        for row in sub:
            by_strategy.setdefault(row.strategy, []).append(row)

        baseline = by_strategy["none"]
        base_iters = float(np.mean([r.outer_cg_iters for r in baseline]))
        base_total_ms = float(np.mean([r.total_ms for r in baseline]))
        lines.append(f"{solve_kind.upper()} summary")
        lines.append(
            "strategy               outer_cg   solve_ms   warm_ms   precond_ms   precond_calls   toeplitz_iters   rel_res"
        )
        lines.append(
            "---------------------  ---------  ---------  --------  -----------  -------------  --------------  --------"
        )
        for name in sorted(by_strategy):
            group = by_strategy[name]
            outer_cg = float(np.mean([r.outer_cg_iters for r in group]))
            total_ms = float(np.mean([r.total_ms for r in group]))
            warm_ms = float(np.mean([r.warmstart_ms for r in group]))
            precond_ms = float(np.mean([r.precond_ms for r in group]))
            precond_calls = float(np.mean([r.precond_calls for r in group]))
            toeplitz_iters = float(np.mean([r.warmstart_iters + r.precond_iters_total for r in group]))
            rel_res = float(np.mean([r.rel_res for r in group]))
            iter_delta = 100.0 * (1.0 - outer_cg / base_iters)
            time_delta = 100.0 * (1.0 - total_ms / base_total_ms)
            lines.append(
                f"{name:<21}  {outer_cg:>9.1f}  {total_ms:>9.1f}  {warm_ms:>8.1f}  {precond_ms:>11.1f}  "
                f"{precond_calls:>13.1f}  {toeplitz_iters:>14.1f}  {rel_res:>8.2e}"
            )
            lines.append(
                f"  vs none: outer-CG reduction {iter_delta:+.1f}% | total-time reduction {time_delta:+.1f}%"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark warm start vs true preconditioning for PG solves.")
    parser.add_argument("--n", type=int, default=1200)
    parser.add_argument("--d", type=int, default=2)
    parser.add_argument("--outer-iters", type=int, default=4)
    parser.add_argument("--n-e-probes", type=int, default=4)
    parser.add_argument("--n-m-probes", type=int, default=4)
    parser.add_argument("--true-lengthscale", type=float, default=0.6)
    parser.add_argument("--true-variance", type=float, default=1.0)
    parser.add_argument("--true-total-count", type=float, default=2.5)
    parser.add_argument("--init-lengthscale", type=float, default=0.3)
    parser.add_argument("--init-variance", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--spectral-eps", type=float, default=1e-4)
    parser.add_argument("--trunc-eps", type=float, default=1e-4)
    parser.add_argument("--nufft-eps", type=float, default=1e-7)
    parser.add_argument("--cg-tol", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    rows = run_benchmark(
        n=args.n,
        d=args.d,
        outer_iters=args.outer_iters,
        n_e_probes=args.n_e_probes,
        n_m_probes=args.n_m_probes,
        true_length_scale=args.true_lengthscale,
        true_variance=args.true_variance,
        true_total_count=args.true_total_count,
        init_lengthscale=args.init_lengthscale,
        init_variance=args.init_variance,
        lr=args.lr,
        spectral_eps=args.spectral_eps,
        trunc_eps=args.trunc_eps,
        nufft_eps=args.nufft_eps,
        cg_tol=args.cg_tol,
        seed=args.seed,
    )
    if args.csv is not None:
        _write_csv(rows, args.csv)
    print(_summarize(rows))


if __name__ == "__main__":
    main()
