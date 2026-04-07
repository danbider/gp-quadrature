from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from cg import ConjugateGradients
from fixed_hyperparam_preconditioning import _generate_nb_data
from pg_classifier import (
    _PGNegativeBinomialLikelihood,
    _VariationalState,
    _build_spectral_state,
    _build_weighted_toeplitz,
    _make_kernel,
    _pg_omega_expectation,
    _sample_rademacher,
)


@dataclass
class IterRecord:
    n: int
    seed: int
    strategy: str
    outer_iter: int
    e_cg_iters: int
    m_cg_iters: int
    e_ms: float
    m_ms: float
    lengthscale: float
    variance: float
    residual: float
    approx_mae: float


@dataclass
class SummaryRecord:
    n: int
    seed: int
    strategy: str
    outer_iters: int
    mean_e_cg_iters: float
    mean_m_cg_iters: float
    total_e_ms: float
    total_m_ms: float
    total_ms: float
    final_lengthscale: float
    final_variance: float
    final_mae: float


def _make_weighted_sigma_apply(
    spectral,
    delta: torch.Tensor,
    *,
    cg_tol: float,
    use_jacobi: bool,
):
    info = {"cg_iters": 0}
    delta_complex = delta.to(dtype=spectral.ws.dtype, device=delta.device)
    weighted_toeplitz = _build_weighted_toeplitz(delta_complex, spectral)
    diag = (1.0 + delta_complex.real.sum() * spectral.ws2.real).to(
        dtype=spectral.ws.dtype,
        device=delta.device,
    )

    def sigma_apply(z: torch.Tensor) -> torch.Tensor:
        vector_input = z.dim() == 1
        if vector_input:
            z = z.unsqueeze(0)
        z = z.to(dtype=spectral.ws.dtype)
        rhs = spectral.ws * spectral.fadj_batched(z)

        def A_feat(u: torch.Tensor) -> torch.Tensor:
            if u.dim() == 1:
                t = spectral.ws * u
                return u + spectral.ws * weighted_toeplitz(t)
            t = u * spectral.ws[None, :]
            return u + spectral.ws[None, :] * weighted_toeplitz(t)

        M_inv_apply = None
        if use_jacobi:
            def jacobi(v: torch.Tensor) -> torch.Tensor:
                if v.dim() == 1:
                    return v / diag
                return v / diag[None, :]

            M_inv_apply = jacobi

        cg = ConjugateGradients(
            A_feat,
            rhs,
            x0=torch.zeros_like(rhs),
            tol=cg_tol,
            max_iter=2000,
            early_stopping=True,
            M_inv_apply=M_inv_apply,
        )
        x = cg.solve()
        info["cg_iters"] = int(cg.iters_completed)
        result = spectral.fwd_batched(spectral.ws * x).real
        return result.squeeze(0) if vector_input else result

    return sigma_apply, info


def _make_weighted_feature_solver(
    delta: torch.Tensor,
    spectral,
    *,
    cg_tol: float,
    use_jacobi: bool,
):
    omega = delta.to(dtype=spectral.ws.dtype, device=delta.device).flatten()
    D2_real = spectral.ws2.real
    eps_d = max(float(D2_real.mean()) * 1e-14, 1e-14)
    Ds = torch.sqrt(torch.clamp(D2_real, min=eps_d)).to(dtype=spectral.ws.dtype)
    Dsinv = 1.0 / Ds
    info = {"cg_iters": 0}
    weighted_toeplitz = _build_weighted_toeplitz(omega, spectral)
    diag = (1.0 + omega.real.sum() * D2_real).to(dtype=spectral.ws.dtype, device=delta.device)

    def apply_omega(v: torch.Tensor) -> torch.Tensor:
        if v.dim() == 2:
            return omega[:, None] * v
        return omega * v

    def apply_S(Y: torch.Tensor) -> torch.Tensor:
        if Y.dim() == 1:
            t = Ds * Y
            return Ds * weighted_toeplitz(t)
        t = Y * Ds[None, :]
        return Ds[None, :] * weighted_toeplitz(t)

    def apply_IpS(Y: torch.Tensor) -> torch.Tensor:
        return Y + apply_S(Y)

    def solve_A_beta(q: torch.Tensor):
        rhs = Ds * q if q.dim() == 1 else q * Ds[None, :]
        M_inv_apply = None
        if use_jacobi:
            def jacobi(v: torch.Tensor) -> torch.Tensor:
                if v.dim() == 1:
                    return v / diag
                return v / diag[None, :]

            M_inv_apply = jacobi

        cg = ConjugateGradients(
            apply_IpS,
            rhs,
            x0=torch.zeros_like(rhs),
            tol=cg_tol,
            max_iter=2000,
            early_stopping=True,
            M_inv_apply=M_inv_apply,
        )
        y = cg.solve()
        info["cg_iters"] = int(cg.iters_completed)
        beta = Dsinv * y if q.dim() == 1 else y * Dsinv[None, :]
        return beta, int(cg.iters_completed)

    return solve_A_beta, apply_omega, info


def _run_estep_with_strategy(
    y: torch.Tensor,
    kappa: torch.Tensor,
    pg_b: torch.Tensor,
    likelihood: _PGNegativeBinomialLikelihood,
    variational: _VariationalState,
    spectral,
    *,
    n_probes: int,
    cg_tol: float,
    seed: int | None,
    use_jacobi: bool,
    rho0: float,
):
    sigma_apply, sigma_info = _make_weighted_sigma_apply(
        spectral,
        variational.delta,
        cg_tol=cg_tol,
        use_jacobi=use_jacobi,
    )
    probes = variational.probes
    if n_probes > 0 and (probes is None or probes.shape[0] != n_probes):
        probes = _sample_rademacher(
            (n_probes, y.numel()),
            device=y.device,
            dtype=y.dtype,
            seed=seed,
        )
    if n_probes > 0:
        Z = torch.cat([kappa[None, :], probes], dim=0)
    else:
        Z = kappa[None, :]

    S_all = sigma_apply(Z)
    mean = S_all[0]
    Sz = S_all[1:] if n_probes > 0 else torch.empty((0, y.numel()), device=y.device, dtype=y.dtype)
    sigma_diag = (probes * Sz).mean(dim=0) if n_probes > 0 else torch.zeros_like(mean)

    c2 = (sigma_diag + mean.pow(2)).clamp_min(1e-12)
    c = torch.sqrt(c2)
    Lambda = _pg_omega_expectation(c, pg_b)
    variational.delta.mul_(1.0 - rho0).add_(rho0 * Lambda)
    variational.delta.clamp_(min=0.0)
    variational.mean = mean
    variational.sigma_diag = sigma_diag
    variational.probes = probes
    return variational, {
        "residual": float((variational.delta - Lambda).abs().max().item()),
        "metric": likelihood.fit_metric(mean, sigma_diag, y),
        "cg_iters": sigma_info["cg_iters"],
    }


def _compute_mstep_gradient_with_strategy(
    kappa: torch.Tensor,
    delta: torch.Tensor,
    spectral,
    *,
    n_probes: int,
    cg_tol: float,
    seed: int | None,
    use_jacobi: bool,
):
    solve_A_beta, apply_omega, solve_info = _make_weighted_feature_solver(
        delta,
        spectral,
        cg_tol=cg_tol,
        use_jacobi=use_jacobi,
    )

    probes = _sample_rademacher(
        (n_probes, kappa.numel()),
        device=kappa.device,
        dtype=kappa.dtype,
        seed=seed,
    ).to(dtype=spectral.ws.dtype)
    Q_block = spectral.fadj_batched(probes)
    q_y = spectral.fadj_batched(kappa.to(dtype=spectral.ws.dtype).unsqueeze(0))
    Q_all = torch.cat([Q_block, q_y], dim=0)
    beta_all, cg_iters = solve_A_beta(Q_all)
    beta_probes = beta_all[:-1, :]
    beta_x = beta_all[-1, :]

    Rfeat = spectral.fadj_batched(apply_omega(probes.mT).T).T
    X = Rfeat.conj() * beta_probes.T
    vals = (X.mT @ spectral.Dprime).real
    term2 = vals.mean(dim=0)

    abs2 = (beta_x.conj() * beta_x).real
    term1 = spectral.Dprime.real.T @ abs2
    grad = 0.5 * (term1 - term2)
    return {
        "grad": grad,
        "cg_iters": cg_iters,
        "beta_mean": beta_x,
        "term1": term1,
        "term2": term2,
    }


def _run_case(
    *,
    n: int,
    seed: int,
    outer_iters: int,
    n_e_probes: int,
    n_m_probes: int,
    cg_tol: float,
    lr: float,
    lengthscale_init: float,
    variance_init: float,
    true_lengthscale: float,
    true_variance: float,
    total_count: float,
    rho0: float,
    spectral_eps: float,
    trunc_eps: float,
    nufft_eps: float,
    use_jacobi: bool,
) -> tuple[list[IterRecord], SummaryRecord]:
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    np.random.seed(seed)

    X, y = _generate_nb_data(
        n=n,
        d=2,
        seed=seed,
        lengthscale=true_lengthscale,
        variance=true_variance,
        total_count=total_count,
        spectral_eps=spectral_eps,
        trunc_eps=trunc_eps,
        nufft_eps=nufft_eps,
    )
    device = torch.device("cpu")
    rdtype = torch.float64
    cdtype = torch.complex128
    X = X.to(device=device, dtype=rdtype)
    y = y.to(device=device, dtype=rdtype)

    likelihood = _PGNegativeBinomialLikelihood(total_count=total_count)
    kernel = _make_kernel(
        "squared_exponential",
        dimension=2,
        lengthscale=lengthscale_init,
        variance=variance_init,
    )
    optimizer = Adam(kernel._gp_params_ref.parameters(), lr=lr, maximize=True)

    pg_b = likelihood.pg_b(y)
    variational = _VariationalState(delta=0.25 * pg_b.clone())
    records: list[IterRecord] = []
    total_e_ms = 0.0
    total_m_ms = 0.0
    t0_total = time.perf_counter()

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
        kappa = likelihood.kappa(y)
        pg_b = likelihood.pg_b(y)

        e_t0 = time.perf_counter()
        variational, estep_info = _run_estep_with_strategy(
            y,
            kappa,
            pg_b,
            likelihood,
            variational,
            spectral,
            n_probes=n_e_probes,
            cg_tol=cg_tol,
            seed=seed + 1000 * outer + 17,
            use_jacobi=use_jacobi,
            rho0=rho0,
        )
        e_ms = 1000.0 * (time.perf_counter() - e_t0)
        total_e_ms += e_ms

        m_t0 = time.perf_counter()
        mstep_out = _compute_mstep_gradient_with_strategy(
            kappa,
            variational.delta,
            spectral,
            n_probes=n_m_probes,
            cg_tol=cg_tol,
            seed=seed + 1000 * outer + 29,
            use_jacobi=use_jacobi,
        )
        m_ms = 1000.0 * (time.perf_counter() - m_t0)
        total_m_ms += m_ms

        grad = mstep_out["grad"].real
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

        records.append(
            IterRecord(
                n=n,
                seed=seed,
                strategy="jacobi" if use_jacobi else "none",
                outer_iter=outer,
                e_cg_iters=int(estep_info["cg_iters"]),
                m_cg_iters=int(mstep_out["cg_iters"]),
                e_ms=e_ms,
                m_ms=m_ms,
                lengthscale=float(kernel.lengthscale),
                variance=float(kernel.variance),
                residual=float(estep_info["residual"]),
                approx_mae=float(estep_info["metric"]),
            )
        )

    summary = SummaryRecord(
        n=n,
        seed=seed,
        strategy="jacobi" if use_jacobi else "none",
        outer_iters=outer_iters,
        mean_e_cg_iters=float(np.mean([r.e_cg_iters for r in records])),
        mean_m_cg_iters=float(np.mean([r.m_cg_iters for r in records])),
        total_e_ms=total_e_ms,
        total_m_ms=total_m_ms,
        total_ms=1000.0 * (time.perf_counter() - t0_total),
        final_lengthscale=float(kernel.lengthscale),
        final_variance=float(kernel.variance),
        final_mae=float(records[-1].approx_mae),
    )
    return records, summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare no preconditioner vs exact Jacobi under the current weighted-Toeplitz PG training loop.")
    parser.add_argument("--n-values", nargs="+", type=int, default=[5000, 50000])
    parser.add_argument("--outer-iters", type=int, default=8)
    parser.add_argument("--seed", type=int, default=760)
    parser.add_argument("--n-e-probes", type=int, default=1)
    parser.add_argument("--n-m-probes", type=int, default=1)
    parser.add_argument("--cg-tol", type=float, default=1e-6)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--lengthscale-init", type=float, default=0.30)
    parser.add_argument("--variance-init", type=float, default=1.00)
    parser.add_argument("--true-lengthscale", type=float, default=0.20)
    parser.add_argument("--true-variance", type=float, default=1.00)
    parser.add_argument("--total-count", type=float, default=3.0)
    parser.add_argument("--spectral-eps", type=float, default=1e-4)
    parser.add_argument("--trunc-eps", type=float, default=1e-4)
    parser.add_argument("--nufft-eps", type=float, default=1e-7)
    parser.add_argument("--iter-csv", type=Path, default=Path("jacobi_learning_iters.csv"))
    parser.add_argument("--summary-csv", type=Path, default=Path("jacobi_learning_summary.csv"))
    args = parser.parse_args()

    iter_rows: list[dict] = []
    summary_rows: list[dict] = []
    for n in args.n_values:
        for use_jacobi in (False, True):
            records, summary = _run_case(
                n=n,
                seed=args.seed,
                outer_iters=args.outer_iters,
                n_e_probes=args.n_e_probes,
                n_m_probes=args.n_m_probes,
                cg_tol=args.cg_tol,
                lr=args.lr,
                lengthscale_init=args.lengthscale_init,
                variance_init=args.variance_init,
                true_lengthscale=args.true_lengthscale,
                true_variance=args.true_variance,
                total_count=args.total_count,
                rho0=0.7,
                spectral_eps=args.spectral_eps,
                trunc_eps=args.trunc_eps,
                nufft_eps=args.nufft_eps,
                use_jacobi=use_jacobi,
            )
            iter_rows.extend(asdict(r) for r in records)
            summary_rows.append(asdict(summary))
            print(asdict(summary))

    _write_csv(args.iter_csv, iter_rows)
    _write_csv(args.summary_csv, summary_rows)
    print(f"wrote {len(iter_rows)} iteration rows to {args.iter_csv}")
    print(f"wrote {len(summary_rows)} summary rows to {args.summary_csv}")


if __name__ == "__main__":
    main()
