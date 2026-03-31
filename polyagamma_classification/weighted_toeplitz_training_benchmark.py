from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from fixed_hyperparam_preconditioning import _delta_summary, _generate_nb_data
from pg_classifier import (
    _PGNegativeBinomialLikelihood,
    _VariationalState,
    _build_spectral_state,
    _compute_mstep_gradient,
    _make_kernel,
    _run_estep,
)


@dataclass(frozen=True)
class Mode:
    name: str
    use_exact_weighted_toeplitz_operator: bool
    use_toeplitz_preconditioner: bool


@dataclass
class Record:
    n: int
    feature_dim: int
    mtot: int
    delta_cv: float
    solve_kind: str
    mode: str
    cg_iters: float
    preconditioner_calls: float
    preconditioner_iters: float
    wall_ms: float
    lengthscale: float
    variance: float


def _modes() -> list[Mode]:
    return [
        Mode("nufft_none", use_exact_weighted_toeplitz_operator=False, use_toeplitz_preconditioner=False),
        Mode("nufft_toeplitz_pcg", use_exact_weighted_toeplitz_operator=False, use_toeplitz_preconditioner=True),
        Mode("weighted_toeplitz_none", use_exact_weighted_toeplitz_operator=True, use_toeplitz_preconditioner=False),
        Mode("weighted_toeplitz_toeplitz_pcg", use_exact_weighted_toeplitz_operator=True, use_toeplitz_preconditioner=True),
    ]


def run_benchmark(
    *,
    n_values: list[int],
    d: int,
    burnin_iters: int,
    n_e_probes: int,
    n_m_probes: int,
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
    rows: list[Record] = []

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

        base_state = _VariationalState(delta=0.25 * pg_b.clone())
        base_state, _ = _run_estep(
            y,
            kappa,
            pg_b,
            likelihood,
            base_state,
            spectral,
            max_iters=burnin_iters,
            rho0=0.7,
            gamma=1e-3,
            tol=1e-6,
            n_probes=n_e_probes,
            cg_tol=cg_tol,
            reuse_probes=True,
            use_toeplitz_warm_start=False,
            use_exact_weighted_toeplitz_operator=False,
            seed=data_seed + 17,
            verbose=0,
            use_toeplitz_preconditioner=True,
        )
        delta_cv, _ = _delta_summary(base_state.delta)

        for mode in _modes():
            t0 = time.perf_counter()
            _, estep_info = _run_estep(
                y,
                kappa,
                pg_b,
                likelihood,
                _VariationalState(delta=base_state.delta.clone()),
                spectral,
                max_iters=1,
                rho0=0.7,
                gamma=1e-3,
                tol=1e-6,
                n_probes=n_e_probes,
                cg_tol=cg_tol,
                reuse_probes=True,
                use_toeplitz_warm_start=False,
                use_exact_weighted_toeplitz_operator=mode.use_exact_weighted_toeplitz_operator,
                seed=data_seed + 101,
                verbose=0,
                use_toeplitz_preconditioner=mode.use_toeplitz_preconditioner,
            )
            estep_ms = 1000.0 * (time.perf_counter() - t0)
            rows.append(
                Record(
                    n=n,
                    feature_dim=int(spectral.xis.shape[0]),
                    mtot=int(spectral.mtot),
                    delta_cv=delta_cv,
                    solve_kind="estep",
                    mode=mode.name,
                    cg_iters=estep_info["cg_iters"],
                    preconditioner_calls=estep_info["preconditioner_calls"],
                    preconditioner_iters=estep_info["preconditioner_iters"],
                    wall_ms=estep_ms,
                    lengthscale=lengthscale,
                    variance=variance,
                )
            )

            t1 = time.perf_counter()
            mstep_out = _compute_mstep_gradient(
                kappa,
                base_state.delta,
                spectral,
                n_probes=n_m_probes,
                cg_tol=cg_tol,
                use_toeplitz_warm_start=False,
                use_exact_weighted_toeplitz_operator=mode.use_exact_weighted_toeplitz_operator,
                seed=data_seed + 202,
                use_toeplitz_preconditioner=mode.use_toeplitz_preconditioner,
            )
            mstep_ms = 1000.0 * (time.perf_counter() - t1)
            rows.append(
                Record(
                    n=n,
                    feature_dim=int(spectral.xis.shape[0]),
                    mtot=int(spectral.mtot),
                    delta_cv=delta_cv,
                    solve_kind="mstep",
                    mode=mode.name,
                    cg_iters=float(mstep_out["cg_iters"].item()),
                    preconditioner_calls=float(mstep_out["preconditioner_calls"].item()),
                    preconditioner_iters=float(mstep_out["preconditioner_iters"].item()),
                    wall_ms=mstep_ms,
                    lengthscale=lengthscale,
                    variance=variance,
                )
            )

    return rows


def _write_csv(path: Path, rows: list[Record]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark exact weighted-Toeplitz training solves against the current NUFFT backend.")
    parser.add_argument("--n-values", nargs="+", type=int, default=[5000, 10000, 50000])
    parser.add_argument("--d", type=int, default=2)
    parser.add_argument("--burnin-iters", type=int, default=3)
    parser.add_argument("--n-e-probes", type=int, default=2)
    parser.add_argument("--n-m-probes", type=int, default=2)
    parser.add_argument("--lengthscale", type=float, default=0.35)
    parser.add_argument("--variance", type=float, default=1.0)
    parser.add_argument("--total-count", type=float, default=3.0)
    parser.add_argument("--spectral-eps", type=float, default=1e-4)
    parser.add_argument("--trunc-eps", type=float, default=1e-4)
    parser.add_argument("--nufft-eps", type=float, default=1e-7)
    parser.add_argument("--cg-tol", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=Path, default=Path("weighted_toeplitz_training_benchmark.csv"))
    args = parser.parse_args()

    rows = run_benchmark(
        n_values=args.n_values,
        d=args.d,
        burnin_iters=args.burnin_iters,
        n_e_probes=args.n_e_probes,
        n_m_probes=args.n_m_probes,
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
