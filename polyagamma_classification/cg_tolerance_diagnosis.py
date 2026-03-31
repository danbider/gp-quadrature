from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.distributions import NegativeBinomial
from torch.optim import Adam

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from vanilla_gp_sampling import sample_gp_spectral_approx

from pg_classifier import (
    _PGNegativeBinomialLikelihood,
    _VariationalState,
    _build_spectral_state,
    _build_weighted_toeplitz,
    _compute_mstep_gradient,
    _make_kernel,
    _run_estep,
    ConjugateGradients,
)


@dataclass
class TrajectoryRow:
    run: str
    iter: int
    e_cg_tol: float
    m_cg_tol: float
    lengthscale: float
    variance: float
    mean_count_mae: float
    e_residual: float
    e_cg_iters: float
    m_cg_iters: float


@dataclass
class OneStepRow:
    iter: int
    lengthscale: float
    variance: float
    ref_e_cg_tol: float
    ref_m_cg_tol: float
    alt_tol: float
    e_delta_rel: float
    e_mean_rel: float
    e_sigma_rel: float
    grad_rel_from_loose_e: float
    grad_cos_from_loose_e: float
    grad_rel_from_loose_m: float
    grad_cos_from_loose_m: float
    ref_grad_lengthscale: float
    ref_grad_variance: float
    loose_e_grad_lengthscale: float
    loose_e_grad_variance: float
    loose_m_grad_lengthscale: float
    loose_m_grad_variance: float


@dataclass
class LocalSolveRow:
    ref_tol: float
    alt_tol: float
    operator_size: int
    operator_cond: float
    kz_norm: float
    correction_norm: float
    sigma_norm: float
    cancellation_ratio: float
    ref_direct_cg_iters: int
    alt_direct_cg_iters: int
    ref_x_rel_error: float
    alt_x_rel_error: float
    ref_sigma_rel_error: float
    alt_sigma_rel_error: float


def _clone_variational(state: _VariationalState) -> _VariationalState:
    return _VariationalState(
        delta=state.delta.clone(),
        mean=None if state.mean is None else state.mean.clone(),
        sigma_diag=None if state.sigma_diag is None else state.sigma_diag.clone(),
        probes=None if state.probes is None else state.probes.clone(),
    )


def _rel_norm(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = max(float(torch.linalg.norm(a).item()), 1e-16)
    return float((torch.linalg.norm(a - b) / denom).item())


def _grad_stats(ref: torch.Tensor, alt: torch.Tensor) -> tuple[float, float]:
    ref_r = ref.real
    alt_r = alt.real
    rel = _rel_norm(ref_r, alt_r)
    denom = max(float(torch.linalg.norm(ref_r).item() * torch.linalg.norm(alt_r).item()), 1e-16)
    cos = float(torch.dot(ref_r, alt_r).item() / denom)
    return rel, cos


def _generate_nb_data(
    *,
    n: int,
    d: int,
    seed: int,
    true_lengthscale: float,
    true_variance: float,
    total_count: float,
    spectral_eps: float,
    trunc_eps: float,
    nufft_eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    X = torch.rand(n, d, dtype=torch.float64) * 2.0 - 1.0
    latent = sample_gp_spectral_approx(
        X,
        num_samples=1,
        length_scale=true_lengthscale,
        variance=true_variance,
        spectral_eps=spectral_eps,
        trunc_eps=trunc_eps,
        nufft_eps=nufft_eps,
        seed=seed + 19,
    )
    y = NegativeBinomial(
        total_count=torch.tensor(total_count, dtype=torch.float64),
        logits=latent,
    ).sample()
    return X.numpy(), y.numpy().astype(np.float64)


def _make_state(
    *,
    X_np: np.ndarray,
    y_np: np.ndarray,
    total_count: float,
    lengthscale_init: float,
    variance_init: float,
) -> tuple[torch.Tensor, torch.Tensor, _PGNegativeBinomialLikelihood, torch.Tensor, torch.Tensor, object, _VariationalState]:
    device = torch.device("cpu")
    rdtype = torch.float64
    X_t = torch.as_tensor(X_np, device=device, dtype=rdtype)
    y_t = torch.as_tensor(y_np, device=device, dtype=rdtype)
    likelihood = _PGNegativeBinomialLikelihood(total_count=total_count)
    kappa = likelihood.kappa(y_t)
    pg_b = likelihood.pg_b(y_t)
    kernel = _make_kernel(
        "squared_exponential",
        dimension=X_t.shape[1],
        lengthscale=lengthscale_init,
        variance=variance_init,
    )
    variational = _VariationalState(delta=0.25 * pg_b.clone())
    return X_t, y_t, likelihood, kappa, pg_b, kernel, variational


def _manual_training_run(
    *,
    label: str,
    X_t: torch.Tensor,
    y_t: torch.Tensor,
    likelihood: _PGNegativeBinomialLikelihood,
    kernel,
    variational: _VariationalState,
    max_iter: int,
    e_step_iters: int,
    rho0: float,
    gamma: float,
    lr: float,
    n_e_probes: int,
    n_m_probes: int,
    e_cg_tol: float,
    m_cg_tol: float,
    spectral_eps: float,
    trunc_eps: float,
    nufft_eps: float,
    random_state: int,
) -> list[TrajectoryRow]:
    optimizer = Adam(kernel._gp_params_ref.parameters(), lr=lr, maximize=True)
    rows: list[TrajectoryRow] = []
    likelihood = _PGNegativeBinomialLikelihood(total_count=likelihood.total_count)
    kappa = likelihood.kappa(y_t)
    pg_b = likelihood.pg_b(y_t)

    for outer in range(max_iter):
        spectral = _build_spectral_state(
            X_t,
            kernel,
            spectral_eps=spectral_eps,
            trunc_eps=trunc_eps,
            nufft_eps=nufft_eps,
            rdtype=torch.float64,
            cdtype=torch.complex128,
            device=X_t.device,
        )
        variational, estep_info = _run_estep(
            y_t,
            kappa,
            pg_b,
            likelihood,
            variational,
            spectral,
            max_iters=e_step_iters,
            rho0=rho0,
            gamma=gamma,
            tol=1e-6,
            n_probes=n_e_probes,
            cg_tol=e_cg_tol,
            reuse_probes=True,
            use_toeplitz_warm_start=False,
            use_exact_weighted_toeplitz_operator=True,
            seed=random_state + 1000 * outer,
            verbose=0,
            use_toeplitz_preconditioner=False,
        )
        mstep_out = _compute_mstep_gradient(
            kappa,
            variational.delta,
            spectral,
            n_probes=n_m_probes,
            cg_tol=m_cg_tol,
            use_toeplitz_warm_start=False,
            use_exact_weighted_toeplitz_operator=True,
            seed=random_state + 1000 * outer,
            use_toeplitz_preconditioner=False,
        )
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

        rows.append(
            TrajectoryRow(
                run=label,
                iter=outer,
                e_cg_tol=e_cg_tol,
                m_cg_tol=m_cg_tol,
                lengthscale=float(kernel.lengthscale),
                variance=float(kernel.variance),
                mean_count_mae=float(estep_info["metric"]),
                e_residual=float(estep_info["residual"]),
                e_cg_iters=float(estep_info["cg_iters"]),
                m_cg_iters=float(mstep_out["cg_iters"].item()),
            )
        )

    return rows


def _reference_with_local_diagnostics(
    *,
    X_t: torch.Tensor,
    y_t: torch.Tensor,
    likelihood: _PGNegativeBinomialLikelihood,
    kernel,
    variational: _VariationalState,
    max_iter: int,
    e_step_iters: int,
    rho0: float,
    gamma: float,
    lr: float,
    n_e_probes: int,
    n_m_probes: int,
    ref_e_cg_tol: float,
    ref_m_cg_tol: float,
    alt_tol: float,
    spectral_eps: float,
    trunc_eps: float,
    nufft_eps: float,
    random_state: int,
) -> tuple[list[TrajectoryRow], list[OneStepRow]]:
    optimizer = Adam(kernel._gp_params_ref.parameters(), lr=lr, maximize=True)
    traj: list[TrajectoryRow] = []
    diags: list[OneStepRow] = []
    kappa = likelihood.kappa(y_t)
    pg_b = likelihood.pg_b(y_t)

    for outer in range(max_iter):
        spectral = _build_spectral_state(
            X_t,
            kernel,
            spectral_eps=spectral_eps,
            trunc_eps=trunc_eps,
            nufft_eps=nufft_eps,
            rdtype=torch.float64,
            cdtype=torch.complex128,
            device=X_t.device,
        )
        state_in = _clone_variational(variational)
        seed = random_state + 1000 * outer

        ref_state, ref_einfo = _run_estep(
            y_t,
            kappa,
            pg_b,
            likelihood,
            _clone_variational(state_in),
            spectral,
            max_iters=e_step_iters,
            rho0=rho0,
            gamma=gamma,
            tol=1e-6,
            n_probes=n_e_probes,
            cg_tol=ref_e_cg_tol,
            reuse_probes=True,
            use_toeplitz_warm_start=False,
            use_exact_weighted_toeplitz_operator=True,
            seed=seed,
            verbose=0,
            use_toeplitz_preconditioner=False,
        )
        loose_e_state, _ = _run_estep(
            y_t,
            kappa,
            pg_b,
            likelihood,
            _clone_variational(state_in),
            spectral,
            max_iters=e_step_iters,
            rho0=rho0,
            gamma=gamma,
            tol=1e-6,
            n_probes=n_e_probes,
            cg_tol=alt_tol,
            reuse_probes=True,
            use_toeplitz_warm_start=False,
            use_exact_weighted_toeplitz_operator=True,
            seed=seed,
            verbose=0,
            use_toeplitz_preconditioner=False,
        )

        ref_m = _compute_mstep_gradient(
            kappa,
            ref_state.delta,
            spectral,
            n_probes=n_m_probes,
            cg_tol=ref_m_cg_tol,
            use_toeplitz_warm_start=False,
            use_exact_weighted_toeplitz_operator=True,
            seed=seed,
            use_toeplitz_preconditioner=False,
        )
        loose_m = _compute_mstep_gradient(
            kappa,
            ref_state.delta,
            spectral,
            n_probes=n_m_probes,
            cg_tol=alt_tol,
            use_toeplitz_warm_start=False,
            use_exact_weighted_toeplitz_operator=True,
            seed=seed,
            use_toeplitz_preconditioner=False,
        )
        loose_e_m = _compute_mstep_gradient(
            kappa,
            loose_e_state.delta,
            spectral,
            n_probes=n_m_probes,
            cg_tol=ref_m_cg_tol,
            use_toeplitz_warm_start=False,
            use_exact_weighted_toeplitz_operator=True,
            seed=seed,
            use_toeplitz_preconditioner=False,
        )

        ref_grad = ref_m["grad"].real
        loose_e_grad = loose_e_m["grad"].real
        loose_m_grad = loose_m["grad"].real
        grad_rel_from_loose_e, grad_cos_from_loose_e = _grad_stats(ref_grad, loose_e_grad)
        grad_rel_from_loose_m, grad_cos_from_loose_m = _grad_stats(ref_grad, loose_m_grad)

        diags.append(
            OneStepRow(
                iter=outer,
                lengthscale=float(kernel.lengthscale),
                variance=float(kernel.variance),
                ref_e_cg_tol=ref_e_cg_tol,
                ref_m_cg_tol=ref_m_cg_tol,
                alt_tol=alt_tol,
                e_delta_rel=_rel_norm(ref_state.delta, loose_e_state.delta),
                e_mean_rel=_rel_norm(ref_state.mean, loose_e_state.mean),
                e_sigma_rel=_rel_norm(ref_state.sigma_diag, loose_e_state.sigma_diag),
                grad_rel_from_loose_e=grad_rel_from_loose_e,
                grad_cos_from_loose_e=grad_cos_from_loose_e,
                grad_rel_from_loose_m=grad_rel_from_loose_m,
                grad_cos_from_loose_m=grad_cos_from_loose_m,
                ref_grad_lengthscale=float(ref_grad[0].item()),
                ref_grad_variance=float(ref_grad[1].item()),
                loose_e_grad_lengthscale=float(loose_e_grad[0].item()),
                loose_e_grad_variance=float(loose_e_grad[1].item()),
                loose_m_grad_lengthscale=float(loose_m_grad[0].item()),
                loose_m_grad_variance=float(loose_m_grad[1].item()),
            )
        )

        raw = kernel._gp_params_ref.raw
        raw.grad = torch.stack(
            [
                ref_grad[0].to(dtype=raw.dtype, device=raw.device) * kernel.lengthscale,
                ref_grad[1].to(dtype=raw.dtype, device=raw.device) * kernel.variance,
                torch.tensor(0.0, dtype=raw.dtype, device=raw.device),
            ]
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        variational = ref_state

        traj.append(
            TrajectoryRow(
                run=f"reference_e{ref_e_cg_tol:g}_m{ref_m_cg_tol:g}",
                iter=outer,
                e_cg_tol=ref_e_cg_tol,
                m_cg_tol=ref_m_cg_tol,
                lengthscale=float(kernel.lengthscale),
                variance=float(kernel.variance),
                mean_count_mae=float(ref_einfo["metric"]),
                e_residual=float(ref_einfo["residual"]),
                e_cg_iters=float(ref_einfo["cg_iters"]),
                m_cg_iters=float(ref_m["cg_iters"].item()),
            )
        )

    return traj, diags


def _write_csv(path: Path, rows: list[object]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _local_sigma_diagnostic(
    *,
    X_t: torch.Tensor,
    kappa: torch.Tensor,
    kernel,
    variational: _VariationalState,
    ref_tol: float,
    alt_tol: float,
    spectral_eps: float,
    trunc_eps: float,
    nufft_eps: float,
) -> LocalSolveRow:
    spectral = _build_spectral_state(
        X_t,
        kernel,
        spectral_eps=spectral_eps,
        trunc_eps=trunc_eps,
        nufft_eps=nufft_eps,
        rdtype=torch.float64,
        cdtype=torch.complex128,
        device=X_t.device,
    )
    delta_complex = variational.delta.to(dtype=spectral.ws.dtype, device=variational.delta.device)
    weighted_toeplitz = _build_weighted_toeplitz(delta_complex, spectral)
    M = spectral.ws.numel()

    A = torch.empty((M, M), dtype=spectral.ws.dtype, device=X_t.device)
    eye = torch.eye(M, dtype=spectral.ws.dtype, device=X_t.device)
    for j in range(M):
        ej = eye[:, j]
        A[:, j] = ej + spectral.ws * weighted_toeplitz(spectral.ws * ej)

    eigs = torch.linalg.eigvalsh(A.real.cpu())
    operator_cond = float((eigs.max() / eigs.min()).item())

    z = kappa.unsqueeze(0).to(dtype=spectral.ws.dtype)
    Delta = delta_complex.view(1, -1)
    z_feat = spectral.fadj_batched(z)
    Kz = spectral.fwd_batched(spectral.ws2 * z_feat).squeeze(0)
    rhs = (spectral.ws * spectral.fadj_batched(Delta * Kz.unsqueeze(0))).squeeze(0)

    x_exact = torch.linalg.solve(A, rhs)
    correction_exact = spectral.fwd(spectral.ws * x_exact)
    sigma_exact = (Kz - correction_exact).real

    ref_cg = ConjugateGradients(
        A,
        rhs,
        x0=torch.zeros_like(rhs),
        tol=ref_tol,
        max_iter=2000,
        early_stopping=True,
    )
    x_ref = ref_cg.solve()
    alt_cg = ConjugateGradients(
        A,
        rhs,
        x0=torch.zeros_like(rhs),
        tol=alt_tol,
        max_iter=2000,
        early_stopping=True,
    )
    x_alt = alt_cg.solve()

    sigma_ref = (Kz - spectral.fwd(spectral.ws * x_ref)).real
    sigma_alt = (Kz - spectral.fwd(spectral.ws * x_alt)).real

    return LocalSolveRow(
        ref_tol=ref_tol,
        alt_tol=alt_tol,
        operator_size=M,
        operator_cond=operator_cond,
        kz_norm=float(torch.linalg.norm(Kz).item()),
        correction_norm=float(torch.linalg.norm(correction_exact).item()),
        sigma_norm=float(torch.linalg.norm(sigma_exact).item()),
        cancellation_ratio=float(
            max(
                torch.linalg.norm(Kz).item(),
                torch.linalg.norm(correction_exact).item(),
            )
            / max(torch.linalg.norm(sigma_exact).item(), 1e-16)
        ),
        ref_direct_cg_iters=int(ref_cg.iters_completed),
        alt_direct_cg_iters=int(alt_cg.iters_completed),
        ref_x_rel_error=_rel_norm(x_exact, x_ref),
        alt_x_rel_error=_rel_norm(x_exact, x_alt),
        ref_sigma_rel_error=_rel_norm(sigma_exact, sigma_ref),
        alt_sigma_rel_error=_rel_norm(sigma_exact, sigma_alt),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose whether loose CG tolerance affects the E-step or M-step more in large-n PG NB training.")
    parser.add_argument("--n", type=int, default=50_000)
    parser.add_argument("--d", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=8)
    parser.add_argument("--e-step-iters", type=int, default=1)
    parser.add_argument("--n-e-probes", type=int, default=1)
    parser.add_argument("--n-m-probes", type=int, default=1)
    parser.add_argument("--true-lengthscale", type=float, default=0.30)
    parser.add_argument("--true-variance", type=float, default=1.0)
    parser.add_argument("--total-count", type=float, default=3.0)
    parser.add_argument("--lengthscale-init", type=float, default=0.30)
    parser.add_argument("--variance-init", type=float, default=1.0)
    parser.add_argument("--rho0", type=float, default=0.7)
    parser.add_argument("--gamma", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--ref-e-cg-tol", type=float, default=1e-6)
    parser.add_argument("--ref-m-cg-tol", type=float, default=1e-6)
    parser.add_argument("--alt-cg-tol", type=float, default=1e-5)
    parser.add_argument("--spectral-eps", type=float, default=1e-4)
    parser.add_argument("--trunc-eps", type=float, default=1e-4)
    parser.add_argument("--nufft-eps", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--traj-csv", type=Path, default=Path("cg_tolerance_diagnosis_trajectories.csv"))
    parser.add_argument("--diag-csv", type=Path, default=Path("cg_tolerance_diagnosis_onestep.csv"))
    parser.add_argument("--local-csv", type=Path, default=Path("cg_tolerance_diagnosis_local.csv"))
    args = parser.parse_args()

    X_np, y_np = _generate_nb_data(
        n=args.n,
        d=args.d,
        seed=args.seed,
        true_lengthscale=args.true_lengthscale,
        true_variance=args.true_variance,
        total_count=args.total_count,
        spectral_eps=args.spectral_eps,
        trunc_eps=args.trunc_eps,
        nufft_eps=args.nufft_eps,
    )

    X_t, y_t, likelihood, kappa, _, kernel_ref, variational_ref = _make_state(
        X_np=X_np,
        y_np=y_np,
        total_count=args.total_count,
        lengthscale_init=args.lengthscale_init,
        variance_init=args.variance_init,
    )

    local_row = _local_sigma_diagnostic(
        X_t=X_t,
        kappa=kappa,
        kernel=kernel_ref,
        variational=variational_ref,
        ref_tol=args.ref_e_cg_tol,
        alt_tol=args.alt_cg_tol,
        spectral_eps=args.spectral_eps,
        trunc_eps=args.trunc_eps,
        nufft_eps=args.nufft_eps,
    )

    t0 = time.time()
    ref_traj, diag_rows = _reference_with_local_diagnostics(
        X_t=X_t,
        y_t=y_t,
        likelihood=likelihood,
        kernel=kernel_ref,
        variational=variational_ref,
        max_iter=args.max_iter,
        e_step_iters=args.e_step_iters,
        rho0=args.rho0,
        gamma=args.gamma,
        lr=args.lr,
        n_e_probes=args.n_e_probes,
        n_m_probes=args.n_m_probes,
        ref_e_cg_tol=args.ref_e_cg_tol,
        ref_m_cg_tol=args.ref_m_cg_tol,
        alt_tol=args.alt_cg_tol,
        spectral_eps=args.spectral_eps,
        trunc_eps=args.trunc_eps,
        nufft_eps=args.nufft_eps,
        random_state=args.seed,
    )
    ref_dt = time.time() - t0

    traj_rows = list(ref_traj)
    combos = [
        ("loose_both", args.alt_cg_tol, args.alt_cg_tol),
        ("loose_e_only", args.alt_cg_tol, args.ref_m_cg_tol),
        ("loose_m_only", args.ref_e_cg_tol, args.alt_cg_tol),
    ]
    for label, e_tol, m_tol in combos:
        _, _, likelihood_i, _, _, kernel_i, variational_i = _make_state(
            X_np=X_np,
            y_np=y_np,
            total_count=args.total_count,
            lengthscale_init=args.lengthscale_init,
            variance_init=args.variance_init,
        )
        traj_rows.extend(
            _manual_training_run(
                label=label,
                X_t=X_t,
                y_t=y_t,
                likelihood=likelihood_i,
                kernel=kernel_i,
                variational=variational_i,
                max_iter=args.max_iter,
                e_step_iters=args.e_step_iters,
                rho0=args.rho0,
                gamma=args.gamma,
                lr=args.lr,
                n_e_probes=args.n_e_probes,
                n_m_probes=args.n_m_probes,
                e_cg_tol=e_tol,
                m_cg_tol=m_tol,
                spectral_eps=args.spectral_eps,
                trunc_eps=args.trunc_eps,
                nufft_eps=args.nufft_eps,
                random_state=args.seed,
            )
        )

    _write_csv(args.traj_csv, traj_rows)
    _write_csv(args.diag_csv, diag_rows)
    _write_csv(args.local_csv, [local_row])

    print(f"reference+diagnostics time: {ref_dt:.2f}s")
    print(f"wrote trajectory rows to {args.traj_csv}")
    print(f"wrote one-step rows to {args.diag_csv}")
    print(f"wrote local solve row to {args.local_csv}")


if __name__ == "__main__":
    main()
