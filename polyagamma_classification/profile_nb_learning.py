from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.distributions import NegativeBinomial
from torch.optim import Adam

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from cg import ConjugateGradients
from vanilla_gp_sampling import sample_gp_fast, sample_gp_spectral_approx

from pg_classifier import (
    _PGNegativeBinomialLikelihood,
    _VariationalState,
    _build_spectral_state,
    _make_kernel,
    _negative_binomial_total_count_gradient,
    _pg_omega_expectation,
    _sample_rademacher,
)


@dataclass
class OuterProfile:
    n: int
    d: int
    seed: int
    learn_total_count: bool
    outer: int
    spectral_build_ms: float
    estep_total_ms: float
    estep_fixed_ms: float
    estep_warmstart_ms: float
    estep_main_cg_ms: float
    e_cg_iters: float
    e_warmstart_iters: float
    e_batch_size: float
    e_fixed_nufft_transforms: float
    e_main_nufft_pairs: float
    mstep_total_ms: float
    mstep_fixed_ms: float
    mstep_warmstart_ms: float
    mstep_main_cg_ms: float
    m_cg_iters: float
    m_warmstart_iters: float
    m_batch_size: float
    m_fixed_nufft_transforms: float
    m_main_nufft_pairs: float
    r_step_ms: float
    total_ms: float
    lengthscale: float
    variance: float
    total_count: float
    fit_metric: float


def _clone_variational_state(state: _VariationalState) -> _VariationalState:
    return _VariationalState(
        delta=state.delta.clone(),
        mean=None if state.mean is None else state.mean.clone(),
        sigma_diag=None if state.sigma_diag is None else state.sigma_diag.clone(),
        probes=None if state.probes is None else state.probes.clone(),
    )


def _timed_sigma_apply(
    spectral,
    delta: torch.Tensor,
    z: torch.Tensor,
    *,
    cg_tol: float,
    use_toeplitz_warm_start: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    delta_complex = delta.to(dtype=spectral.ws.dtype, device=delta.device)
    batch_size = 1 if z.dim() == 1 else z.shape[0]
    vector_input = z.dim() == 1
    if vector_input:
        z = z.unsqueeze(0)
    z = z.to(dtype=spectral.ws.dtype)

    info = {
        "fixed_ms": 0.0,
        "warmstart_ms": 0.0,
        "main_cg_ms": 0.0,
        "total_ms": 0.0,
        "cg_iters": 0.0,
        "warmstart_iters": 0.0,
        "batch_size": float(batch_size),
    }

    total_t0 = time.perf_counter()
    fixed_t0 = time.perf_counter()
    Delta = delta_complex.view(1, -1)
    z_feat = spectral.fadj_batched(z)
    Kz = spectral.fwd_batched(spectral.ws2 * z_feat)
    rhs = spectral.ws * spectral.fadj_batched(Delta * Kz)
    info["fixed_ms"] += 1000.0 * (time.perf_counter() - fixed_t0)

    def A_feat(u: torch.Tensor) -> torch.Tensor:
        psi_u = spectral.fwd_batched(spectral.ws * u)
        return u + spectral.ws * spectral.fadj_batched(Delta * psi_u)

    x0 = torch.zeros_like(rhs)
    if use_toeplitz_warm_start:
        delta_bar = delta_complex.mean()

        def A_tilde(u: torch.Tensor) -> torch.Tensor:
            if u.dim() == 1:
                return u + delta_bar * (spectral.ws * spectral.toeplitz(spectral.ws * u))
            rows = [spectral.ws * spectral.toeplitz(spectral.ws * u[b]) for b in range(u.shape[0])]
            return u + delta_bar * torch.stack(rows, dim=0)

        warm_t0 = time.perf_counter()
        cg0 = ConjugateGradients(
            A_tilde,
            rhs,
            x0=torch.zeros_like(rhs),
            tol=1e-3,
            max_iter=5000,
            early_stopping=True,
        )
        warm = cg0.solve()
        info["warmstart_ms"] = 1000.0 * (time.perf_counter() - warm_t0)
        info["warmstart_iters"] = float(cg0.iters_completed)
        if cg0.iters_completed < 4500:
            x0 = warm

    main_t0 = time.perf_counter()
    cg = ConjugateGradients(
        A_feat,
        rhs,
        x0=x0,
        tol=cg_tol,
        max_iter=2000,
        early_stopping=True,
    )
    x = cg.solve()
    info["main_cg_ms"] = 1000.0 * (time.perf_counter() - main_t0)
    info["cg_iters"] = float(cg.iters_completed)

    fixed_t1 = time.perf_counter()
    out = spectral.fwd_batched(spectral.ws * x)
    result = (Kz - out).real
    if vector_input:
        result = result.squeeze(0)
    info["fixed_ms"] += 1000.0 * (time.perf_counter() - fixed_t1)
    info["total_ms"] = 1000.0 * (time.perf_counter() - total_t0)
    return result, info


def _timed_feature_space_solve(
    delta: torch.Tensor,
    spectral,
    q: torch.Tensor,
    *,
    cg_tol: float,
    use_toeplitz_warm_start: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    omega = delta.to(dtype=spectral.ws.dtype, device=delta.device).flatten()
    D2_real = spectral.ws2.real
    eps_d = max(float(D2_real.mean()) * 1e-14, 1e-14)
    Ds = torch.sqrt(torch.clamp(D2_real, min=eps_d)).to(dtype=spectral.ws.dtype)
    Dsinv = 1.0 / Ds
    wbar = omega.real.mean().to(dtype=spectral.ws.dtype)
    batch_size = 1 if q.dim() == 1 else q.shape[0]

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

    info = {
        "warmstart_ms": 0.0,
        "main_cg_ms": 0.0,
        "cg_iters": 0.0,
        "warmstart_iters": 0.0,
        "batch_size": float(batch_size),
    }

    rhs = Ds * q if q.dim() == 1 else q * Ds[None, :]
    x0 = torch.zeros_like(rhs)

    if use_toeplitz_warm_start:
        warm_t0 = time.perf_counter()
        cg0 = ConjugateGradients(
            apply_IpS_toeplitz,
            rhs,
            x0=x0,
            tol=1e-2,
            max_iter=5000,
            early_stopping=True,
        )
        warm = cg0.solve()
        info["warmstart_ms"] = 1000.0 * (time.perf_counter() - warm_t0)
        info["warmstart_iters"] = float(cg0.iters_completed)
        if cg0.iters_completed < 4500:
            x0 = warm

    main_t0 = time.perf_counter()
    cg = ConjugateGradients(
        apply_IpS,
        rhs,
        x0=x0,
        tol=cg_tol,
        max_iter=2000,
        early_stopping=True,
    )
    y = cg.solve()
    info["main_cg_ms"] = 1000.0 * (time.perf_counter() - main_t0)
    info["cg_iters"] = float(cg.iters_completed)
    beta = Dsinv * y if q.dim() == 1 else y * Dsinv[None, :]
    return beta, info


def _profile_estep(
    targets: torch.Tensor,
    likelihood: _PGNegativeBinomialLikelihood,
    variational: _VariationalState,
    spectral,
    *,
    rho0: float,
    n_probes: int,
    cg_tol: float,
    use_toeplitz_warm_start: bool,
    seed: int | None,
) -> tuple[_VariationalState, dict[str, float]]:
    state = _clone_variational_state(variational)
    kappa = likelihood.kappa(targets)
    pg_b = likelihood.pg_b(targets)

    total_t0 = time.perf_counter()
    probes = state.probes
    if n_probes > 0 and (probes is None or probes.shape[0] != n_probes):
        probes = _sample_rademacher(
            (n_probes, targets.numel()),
            device=targets.device,
            dtype=targets.dtype,
            seed=seed,
        )

    if n_probes > 0:
        Z = torch.cat([kappa[None, :], probes], dim=0)
    else:
        Z = kappa[None, :]

    S_all, sigma_timing = _timed_sigma_apply(
        spectral,
        state.delta,
        Z,
        cg_tol=cg_tol,
        use_toeplitz_warm_start=use_toeplitz_warm_start,
    )

    mean = S_all[0]
    if n_probes > 0:
        Sz = S_all[1:]
        sigma_diag = (probes * Sz).mean(dim=0)
    else:
        sigma_diag = torch.zeros_like(mean)

    c2 = (sigma_diag + mean.pow(2)).clamp_min(1e-12)
    c = torch.sqrt(c2)
    Lambda = _pg_omega_expectation(c, pg_b)
    state.delta.mul_(1.0 - rho0).add_(rho0 * Lambda)
    state.delta.clamp_(min=0.0)
    state.mean = mean
    state.sigma_diag = sigma_diag
    state.probes = probes

    batch_size = 1.0 + float(n_probes)
    info = {
        "total_ms": 1000.0 * (time.perf_counter() - total_t0),
        "fixed_ms": sigma_timing["fixed_ms"],
        "warmstart_ms": sigma_timing["warmstart_ms"],
        "main_cg_ms": sigma_timing["main_cg_ms"],
        "cg_iters": sigma_timing["cg_iters"],
        "warmstart_iters": sigma_timing["warmstart_iters"],
        "batch_size": batch_size,
        "fixed_nufft_transforms": 4.0 * batch_size,
        "main_nufft_pairs": batch_size * sigma_timing["cg_iters"],
        "metric": likelihood.fit_metric(mean, sigma_diag, targets),
    }
    return state, info


def _profile_mstep_kernel(
    targets: torch.Tensor,
    likelihood: _PGNegativeBinomialLikelihood,
    delta: torch.Tensor,
    spectral,
    *,
    n_probes: int,
    cg_tol: float,
    use_toeplitz_warm_start: bool,
    seed: int | None,
) -> dict[str, torch.Tensor]:
    total_t0 = time.perf_counter()
    kappa = likelihood.kappa(targets)
    probes = _sample_rademacher(
        (n_probes, targets.numel()),
        device=targets.device,
        dtype=targets.dtype,
        seed=seed,
    ).to(dtype=spectral.ws.dtype)

    fixed_t0 = time.perf_counter()
    Q_block = spectral.fadj_batched(probes)
    q_y = spectral.fadj_batched(kappa.to(dtype=spectral.ws.dtype).unsqueeze(0))
    fixed_ms = 1000.0 * (time.perf_counter() - fixed_t0)

    Q_all = torch.cat([Q_block, q_y], dim=0)
    beta_all, solve_timing = _timed_feature_space_solve(
        delta,
        spectral,
        Q_all,
        cg_tol=cg_tol,
        use_toeplitz_warm_start=use_toeplitz_warm_start,
    )
    beta_probes = beta_all[:-1, :]
    beta_x = beta_all[-1, :]

    fixed_t1 = time.perf_counter()
    omega = delta.to(dtype=spectral.ws.dtype, device=delta.device).flatten()
    Rfeat = spectral.fadj_batched((omega[:, None] * probes.T).T).T
    X = Rfeat.conj() * beta_probes.T
    vals = (X.mT @ spectral.Dprime).real
    term2 = vals.mean(dim=0)

    abs2 = (beta_x.conj() * beta_x).real
    term1 = spectral.Dprime.real.T @ abs2
    grad = 0.5 * (term1 - term2)
    fixed_ms += 1000.0 * (time.perf_counter() - fixed_t1)

    batch_size = 1.0 + float(n_probes)
    return {
        "grad": grad,
        "beta_mean": beta_x,
        "total_ms": 1000.0 * (time.perf_counter() - total_t0),
        "fixed_ms": fixed_ms,
        "warmstart_ms": solve_timing["warmstart_ms"],
        "main_cg_ms": solve_timing["main_cg_ms"],
        "cg_iters": torch.tensor(solve_timing["cg_iters"]),
        "warmstart_iters": torch.tensor(solve_timing["warmstart_iters"]),
        "batch_size": torch.tensor(batch_size),
        "fixed_nufft_transforms": torch.tensor(1.0 + 2.0 * n_probes),
        "main_nufft_pairs": torch.tensor(batch_size * solve_timing["cg_iters"]),
    }


def _generate_nb_data(
    *,
    n: int,
    d: int,
    seed: int,
    true_length_scale: float,
    true_variance: float,
    true_total_count: float,
    sampling: str,
    spectral_eps: float,
    trunc_eps: float,
    nufft_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    X = torch.rand(n, d, dtype=torch.float64) * 2.0 - 1.0
    if sampling == "exact":
        latent = sample_gp_fast(
            X,
            num_samples=1,
            length_scale=true_length_scale,
            variance=true_variance,
            noise_variance=1e-4,
        )
    elif sampling == "approx":
        latent = sample_gp_spectral_approx(
            X,
            num_samples=1,
            length_scale=true_length_scale,
            variance=true_variance,
            spectral_eps=spectral_eps,
            trunc_eps=trunc_eps,
            nufft_eps=nufft_eps,
            seed=seed + 17,
        )
    else:
        raise ValueError(f"Unknown sampling mode: {sampling}")
    y = NegativeBinomial(
        total_count=torch.tensor(true_total_count, dtype=torch.float64),
        logits=latent,
    ).sample()
    return X, y.to(dtype=torch.float64)


def run_profile(
    *,
    n_values: list[int],
    d: int,
    seeds: list[int],
    outer_iters: int,
    learn_total_count_options: list[bool],
    true_length_scale: float,
    true_variance: float,
    true_total_count: float,
    init_lengthscale: float,
    init_variance: float,
    init_total_count: float,
    lr: float,
    total_count_lr: float,
    total_count_update_frequency: int,
    total_count_quadrature_nodes: int,
    sampling: str,
    n_e_probes: int,
    n_m_probes: int,
    cg_tol: float,
    nufft_eps: float,
    spectral_eps: float,
    trunc_eps: float,
    rho0: float,
    use_toeplitz_warm_start: bool,
) -> list[OuterProfile]:
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    device = torch.device("cpu")
    rdtype = torch.float64
    cdtype = torch.complex128

    profiles: list[OuterProfile] = []

    for n in n_values:
        for seed in seeds:
            X, y = _generate_nb_data(
                n=n,
                d=d,
                seed=seed,
                true_length_scale=true_length_scale,
                true_variance=true_variance,
                true_total_count=true_total_count,
                sampling=sampling,
                spectral_eps=spectral_eps,
                trunc_eps=trunc_eps,
                nufft_eps=nufft_eps,
            )
            X = X.to(device=device, dtype=rdtype)
            y = y.to(device=device, dtype=rdtype)

            for learn_total_count in learn_total_count_options:
                kernel = _make_kernel(
                    "squared_exponential",
                    dimension=d,
                    lengthscale=init_lengthscale,
                    variance=init_variance,
                )
                kernel_optimizer = Adam(kernel._gp_params_ref.parameters(), lr=lr, maximize=True)

                raw_total_count = torch.nn.Parameter(
                    torch.tensor(math.log(init_total_count), device=device, dtype=rdtype)
                )
                total_count_optimizer = Adam([raw_total_count], lr=total_count_lr, maximize=True)

                likelihood = _PGNegativeBinomialLikelihood(total_count=init_total_count)
                variational = _VariationalState(delta=0.25 * likelihood.pg_b(y))

                for outer in range(outer_iters):
                    outer_t0 = time.perf_counter()
                    current_total_count = float(torch.exp(raw_total_count).item())
                    likelihood = _PGNegativeBinomialLikelihood(total_count=current_total_count)

                    build_t0 = time.perf_counter()
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
                    spectral_build_ms = 1000.0 * (time.perf_counter() - build_t0)

                    variational, estep = _profile_estep(
                        y,
                        likelihood,
                        variational,
                        spectral,
                        rho0=rho0,
                        n_probes=n_e_probes,
                        cg_tol=cg_tol,
                        use_toeplitz_warm_start=use_toeplitz_warm_start,
                        seed=seed + 1000 * outer,
                    )

                    mstep = _profile_mstep_kernel(
                        y,
                        likelihood,
                        variational.delta,
                        spectral,
                        n_probes=n_m_probes,
                        cg_tol=cg_tol,
                        use_toeplitz_warm_start=use_toeplitz_warm_start,
                        seed=seed + 1000 * outer,
                    )
                    grad = mstep["grad"].real

                    raw = kernel._gp_params_ref.raw
                    raw.grad = torch.stack(
                        [
                            grad[0].to(dtype=raw.dtype, device=raw.device) * kernel.lengthscale,
                            grad[1].to(dtype=raw.dtype, device=raw.device) * kernel.variance,
                            torch.tensor(0.0, dtype=raw.dtype, device=raw.device),
                        ]
                    )
                    kernel_optimizer.step()
                    kernel_optimizer.zero_grad(set_to_none=True)

                    r_t0 = time.perf_counter()
                    if learn_total_count:
                        grad_total_count = _negative_binomial_total_count_gradient(
                            y,
                            variational.mean,
                            variational.sigma_diag,
                            total_count=current_total_count,
                            quadrature_nodes=total_count_quadrature_nodes,
                        )
                        if (outer + 1) % total_count_update_frequency == 0:
                            raw_total_count.grad = (
                                grad_total_count.to(dtype=raw_total_count.dtype, device=raw_total_count.device)
                                * torch.exp(raw_total_count)
                            ).detach()
                            total_count_optimizer.step()
                            total_count_optimizer.zero_grad(set_to_none=True)
                    r_step_ms = 1000.0 * (time.perf_counter() - r_t0)

                    profiles.append(
                        OuterProfile(
                            n=n,
                            d=d,
                            seed=seed,
                            learn_total_count=learn_total_count,
                            outer=outer,
                            spectral_build_ms=spectral_build_ms,
                            estep_total_ms=estep["total_ms"],
                            estep_fixed_ms=estep["fixed_ms"],
                            estep_warmstart_ms=estep["warmstart_ms"],
                            estep_main_cg_ms=estep["main_cg_ms"],
                            e_cg_iters=estep["cg_iters"],
                            e_warmstart_iters=estep["warmstart_iters"],
                            e_batch_size=estep["batch_size"],
                            e_fixed_nufft_transforms=estep["fixed_nufft_transforms"],
                            e_main_nufft_pairs=estep["main_nufft_pairs"],
                            mstep_total_ms=float(mstep["total_ms"]),
                            mstep_fixed_ms=float(mstep["fixed_ms"]),
                            mstep_warmstart_ms=float(mstep["warmstart_ms"]),
                            mstep_main_cg_ms=float(mstep["main_cg_ms"]),
                            m_cg_iters=float(mstep["cg_iters"].item()),
                            m_warmstart_iters=float(mstep["warmstart_iters"].item()),
                            m_batch_size=float(mstep["batch_size"].item()),
                            m_fixed_nufft_transforms=float(mstep["fixed_nufft_transforms"].item()),
                            m_main_nufft_pairs=float(mstep["main_nufft_pairs"].item()),
                            r_step_ms=r_step_ms,
                            total_ms=1000.0 * (time.perf_counter() - outer_t0),
                            lengthscale=float(kernel.lengthscale),
                            variance=float(kernel.variance),
                            total_count=float(torch.exp(raw_total_count).item()),
                            fit_metric=estep["metric"],
                        )
                    )

    return profiles


def _fit_log_slope(xs: np.ndarray, ys: np.ndarray) -> float:
    if len(xs) < 2 or np.any(xs <= 0.0) or np.any(ys <= 0.0):
        return float("nan")
    coeffs = np.polyfit(np.log(xs), np.log(ys), deg=1)
    return float(coeffs[0])


def _summarize(profiles: list[OuterProfile]) -> str:
    if not profiles:
        return "No profiles collected."

    rows = [asdict(p) for p in profiles]
    grouped: dict[tuple[bool, int], list[dict[str, float]]] = {}
    for row in rows:
        grouped.setdefault((bool(row["learn_total_count"]), int(row["n"])), []).append(row)

    header = (
        "mode      n  outer_ms  build_ms  e_ms  m_ms  r_ms  main_cg_%  "
        "e_cg  m_cg  main_pairs  fixed_xfm  us_per_pair_per_n"
    )
    sep = "-" * len(header)
    lines = ["Negative-binomial per-iteration efficiency profile", "", header, sep]

    summary_rows: list[tuple[bool, int, float, float]] = []
    for learn_total_count in [False, True]:
        ns = sorted({int(row["n"]) for row in rows if bool(row["learn_total_count"]) == learn_total_count})
        for n in ns:
            group = grouped[(learn_total_count, n)]
            outer_ms = np.mean([row["total_ms"] for row in group])
            build_ms = np.mean([row["spectral_build_ms"] for row in group])
            e_ms = np.mean([row["estep_total_ms"] for row in group])
            m_ms = np.mean([row["mstep_total_ms"] for row in group])
            r_ms = np.mean([row["r_step_ms"] for row in group])
            main_cg_ms = np.mean([row["estep_main_cg_ms"] + row["mstep_main_cg_ms"] for row in group])
            main_frac = 100.0 * main_cg_ms / outer_ms
            e_cg = np.mean([row["e_cg_iters"] for row in group])
            m_cg = np.mean([row["m_cg_iters"] for row in group])
            main_pairs = np.mean([row["e_main_nufft_pairs"] + row["m_main_nufft_pairs"] for row in group])
            fixed_xfm = np.mean([row["e_fixed_nufft_transforms"] + row["m_fixed_nufft_transforms"] for row in group])
            us_per_pair_per_n = 1000.0 * main_cg_ms / max(main_pairs * n, 1.0)
            mode = "learn-r" if learn_total_count else "fixed-r"
            lines.append(
                f"{mode:7s}  {n:3d}  {outer_ms:8.2f}  {build_ms:8.2f}  {e_ms:5.2f}  "
                f"{m_ms:5.2f}  {r_ms:4.2f}  {main_frac:9.1f}  {e_cg:4.1f}  "
                f"{m_cg:4.1f}  {main_pairs:10.1f}  {fixed_xfm:9.1f}  {us_per_pair_per_n:15.3f}"
            )
            summary_rows.append((learn_total_count, n, outer_ms, main_cg_ms))

    lines.extend(
        [
            "",
            "Interpretation",
            "- `main_cg_%` is the fraction of outer-iteration wall time spent inside the NUFFT-backed main CG solves only.",
            "- `main_pairs` counts batched RHS-pairs in those solves; each pair is one type-2 NUFFT plus one type-1 NUFFT.",
            "- `fixed_xfm` counts non-CG NUFFT transforms per outer iteration.",
            "- `us_per_pair_per_n` normalizes main-CG wall time by both problem size and the number of RHS-pairs.",
        ]
    )

    for learn_total_count in [False, True]:
        mode_rows = [(n, outer_ms, main_ms) for flag, n, outer_ms, main_ms in summary_rows if flag == learn_total_count]
        ns = np.array([row[0] for row in mode_rows], dtype=float)
        outer = np.array([row[1] for row in mode_rows], dtype=float)
        main = np.array([row[2] for row in mode_rows], dtype=float)
        mode = "learn-r" if learn_total_count else "fixed-r"
        lines.append(
            f"- {mode} scaling exponents on this machine: outer_ms ~ n^{_fit_log_slope(ns, outer):.2f}, "
            f"main_cg_ms ~ n^{_fit_log_slope(ns, main):.2f}."
        )

    if all((False, n) in grouped and (True, n) in grouped for n in sorted({int(row["n"]) for row in rows})):
        lines.extend(["", "Learned-r overhead"])
        for n in sorted({int(row["n"]) for row in rows}):
            fixed_group = grouped[(False, n)]
            learn_group = grouped[(True, n)]
            fixed_outer = np.mean([row["total_ms"] for row in fixed_group])
            learn_outer = np.mean([row["total_ms"] for row in learn_group])
            learn_r = np.mean([row["r_step_ms"] for row in learn_group])
            overhead_pct = 100.0 * (learn_outer - fixed_outer) / fixed_outer
            r_share = 100.0 * learn_r / learn_outer
            lines.append(
                f"- n={n:3d}: learned-r changes per-iter wall time by {overhead_pct:+.1f}% "
                f"and the explicit r-step itself is {r_share:.2f}% of the outer iteration."
            )

    return "\n".join(lines)


def _write_csv(path: Path, profiles: list[OuterProfile]) -> None:
    if not profiles:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(profiles[0]).keys()))
        writer.writeheader()
        for record in profiles:
            writer.writerow(asdict(record))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile per-iteration efficiency of the PG NB learner.")
    parser.add_argument("--n-values", type=int, nargs="+", default=[100, 200, 400, 800])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--outer-iters", type=int, default=4)
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--true-lengthscale", type=float, default=0.8)
    parser.add_argument("--true-variance", type=float, default=1.0)
    parser.add_argument("--true-total-count", type=float, default=3.0)
    parser.add_argument("--init-lengthscale", type=float, default=0.3)
    parser.add_argument("--init-variance", type=float, default=1.0)
    parser.add_argument("--init-total-count", type=float, default=1.25)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--total-count-lr", type=float, default=0.04)
    parser.add_argument("--total-count-update-frequency", type=int, default=5)
    parser.add_argument("--total-count-quadrature-nodes", type=int, default=16)
    parser.add_argument("--sampling", choices=["approx", "exact"], default="approx")
    parser.add_argument("--n-e-probes", type=int, default=1)
    parser.add_argument("--n-m-probes", type=int, default=1)
    parser.add_argument("--cg-tol", type=float, default=1e-5)
    parser.add_argument("--nufft-eps", type=float, default=1e-4)
    parser.add_argument("--spectral-eps", type=float, default=1e-4)
    parser.add_argument("--trunc-eps", type=float, default=1e-4)
    parser.add_argument("--rho0", type=float, default=0.7)
    parser.add_argument("--no-warm-start", action="store_true")
    parser.add_argument("--csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = run_profile(
        n_values=args.n_values,
        d=args.dimension,
        seeds=args.seeds,
        outer_iters=args.outer_iters,
        learn_total_count_options=[False, True],
        true_length_scale=args.true_lengthscale,
        true_variance=args.true_variance,
        true_total_count=args.true_total_count,
        init_lengthscale=args.init_lengthscale,
        init_variance=args.init_variance,
        init_total_count=args.init_total_count,
        lr=args.lr,
        total_count_lr=args.total_count_lr,
        total_count_update_frequency=args.total_count_update_frequency,
        total_count_quadrature_nodes=args.total_count_quadrature_nodes,
        sampling=args.sampling,
        n_e_probes=args.n_e_probes,
        n_m_probes=args.n_m_probes,
        cg_tol=args.cg_tol,
        nufft_eps=args.nufft_eps,
        spectral_eps=args.spectral_eps,
        trunc_eps=args.trunc_eps,
        rho0=args.rho0,
        use_toeplitz_warm_start=not args.no_warm_start,
    )
    if args.csv is not None:
        _write_csv(args.csv, profiles)
    print(_summarize(profiles))


if __name__ == "__main__":
    main()
