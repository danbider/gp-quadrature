from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from vanilla_gp_sampling import sample_bernoulli_gp

from pg_classifier import (
    _PGBernoulliLikelihood,
    _VariationalState,
    _build_spectral_state,
    _compute_mstep_gradient,
    _make_kernel,
    _run_estep,
)


@dataclass
class WarmStartRecord:
    outer: int
    delta_cv_e: float
    delta_rel_l2_e: float
    e_main_cg_cold: float
    e_main_cg_warm: float
    e_warmstart_iters: float
    delta_cv_m: float
    delta_rel_l2_m: float
    m_main_cg_cold: float
    m_main_cg_warm: float
    m_warmstart_iters: float
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


def _format_table(records: list[WarmStartRecord]) -> str:
    headers = [
        "outer",
        "cv_e",
        "e_cold",
        "e_warm",
        "e_ws",
        "cv_m",
        "m_cold",
        "m_warm",
        "m_ws",
        "ls",
        "var",
    ]
    rows = []
    for rec in records:
        rows.append(
            [
                f"{rec.outer:d}",
                f"{rec.delta_cv_e:.3f}",
                f"{rec.e_main_cg_cold:.0f}",
                f"{rec.e_main_cg_warm:.0f}",
                f"{rec.e_warmstart_iters:.0f}",
                f"{rec.delta_cv_m:.3f}",
                f"{rec.m_main_cg_cold:.0f}",
                f"{rec.m_main_cg_warm:.0f}",
                f"{rec.m_warmstart_iters:.0f}",
                f"{rec.lengthscale:.3f}",
                f"{rec.variance:.3f}",
            ]
        )

    widths = [max(len(headers[i]), max(len(row[i]) for row in rows)) for i in range(len(headers))]
    header_line = "  ".join(headers[i].rjust(widths[i]) for i in range(len(headers)))
    sep_line = "  ".join("-" * widths[i] for i in range(len(headers)))
    body = ["  ".join(row[i].rjust(widths[i]) for i in range(len(headers))) for row in rows]
    return "\n".join([header_line, sep_line, *body])


def run_diagnostic(
    *,
    n: int = 180,
    d: int = 2,
    outer_iters: int = 10,
    n_e_probes: int = 6,
    n_m_probes: int = 12,
    true_length_scale: float = 0.7,
    true_variance: float = 1.0,
    init_lengthscale: float = 0.3,
    init_variance: float = 1.0,
    lr: float = 0.05,
    spectral_eps: float = 1e-4,
    trunc_eps: float = 1e-4,
    nufft_eps: float = 1e-7,
    cg_tol: float = 1e-6,
    seed: int = 0,
) -> list[WarmStartRecord]:
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cpu")
    rdtype = torch.float64
    cdtype = torch.complex128

    X = torch.rand(n, d, dtype=rdtype, device=device) * 2.0 - 1.0
    y, _ = sample_bernoulli_gp(
        X,
        length_scale=true_length_scale,
        variance=true_variance,
        noise_variance=1e-4,
        seed=seed + 11,
    )
    y = y.to(device=device, dtype=rdtype)

    likelihood = _PGBernoulliLikelihood()
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
    records: list[WarmStartRecord] = []

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

        warm_state = _clone_variational_state(variational)
        warm_state, estep_warm = _run_estep(
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

        cold_state = _clone_variational_state(variational)
        _, estep_cold = _run_estep(
            y,
            kappa,
            pg_b,
            likelihood,
            cold_state,
            spectral,
            max_iters=1,
            rho0=0.7,
            gamma=1e-3,
            tol=1e-6,
            n_probes=n_e_probes,
            cg_tol=cg_tol,
            reuse_probes=True,
            use_toeplitz_warm_start=False,
            seed=seed + 1000 * outer,
            verbose=0,
        )

        delta_cv_m, delta_rel_l2_m = _delta_summary(warm_state.delta)
        mstep_warm = _compute_mstep_gradient(
            kappa,
            warm_state.delta,
            spectral,
            n_probes=n_m_probes,
            cg_tol=cg_tol,
            use_toeplitz_warm_start=True,
            seed=seed + 1000 * outer,
        )
        mstep_cold = _compute_mstep_gradient(
            kappa,
            warm_state.delta,
            spectral,
            n_probes=n_m_probes,
            cg_tol=cg_tol,
            use_toeplitz_warm_start=False,
            seed=seed + 1000 * outer,
        )

        records.append(
            WarmStartRecord(
                outer=outer,
                delta_cv_e=delta_cv_e,
                delta_rel_l2_e=delta_rel_l2_e,
                e_main_cg_cold=estep_cold["cg_iters"],
                e_main_cg_warm=estep_warm["cg_iters"],
                e_warmstart_iters=estep_warm["warmstart_iters"],
                delta_cv_m=delta_cv_m,
                delta_rel_l2_m=delta_rel_l2_m,
                m_main_cg_cold=float(mstep_cold["cg_iters"].item()),
                m_main_cg_warm=float(mstep_warm["cg_iters"].item()),
                m_warmstart_iters=float(mstep_warm["warmstart_iters"].item()),
                lengthscale=float(kernel.lengthscale),
                variance=float(kernel.variance),
            )
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

    return records


def summarize(records: list[WarmStartRecord]) -> str:
    e_cold = np.array([r.e_main_cg_cold for r in records], dtype=float)
    e_warm = np.array([r.e_main_cg_warm for r in records], dtype=float)
    e_ws = np.array([r.e_warmstart_iters for r in records], dtype=float)
    m_cold = np.array([r.m_main_cg_cold for r in records], dtype=float)
    m_warm = np.array([r.m_main_cg_warm for r in records], dtype=float)
    m_ws = np.array([r.m_warmstart_iters for r in records], dtype=float)
    cv_e = np.array([r.delta_cv_e for r in records], dtype=float)
    cv_m = np.array([r.delta_cv_m for r in records], dtype=float)

    def corr(x: np.ndarray, y: np.ndarray) -> float:
        if len(x) < 2 or np.allclose(np.std(x), 0.0) or np.allclose(np.std(y), 0.0):
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    lines = [
        "Warm-start diagnostic on synthetic Bernoulli GP data",
        "",
        _format_table(records),
        "",
        "Summary",
        f"- E-step NUFFT-CG iterations: warm {e_warm.mean():.1f} vs cold {e_cold.mean():.1f} "
        f"({100.0 * (1.0 - e_warm.mean() / e_cold.mean()):.1f}% reduction in expensive iterations).",
        f"- E-step Toeplitz warm-start iterations: mean {e_ws.mean():.1f}.",
        f"- M-step NUFFT-CG iterations: warm {m_warm.mean():.1f} vs cold {m_cold.mean():.1f} "
        f"({100.0 * (1.0 - m_warm.mean() / m_cold.mean()):.1f}% reduction in expensive iterations).",
        f"- M-step Toeplitz warm-start iterations: mean {m_ws.mean():.1f}.",
        f"- Delta closeness-to-scalar (CV before E-step): mean {cv_e.mean():.3f}.",
        f"- Delta closeness-to-scalar (CV before M-step): mean {cv_m.mean():.3f}.",
        f"- Corr(CV before E-step, E-step iteration savings): {corr(cv_e, e_cold - e_warm):+.3f}.",
        f"- Corr(CV before M-step, M-step iteration savings): {corr(cv_m, m_cold - m_warm):+.3f}.",
        "",
        "Interpretation",
        "- Each main CG iteration corresponds to one batched type-2 NUFFT and one batched type-1 NUFFT.",
        "- Toeplitz warm-start iterations are not counted as expensive NUFFT iterations here.",
        "- The warm start is approximating Delta by delta_bar I, so it should work best when Delta is close to a scalar multiple of the identity, not merely diagonal.",
    ]
    return "\n".join(lines)


def main() -> None:
    records = run_diagnostic()
    print(summarize(records))


if __name__ == "__main__":
    main()
