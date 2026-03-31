from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
import sys

import numpy as np
import torch
from torch.distributions import NegativeBinomial

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from vanilla_gp_sampling import sample_gp_spectral_approx

from pg_classifier import (
    PolyagammaGPNegativeBinomialRegressor,
    _estimate_stochastic_variance_sums,
    _evaluate_stochastic_variance_sums,
    _make_feature_space_solver,
)
from efgpnd import NUFFT


def _make_nb_dataset(
    *,
    n_train: int,
    dimension: int,
    true_lengthscale: float,
    true_variance: float,
    true_total_count: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    X = torch.rand(n_train, dimension, dtype=torch.float64) * 2.0 - 1.0
    latent = sample_gp_spectral_approx(
        X,
        num_samples=1,
        length_scale=true_lengthscale,
        variance=true_variance,
        spectral_eps=1e-4,
        trunc_eps=1e-4,
        nufft_eps=1e-7,
        seed=seed,
    ).squeeze(-1)
    y = NegativeBinomial(
        total_count=torch.tensor(true_total_count, dtype=torch.float64),
        logits=latent,
    ).sample()
    return X.cpu().numpy(), y.cpu().numpy()


def _grid_points(side: int, *, dimension: int) -> np.ndarray:
    axes = [np.linspace(-1.25, 1.25, side, dtype=np.float64) for _ in range(dimension)]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([g.reshape(-1) for g in mesh], axis=1)


def _predictive_variance_exact_stats(
    model: PolyagammaGPNegativeBinomialRegressor,
    X_new: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    X_t = torch.as_tensor(X_new, device=model._device_, dtype=model._rdtype_)
    delta = model._variational_state_.delta
    spectral = model._spectral_state_
    solve_A_beta, _, solve_info = _make_feature_space_solver(
        delta,
        spectral,
        cg_tol=model.cg_tol,
        use_toeplitz_warm_start=model.use_toeplitz_warm_start,
        use_toeplitz_preconditioner=model.use_toeplitz_preconditioner,
    )
    block_size = X_t.shape[0] if model.prediction_batch_size is None else max(
        1,
        min(model.prediction_batch_size, X_t.shape[0]),
    )
    ws2_row = spectral.ws2.unsqueeze(0)
    variances: list[torch.Tensor] = []
    total_cg_iters = 0
    total_warm_iters = 0
    num_blocks = 0

    start = time.perf_counter()
    for start_idx in range(0, X_t.shape[0], block_size):
        stop_idx = min(start_idx + block_size, X_t.shape[0])
        X_block = X_t[start_idx:stop_idx]
        block_op = NUFFT(
            X_block,
            torch.zeros_like(X_block),
            spectral.h,
            model.nufft_eps,
            cdtype=spectral.ws.dtype,
            device=X_block.device,
        )
        eye_block = torch.eye(X_block.shape[0], device=X_block.device, dtype=spectral.ws.dtype)
        phi_block = block_op.type1(eye_block, out_shape=spectral.out_shape).reshape(X_block.shape[0], -1)
        beta_block, cg_iters, warm_iters = solve_A_beta(phi_block)
        variance_block = torch.sum(
            phi_block.conj() * (ws2_row * beta_block),
            dim=1,
        ).real.clamp_min(0.0)
        variances.append(variance_block)
        total_cg_iters += int(cg_iters)
        total_warm_iters += int(warm_iters)
        num_blocks += 1
    wall_time = time.perf_counter() - start

    info = {
        "wall_time_s": wall_time,
        "num_blocks": float(num_blocks),
        "cg_iters_total": float(total_cg_iters),
        "warmstart_iters_total": float(total_warm_iters),
        "preconditioner_calls": float(solve_info["preconditioner_calls"]),
        "preconditioner_iters": float(solve_info["preconditioner_iters"]),
    }
    return torch.cat(variances).cpu().numpy(), info


def _predictive_variance_stochastic_stats(
    model: PolyagammaGPNegativeBinomialRegressor,
    X_new: np.ndarray,
    *,
    n_probes: int,
) -> tuple[np.ndarray, dict[str, float]]:
    X_t = torch.as_tensor(X_new, device=model._device_, dtype=model._rdtype_)
    delta = model._variational_state_.delta
    spectral = model._spectral_state_

    start_solve = time.perf_counter()
    est_sums, solve_info = _estimate_stochastic_variance_sums(
        delta,
        spectral,
        cg_tol=model.cg_tol,
        n_probes=n_probes,
        use_toeplitz_warm_start=model.use_toeplitz_warm_start,
        use_toeplitz_preconditioner=model.use_toeplitz_preconditioner,
        seed=None if model.random_state is None else int(model.random_state) + 2_000_000 + n_probes,
    )
    solve_time = time.perf_counter() - start_solve

    start_eval = time.perf_counter()
    variance = _evaluate_stochastic_variance_sums(
        est_sums.to(device=model._device_, dtype=spectral.ws.dtype),
        X_t,
        h=spectral.h,
        nufft_eps=model.nufft_eps,
        cdtype=spectral.ws.dtype,
    )
    eval_time = time.perf_counter() - start_eval

    info = {
        "wall_time_s": solve_time + eval_time,
        "solve_time_s": solve_time,
        "eval_time_s": eval_time,
        "cg_iters_total": float(solve_info["cg_iters"]),
        "warmstart_iters_total": float(solve_info["warmstart_iters"]),
        "preconditioner_calls": float(solve_info["preconditioner_calls"]),
        "preconditioner_iters": float(solve_info["preconditioner_iters"]),
    }
    return variance.cpu().numpy(), info


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark exact vs stochastic PG predictive variance.")
    parser.add_argument("--n-train", type=int, default=500)
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--grid-sides", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--probe-values", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=str, default="stochastic_variance_benchmark.csv")
    args = parser.parse_args()

    X_train, y_train = _make_nb_dataset(
        n_train=args.n_train,
        dimension=args.dimension,
        true_lengthscale=0.28,
        true_variance=1.0,
        true_total_count=3.0,
        seed=args.seed,
    )

    model = PolyagammaGPNegativeBinomialRegressor(
        total_count=3.0,
        lengthscale_init=0.3,
        variance_init=1.0,
        max_iter=3,
        e_step_iters=1,
        final_e_step_iters=1,
        n_e_probes=2,
        n_m_probes=2,
        cg_tol=1e-6,
        use_toeplitz_warm_start=False,
        use_toeplitz_preconditioner=True,
        prediction_batch_size=64,
        random_state=args.seed,
    )
    fit_start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - fit_start
    n_features = int(model._spectral_state_.ws.numel())

    rows: list[dict[str, float | int | str]] = []
    print(
        f"fit complete: n_train={args.n_train} d={args.dimension} "
        f"features={n_features} fit_time={fit_time:.3f}s"
    )

    for side in args.grid_sides:
        X_test = _grid_points(side, dimension=args.dimension)
        print(f"\nbenchmarking side={side} n_test={X_test.shape[0]}")
        exact_var, exact_info = _predictive_variance_exact_stats(model, X_test)
        print(
            f"  exact      time={exact_info['wall_time_s']:.3f}s "
            f"blocks={int(exact_info['num_blocks'])} "
            f"cg_total={int(exact_info['cg_iters_total'])}"
        )
        rows.append(
            {
                "mode": "exact",
                "n_train": args.n_train,
                "n_test": X_test.shape[0],
                "grid_side": side,
                "n_features": n_features,
                "n_probes": 0,
                **exact_info,
            }
        )

        for n_probes in args.probe_values:
            stochastic_var, stochastic_info = _predictive_variance_stochastic_stats(
                model,
                X_test,
                n_probes=n_probes,
            )
            rmse = float(np.sqrt(np.mean((stochastic_var - exact_var) ** 2)))
            rel_rmse = float(rmse / (np.sqrt(np.mean(exact_var**2)) + 1e-12))
            max_abs = float(np.max(np.abs(stochastic_var - exact_var)))
            speedup = float(exact_info["wall_time_s"] / stochastic_info["wall_time_s"])
            print(
                f"  stochastic probes={n_probes:2d} time={stochastic_info['wall_time_s']:.3f}s "
                f"solve={stochastic_info['solve_time_s']:.3f}s eval={stochastic_info['eval_time_s']:.3f}s "
                f"cg_total={int(stochastic_info['cg_iters_total'])} speedup={speedup:.2f}x "
                f"rel_rmse={rel_rmse:.3e} max_abs={max_abs:.3e}"
            )
            rows.append(
                {
                    "mode": "stochastic",
                    "n_train": args.n_train,
                    "n_test": X_test.shape[0],
                    "grid_side": side,
                    "n_features": n_features,
                    "n_probes": n_probes,
                    "rmse": rmse,
                    "rel_rmse": rel_rmse,
                    "max_abs_err": max_abs,
                    "speedup_vs_exact": speedup,
                    **stochastic_info,
                }
            )

    csv_path = Path(args.csv)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nresults written to {csv_path}")


if __name__ == "__main__":
    main()
