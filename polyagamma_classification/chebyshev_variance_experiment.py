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


def chebyshev_lobatto_nodes(a: float, b: float, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    if n_nodes < 2:
        raise ValueError("n_nodes must be at least 2 for Chebyshev-Lobatto interpolation.")
    k = np.arange(n_nodes, dtype=np.float64)
    nodes_std = np.cos(np.pi * k / (n_nodes - 1))
    weights = np.ones(n_nodes, dtype=np.float64)
    weights[0] = 0.5
    weights[-1] = 0.5
    weights *= (-1.0) ** k
    nodes = 0.5 * (a + b) + 0.5 * (b - a) * nodes_std
    scale = 2.0 / (b - a)
    order = np.argsort(nodes)
    return nodes[order], (weights * scale)[order]


def barycentric_interpolation_matrix(
    nodes: np.ndarray,
    weights: np.ndarray,
    targets: np.ndarray,
    *,
    atol: float = 1e-14,
) -> np.ndarray:
    nodes = np.asarray(nodes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    diff = targets[:, None] - nodes[None, :]
    mat = np.empty((targets.size, nodes.size), dtype=np.float64)

    close = np.isclose(diff, 0.0, atol=atol, rtol=0.0)
    matched_rows = close.any(axis=1)
    if np.any(matched_rows):
        matched_idx = np.argmax(close[matched_rows], axis=1)
        mat[matched_rows] = 0.0
        mat[np.where(matched_rows)[0], matched_idx] = 1.0

    unmatched_rows = ~matched_rows
    if np.any(unmatched_rows):
        diff_u = diff[unmatched_rows]
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = weights[None, :] / diff_u
        mat[unmatched_rows] = raw / raw.sum(axis=1, keepdims=True)
    return mat


def tensor_product_chebyshev_interpolate_2d(
    node_values: np.ndarray,
    interp_x: np.ndarray,
    interp_y: np.ndarray,
) -> np.ndarray:
    return interp_x @ node_values @ interp_y.T


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


def _uniform_grid(side: int, *, dimension: int, low: float, high: float) -> tuple[np.ndarray, list[np.ndarray]]:
    axes = [np.linspace(low, high, side, dtype=np.float64) for _ in range(dimension)]
    mesh = np.meshgrid(*axes, indexing="ij")
    points = np.stack([g.reshape(-1) for g in mesh], axis=1)
    return points, axes


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
        seed=None if model.random_state is None else int(model.random_state) + 3_000_000 + n_probes,
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


def _chebyshev_interpolated_variance_stats(
    model: PolyagammaGPNegativeBinomialRegressor,
    target_axes: list[np.ndarray],
    *,
    n_nodes_per_dim: int,
) -> tuple[np.ndarray, dict[str, float]]:
    if len(target_axes) != 2:
        raise ValueError("This experiment currently supports 2D only.")
    (x_axis, y_axis) = target_axes
    x_nodes, x_weights = chebyshev_lobatto_nodes(float(x_axis[0]), float(x_axis[-1]), n_nodes_per_dim)
    y_nodes, y_weights = chebyshev_lobatto_nodes(float(y_axis[0]), float(y_axis[-1]), n_nodes_per_dim)
    mesh = np.meshgrid(x_nodes, y_nodes, indexing="ij")
    node_points = np.stack([g.reshape(-1) for g in mesh], axis=1)

    node_values, exact_info = _predictive_variance_exact_stats(model, node_points)
    node_grid = node_values.reshape(n_nodes_per_dim, n_nodes_per_dim)

    interp_start = time.perf_counter()
    interp_x = barycentric_interpolation_matrix(x_nodes, x_weights, x_axis)
    interp_y = barycentric_interpolation_matrix(y_nodes, y_weights, y_axis)
    interpolated = tensor_product_chebyshev_interpolate_2d(node_grid, interp_x, interp_y)
    interp_time = time.perf_counter() - interp_start

    info = {
        "wall_time_s": exact_info["wall_time_s"] + interp_time,
        "node_eval_time_s": exact_info["wall_time_s"],
        "interp_time_s": interp_time,
        "cg_iters_total": exact_info["cg_iters_total"],
        "warmstart_iters_total": exact_info["warmstart_iters_total"],
        "preconditioner_calls": exact_info["preconditioner_calls"],
        "preconditioner_iters": exact_info["preconditioner_iters"],
        "num_blocks": exact_info["num_blocks"],
        "n_nodes_total": float(node_points.shape[0]),
    }
    return interpolated.reshape(-1), info


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Chebyshev interpolation against stochastic PG variance maps.")
    parser.add_argument("--n-train", type=int, default=500)
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--grid-side", type=int, default=32)
    parser.add_argument("--grid-low", type=float, default=-1.25)
    parser.add_argument("--grid-high", type=float, default=1.25)
    parser.add_argument("--stochastic-probes", type=int, default=128)
    parser.add_argument("--node-values", type=int, nargs="+", default=[5, 7, 9, 11, 13, 15])
    parser.add_argument("--max-iter", type=int, default=3)
    parser.add_argument("--e-step-iters", type=int, default=1)
    parser.add_argument("--final-e-step-iters", type=int, default=1)
    parser.add_argument("--n-e-probes", type=int, default=2)
    parser.add_argument("--n-m-probes", type=int, default=2)
    parser.add_argument("--cg-tol", type=float, default=1e-6)
    parser.add_argument("--prediction-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=str, default="chebyshev_variance_benchmark.csv")
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
        max_iter=args.max_iter,
        e_step_iters=args.e_step_iters,
        final_e_step_iters=args.final_e_step_iters,
        n_e_probes=args.n_e_probes,
        n_m_probes=args.n_m_probes,
        cg_tol=args.cg_tol,
        use_toeplitz_warm_start=False,
        use_toeplitz_preconditioner=True,
        prediction_batch_size=args.prediction_batch_size,
        random_state=args.seed,
    )
    fit_start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - fit_start
    n_features = int(model._spectral_state_.ws.numel())

    X_test, target_axes = _uniform_grid(
        args.grid_side,
        dimension=args.dimension,
        low=args.grid_low,
        high=args.grid_high,
    )
    exact_var, exact_info = _predictive_variance_exact_stats(model, X_test)
    stochastic_var, stochastic_info = _predictive_variance_stochastic_stats(
        model,
        X_test,
        n_probes=args.stochastic_probes,
    )

    exact_rms = float(np.sqrt(np.mean(exact_var**2)))
    stochastic_rmse = float(np.sqrt(np.mean((stochastic_var - exact_var) ** 2)))
    stochastic_rel_rmse = float(stochastic_rmse / (exact_rms + 1e-12))

    rows: list[dict[str, float | int | str]] = [
        {
            "mode": "exact",
            "n_train": args.n_train,
            "grid_side": args.grid_side,
            "n_test": X_test.shape[0],
            "n_features": n_features,
            "fit_time_s": fit_time,
            "exact_rms": exact_rms,
            **exact_info,
        },
        {
            "mode": "stochastic",
            "n_train": args.n_train,
            "grid_side": args.grid_side,
            "n_test": X_test.shape[0],
            "n_features": n_features,
            "fit_time_s": fit_time,
            "n_probes": args.stochastic_probes,
            "rmse": stochastic_rmse,
            "rel_rmse": stochastic_rel_rmse,
            "max_abs_err": float(np.max(np.abs(stochastic_var - exact_var))),
            **stochastic_info,
        },
    ]

    print(
        f"fit complete: n_train={args.n_train} d={args.dimension} "
        f"features={n_features} fit_time={fit_time:.3f}s"
    )
    print(
        f"exact map: time={exact_info['wall_time_s']:.3f}s "
        f"cg_total={int(exact_info['cg_iters_total'])} exact_rms={exact_rms:.4e}"
    )
    print(
        f"stochastic {args.stochastic_probes} probes: time={stochastic_info['wall_time_s']:.3f}s "
        f"cg_total={int(stochastic_info['cg_iters_total'])} "
        f"rel_rmse={stochastic_rel_rmse:.4e}"
    )

    best_match: dict[str, float | int | str] | None = None
    for n_nodes in args.node_values:
        interp_var, interp_info = _chebyshev_interpolated_variance_stats(
            model,
            target_axes,
            n_nodes_per_dim=n_nodes,
        )
        rmse = float(np.sqrt(np.mean((interp_var - exact_var) ** 2)))
        rel_rmse = float(rmse / (exact_rms + 1e-12))
        max_abs = float(np.max(np.abs(interp_var - exact_var)))
        row = {
            "mode": "chebyshev",
            "n_train": args.n_train,
            "grid_side": args.grid_side,
            "n_test": X_test.shape[0],
            "n_features": n_features,
            "fit_time_s": fit_time,
            "n_nodes_per_dim": n_nodes,
            "n_nodes_total": int(n_nodes * n_nodes),
            "rmse": rmse,
            "rel_rmse": rel_rmse,
            "max_abs_err": max_abs,
            "speedup_vs_stochastic": float(stochastic_info["wall_time_s"] / interp_info["wall_time_s"]),
            **interp_info,
        }
        rows.append(row)
        print(
            f"chebyshev {n_nodes}x{n_nodes}: time={interp_info['wall_time_s']:.3f}s "
            f"node_eval={interp_info['node_eval_time_s']:.3f}s interp={interp_info['interp_time_s']:.3f}s "
            f"cg_total={int(interp_info['cg_iters_total'])} rel_rmse={rel_rmse:.4e} "
            f"speedup_vs_stoch={row['speedup_vs_stochastic']:.2f}x"
        )
        if rel_rmse <= stochastic_rel_rmse and best_match is None:
            best_match = row

    if best_match is None:
        print("no Chebyshev node count in the sweep matched the stochastic error target")
    else:
        print(
            f"matched stochastic error with {best_match['n_nodes_per_dim']}x{best_match['n_nodes_per_dim']} nodes: "
            f"time={best_match['wall_time_s']:.3f}s vs stochastic={stochastic_info['wall_time_s']:.3f}s "
            f"(speedup {best_match['speedup_vs_stochastic']:.2f}x)"
        )

    csv_path = Path(args.csv)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"results written to {csv_path}")


if __name__ == "__main__":
    main()
