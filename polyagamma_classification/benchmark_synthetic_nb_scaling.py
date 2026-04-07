from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
DEFAULT_OUTPUT_DIR = ROOT / "data" / "synthetic_nb_scaling"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Synthetic learning-curve and scaling benchmark for the updated "
            "PG negative-binomial regressor versus GPcounts."
        )
    )
    parser.add_argument(
        "--child-method",
        choices=["pg_nb", "gpcounts_sparse", "gpcounts_full"],
        help="Internal mode: run one fit on a generated synthetic dataset and emit JSON.",
    )
    parser.add_argument(
        "--child-generate-dataset",
        action="store_true",
        help="Internal mode: generate one notebook-style synthetic dataset file.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        help="Internal mode: pre-generated synthetic dataset file to load.",
    )
    parser.add_argument(
        "--max-n",
        type=int,
        help="Internal mode: maximum pooled training size for dataset generation.",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["250", "500", "1000", "2000", "5000"],
        help="Training set sizes for parent mode.",
    )
    parser.add_argument(
        "--n-points",
        default="1000",
        help="Training size in child mode.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=["0", "1", "2"],
        help="Synthetic dataset seeds for parent mode.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pg-python",
        type=Path,
        default=PARENT / "venv" / "bin" / "python",
    )
    parser.add_argument(
        "--gpcounts-python",
        type=Path,
        default=Path("/opt/anaconda3/envs/ssm310/bin/python"),
    )
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument(
        "--gpcounts-sparse-m",
        type=int,
        default=128,
        help=(
            "Minimum inducing-point count for sparse GPcounts. "
            "The benchmark uses max(this value, 5%% of n)."
        ),
    )
    parser.add_argument(
        "--gpcounts-sparse-m-cap",
        type=int,
        default=256,
        help=(
            "Hard cap for sparse GPcounts inducing points. "
            "Use 0 to disable the cap."
        ),
    )
    parser.add_argument("--n-test", type=int, default=1024)
    parser.add_argument("--true-lengthscale", type=float, default=0.01)
    parser.add_argument("--true-variance", type=float, default=1.0)
    parser.add_argument("--true-total-count", type=float, default=3.0)
    parser.add_argument("--grid-size", type=int, default=24)
    parser.add_argument(
        "--rff-features",
        type=int,
        default=4096,
        help="Random Fourier feature count used for the synthetic latent draw.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args()


def _set_runtime_env() -> None:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def _generate_notebook_dataset_file(
    dataset_path: Path,
    *,
    max_n: int,
    seed: int,
    true_lengthscale: float,
    true_variance: float,
    true_total_count: float,
    grid_size: int,
) -> None:
    _set_runtime_env()
    import numpy as np
    import torch
    from torch.distributions import NegativeBinomial

    if str(PARENT) not in sys.path:
        sys.path.append(str(PARENT))
    from vanilla_gp_sampling import sample_gp_spectral_approx

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_train_pool = torch.rand(max_n, 2) * 2.0 - 1.0
    grid_1d = torch.linspace(-1.1, 1.1, grid_size)
    gx, gy = torch.meshgrid(grid_1d, grid_1d, indexing="ij")
    x_grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)

    x_all = torch.cat([x_train_pool, x_grid], dim=0)
    f_all = sample_gp_spectral_approx(
        x_all,
        num_samples=1,
        length_scale=true_lengthscale,
        variance=true_variance,
        spectral_eps=1e-4,
        trunc_eps=1e-4,
        nufft_eps=1e-7,
        seed=12,
    )
    y_all = NegativeBinomial(
        total_count=torch.tensor(true_total_count, dtype=torch.float64),
        logits=f_all,
    ).sample()

    y_train_pool = y_all[:max_n].cpu().numpy().astype(np.float64)
    f_train_pool = f_all[:max_n].cpu().numpy().astype(np.float64)
    f_grid = f_all[max_n:].cpu().numpy().astype(np.float64)
    y_grid = y_all[max_n:].cpu().numpy().astype(np.float64)
    mean_grid = (true_total_count * torch.exp(f_all[max_n:])).cpu().numpy().astype(np.float64)

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dataset_path,
        x_train_pool=x_train_pool.cpu().numpy().astype(np.float64),
        y_train_pool=y_train_pool,
        true_logits_train_pool=f_train_pool,
        x_test=x_grid.cpu().numpy().astype(np.float64),
        y_test=y_grid,
        true_logits_test=f_grid,
        true_mean_test=mean_grid,
        truth_lengthscale=np.array([true_lengthscale], dtype=np.float64),
        truth_variance=np.array([true_variance], dtype=np.float64),
        truth_total_count=np.array([true_total_count], dtype=np.float64),
        grid_1d=grid_1d.cpu().numpy().astype(np.float64),
    )


def _load_dataset_from_file(dataset_path: Path, n_train: int) -> dict[str, Any]:
    import numpy as np

    data = np.load(dataset_path)
    if n_train > data["x_train_pool"].shape[0]:
        raise ValueError(
            f"Requested n_train={n_train} but dataset only has {data['x_train_pool'].shape[0]} training points."
        )
    return {
        "x_train": data["x_train_pool"][:n_train].astype(np.float64),
        "y_train": data["y_train_pool"][:n_train].astype(np.float64),
        "true_logits_train": data["true_logits_train_pool"][:n_train].astype(np.float64),
        "x_test": data["x_test"].astype(np.float64),
        "y_test": data["y_test"].astype(np.float64),
        "true_logits_test": data["true_logits_test"].astype(np.float64),
        "true_mean_test": data["true_mean_test"].astype(np.float64),
        "truth": {
            "lengthscale": float(data["truth_lengthscale"][0]),
            "variance": float(data["truth_variance"][0]),
            "total_count": float(data["truth_total_count"][0]),
            "alpha": float(1.0 / data["truth_total_count"][0]),
        },
    }


def _rff_gp_sample(
    x: Any,
    *,
    seed: int,
    lengthscale: float,
    variance: float,
    n_features: int,
) -> Any:
    import numpy as np

    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    dimension = x.shape[1]
    omega = rng.normal(loc=0.0, scale=1.0 / float(lengthscale), size=(n_features, dimension))
    phase = rng.uniform(0.0, 2.0 * np.pi, size=n_features)
    weights = rng.normal(size=n_features)
    features = math.sqrt(2.0 * float(variance) / n_features) * np.cos(x @ omega.T + phase[None, :])
    return features @ weights


def _generate_dataset(
    *,
    n_train: int,
    n_test: int,
    seed: int,
    true_lengthscale: float,
    true_variance: float,
    true_total_count: float,
    rff_features: int,
) -> dict[str, Any]:
    import numpy as np

    rng = np.random.default_rng(seed)
    x_train = rng.uniform(-1.0, 1.0, size=(n_train, 2))
    x_test = rng.uniform(-1.0, 1.0, size=(n_test, 2))
    x_all = np.vstack([x_train, x_test])
    f_all = _rff_gp_sample(
        x_all,
        seed=seed + 10_000,
        lengthscale=true_lengthscale,
        variance=true_variance,
        n_features=rff_features,
    )

    logits_train = f_all[:n_train]
    logits_test = f_all[n_train:]
    p_train = 1.0 / (1.0 + np.exp(logits_train))
    p_test = 1.0 / (1.0 + np.exp(logits_test))

    y_train = rng.negative_binomial(true_total_count, p_train).astype(np.float64)
    y_test = rng.negative_binomial(true_total_count, p_test).astype(np.float64)
    mean_train = true_total_count * np.exp(logits_train)
    mean_test = true_total_count * np.exp(logits_test)

    return {
        "x_train": x_train,
        "y_train": y_train,
        "true_logits_train": logits_train,
        "true_mean_train": mean_train,
        "x_test": x_test,
        "y_test": y_test,
        "true_logits_test": logits_test,
        "true_mean_test": mean_test,
        "truth": {
            "lengthscale": float(true_lengthscale),
            "variance": float(true_variance),
            "total_count": float(true_total_count),
            "alpha": float(1.0 / true_total_count),
        },
    }


def _prepare_dataset(args: argparse.Namespace) -> dict[str, Any]:
    n_train = int(args.n_points)
    if args.dataset_path is not None:
        return _load_dataset_from_file(args.dataset_path, n_train)
    return _generate_dataset(
        n_train=n_train,
        n_test=args.n_test,
        seed=args.seed,
        true_lengthscale=args.true_lengthscale,
        true_variance=args.true_variance,
        true_total_count=args.true_total_count,
        rff_features=args.rff_features,
    )


def _resolve_sparse_m(n_train: int, requested_min: int, requested_cap: int) -> int:
    default_m = max(1, int(round(0.05 * n_train)))
    if requested_min <= 0:
        resolved = default_m
    else:
        resolved = max(int(requested_min), default_m)
    if requested_cap > 0:
        resolved = min(resolved, int(requested_cap))
    return max(1, resolved)


def _nb_logpmf(y: Any, mean_count: Any, total_count: float) -> Any:
    import numpy as np
    import scipy.special as sps

    y = np.asarray(y, dtype=np.float64)
    mu = np.clip(np.asarray(mean_count, dtype=np.float64), 1e-10, None)
    r = float(max(total_count, 1e-10))
    return (
        sps.gammaln(y + r)
        - sps.gammaln(r)
        - sps.gammaln(y + 1.0)
        + r * (math.log(r) - np.log(r + mu))
        + y * (np.log(mu) - np.log(r + mu))
    )


def _safe_corrcoef(x: Any, y: Any) -> float:
    import numpy as np

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size == 0 or y.size == 0:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _common_metrics(
    *,
    truth: dict[str, float],
    true_mean_test: Any,
    y_test: Any,
    mean_test_hat: Any,
    lengthscale_hat: float,
    variance_hat: float,
    total_count_hat: float,
) -> dict[str, Any]:
    import numpy as np

    true_mean_test = np.asarray(true_mean_test, dtype=np.float64)
    mean_test_hat = np.asarray(mean_test_hat, dtype=np.float64)
    y_test = np.asarray(y_test, dtype=np.float64)
    safe_true_mean = np.clip(true_mean_test, 1e-10, None)
    safe_pred_mean = np.clip(mean_test_hat, 1e-10, None)
    true_log_mean = np.log(safe_true_mean)
    pred_log_mean = np.log(safe_pred_mean)
    true_r = float(max(truth["total_count"], 1e-10))
    pred_r = float(max(total_count_hat, 1e-10))
    true_zero_prob = (true_r / (true_r + safe_true_mean)) ** true_r
    pred_zero_prob = (pred_r / (pred_r + safe_pred_mean)) ** pred_r

    return {
        "test_mean_mae": float(np.mean(np.abs(mean_test_hat - true_mean_test))),
        "test_mean_rmse": float(np.sqrt(np.mean((mean_test_hat - true_mean_test) ** 2))),
        "test_mean_correlation": _safe_corrcoef(true_mean_test, mean_test_hat),
        "test_log_mean_rmse": float(np.sqrt(np.mean((pred_log_mean - true_log_mean) ** 2))),
        "test_log_mean_correlation": _safe_corrcoef(true_log_mean, pred_log_mean),
        "test_zero_prob_mae": float(np.mean(np.abs(pred_zero_prob - true_zero_prob))),
        "test_zero_prob_correlation": _safe_corrcoef(true_zero_prob, pred_zero_prob),
        "test_count_nll_per_point": float(-np.mean(_nb_logpmf(y_test, mean_test_hat, total_count_hat))),
        "lengthscale_hat": float(lengthscale_hat),
        "variance_hat": float(variance_hat),
        "total_count_hat": float(total_count_hat),
        "lengthscale_rel_error": float(abs(lengthscale_hat - truth["lengthscale"]) / truth["lengthscale"]),
        "variance_rel_error": float(abs(variance_hat - truth["variance"]) / truth["variance"]),
        "total_count_rel_error": float(abs(total_count_hat - truth["total_count"]) / truth["total_count"]),
    }


def run_pg_child(args: argparse.Namespace) -> dict[str, Any]:
    _set_runtime_env()

    from pg_classifier import PolyagammaGPNegativeBinomialRegressor

    n_train = int(args.n_points)
    data = _prepare_dataset(args)

    reg = PolyagammaGPNegativeBinomialRegressor(
        total_count=5.25,
        learn_total_count=True,
        total_count_lr=0.05,
        total_count_update_frequency=1,
        total_count_quadrature_nodes=16,
        lengthscale_init=0.30,
        variance_init=1.00,
        max_iter=50,
        e_step_iters=1,
        final_e_step_iters=2,
        rho0=1.0,
        gamma=1e-3,
        lr=0.05,
        n_e_probes=1,
        n_m_probes=1,
        cg_tol=1e-6,
        nufft_eps=1e-4,
        spectral_eps=1e-4,
        trunc_eps=1e-4,
        prediction_batch_size=96,
        predictive_variance_method="chebyshev",
        predictive_variance_chebyshev_nodes=7,
        use_exact_weighted_toeplitz_operator=True,
        random_state=args.seed,
        device="cpu",
        store_history=False,
        verbose=0,
    )

    started = time.perf_counter()
    reg.fit(data["x_train"], data["y_train"])
    fit_time = time.perf_counter() - started
    mean_test_hat = reg.predict_response_mean(data["x_test"])

    row = {
        "method": "pg_nb",
        "status": "ok",
        "seed": int(args.seed),
        "n_points": int(n_train),
        "runtime_sec": float(fit_time),
        "train_mean_count_mae": float(reg.training_mean_absolute_error_),
    }
    row.update(
        _common_metrics(
            truth=data["truth"],
            true_mean_test=data["true_mean_test"],
            y_test=data["y_test"],
            mean_test_hat=mean_test_hat,
            lengthscale_hat=reg.lengthscale_,
            variance_hat=reg.variance_,
            total_count_hat=reg.total_count_,
        )
    )
    return row


def run_gpcounts_child(args: argparse.Namespace, *, sparse: bool) -> dict[str, Any]:
    _set_runtime_env()
    import numpy as np
    import pandas as pd

    gpcounts_root = ROOT / "GPcounts"
    if str(gpcounts_root) not in sys.path:
        sys.path.insert(0, str(gpcounts_root))
    from GPcounts.GP_NB_ZINB import GP_nb_zinb

    n_train = int(args.n_points)
    data = _prepare_dataset(args)

    x_train_df = pd.DataFrame(data["x_train"], columns=["x1", "x2"])
    y_train_df = pd.DataFrame([data["y_train"]], index=["synthetic_gene"])
    gp = GP_nb_zinb(
        X=x_train_df,
        y=y_train_df,
        sparse=sparse,
        M=(
            _resolve_sparse_m(
                n_train,
                int(args.gpcounts_sparse_m),
                int(args.gpcounts_sparse_m_cap),
            )
            if sparse
            else 0
        ),
        safe_mode=False,
        scale=None,
        save=False,
    )

    started = time.perf_counter()
    _ = gp.model_log_likelihood(
        lik_name="Negative_binomial",
        transform=True,
        txt="synthetic_scaling",
        kernel_type="RBF",
        models_number=1,
    )
    fit_time = time.perf_counter() - started

    f_mean_test, f_var_test = gp.model.predict_f(data["x_test"])
    mean_test_hat = np.exp(
        np.asarray(f_mean_test.numpy()).reshape(-1)
        + 0.5 * np.asarray(f_var_test.numpy()).reshape(-1)
    )
    alpha_hat = float(gp.model.likelihood.alpha.numpy())
    total_count_hat = float(1.0 / max(alpha_hat, 1e-12))
    kernel = gp.model.kernel
    lengthscale_hat = float(np.asarray(kernel.lengthscales.numpy()).reshape(-1)[0])
    variance_hat = float(np.asarray(kernel.variance.numpy()).reshape(-1)[0])

    row = {
        "method": "gpcounts_sparse" if sparse else "gpcounts_full",
        "status": "ok",
        "seed": int(args.seed),
        "n_points": int(n_train),
        "runtime_sec": float(fit_time),
        "alpha_hat": alpha_hat,
        "inducing_points": 0,
    }
    if sparse:
        inducing = getattr(gp.model, "inducing_variable", None)
        if inducing is not None and getattr(inducing, "Z", None) is not None:
            row["inducing_points"] = int(np.asarray(inducing.Z.numpy()).shape[0])

    row.update(
        _common_metrics(
            truth=data["truth"],
            true_mean_test=data["true_mean_test"],
            y_test=data["y_test"],
            mean_test_hat=mean_test_hat,
            lengthscale_hat=lengthscale_hat,
            variance_hat=variance_hat,
            total_count_hat=total_count_hat,
        )
    )
    return row


def _run_child(args: argparse.Namespace) -> int:
    if args.child_method == "pg_nb":
        row = run_pg_child(args)
    elif args.child_method == "gpcounts_sparse":
        row = run_gpcounts_child(args, sparse=True)
    else:
        row = run_gpcounts_child(args, sparse=False)
    print(json.dumps(row, sort_keys=True))
    return 0


def _python_for_method(method: str, args: argparse.Namespace) -> Path:
    if method == "pg_nb":
        return args.pg_python
    return args.gpcounts_python


def _launch_child(
    method: str,
    n_points: int,
    seed: int,
    args: argparse.Namespace,
    *,
    dataset_path: Path | None = None,
) -> dict[str, Any]:
    python = _python_for_method(method, args)
    cmd = [
        str(python),
        str(Path(__file__).resolve()),
        "--child-method",
        method,
        "--n-points",
        str(n_points),
        "--seed",
        str(seed),
        "--n-test",
        str(args.n_test),
        "--true-lengthscale",
        str(args.true_lengthscale),
        "--true-variance",
        str(args.true_variance),
        "--true-total-count",
        str(args.true_total_count),
        "--grid-size",
        str(args.grid_size),
        "--rff-features",
        str(args.rff_features),
        "--gpcounts-sparse-m",
        str(args.gpcounts_sparse_m),
        "--gpcounts-sparse-m-cap",
        str(args.gpcounts_sparse_m_cap),
    ]
    if dataset_path is not None:
        cmd.extend(["--dataset-path", str(dataset_path)])
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    env.setdefault("PYTHONUNBUFFERED", "1")

    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            text=True,
            capture_output=True,
            timeout=args.timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "method": method,
            "seed": int(seed),
            "n_points": int(n_points),
            "status": "timeout",
        }
    wall_time = time.perf_counter() - started

    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    if proc.returncode != 0:
        return {
            "method": method,
            "seed": int(seed),
            "n_points": int(n_points),
            "status": "error",
            "returncode": int(proc.returncode),
            "stdout_tail": stdout.splitlines()[-20:],
            "stderr_tail": stderr.splitlines()[-20:],
        }
    lines = [line for line in stdout.splitlines() if line.strip()]
    payload = json.loads(lines[-1])
    payload["subprocess_wall_time_sec"] = float(wall_time)
    payload["python"] = str(python)
    return payload


def _generate_dataset_via_pg_python(
    dataset_path: Path,
    *,
    max_n: int,
    seed: int,
    args: argparse.Namespace,
) -> None:
    cmd = [
        str(args.pg_python),
        str(Path(__file__).resolve()),
        "--child-generate-dataset",
        "--dataset-path",
        str(dataset_path),
        "--max-n",
        str(max_n),
        "--seed",
        str(seed),
        "--true-lengthscale",
        str(args.true_lengthscale),
        "--true-variance",
        str(args.true_variance),
        "--true-total-count",
        str(args.true_total_count),
        "--grid-size",
        str(args.grid_size),
    ]
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=args.timeout_sec,
        check=False,
    )
    if proc.returncode != 0:
        stdout = proc.stdout.strip().splitlines()[-20:]
        stderr = proc.stderr.strip().splitlines()[-20:]
        raise RuntimeError(
            "Dataset generation failed "
            f"(returncode={proc.returncode}, stdout_tail={stdout}, stderr_tail={stderr})"
        )
    if not dataset_path.exists():
        raise RuntimeError(f"Dataset generation reported success but file is missing: {dataset_path}")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import numpy as np

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in ok_rows:
        groups.setdefault((str(row["method"]), int(row["n_points"])), []).append(row)

    summary: list[dict[str, Any]] = []
    metrics = [
        "runtime_sec",
        "test_mean_mae",
        "test_mean_rmse",
        "test_mean_correlation",
        "test_log_mean_rmse",
        "test_log_mean_correlation",
        "test_zero_prob_mae",
        "test_zero_prob_correlation",
        "test_count_nll_per_point",
        "lengthscale_hat",
        "variance_hat",
        "total_count_hat",
        "lengthscale_rel_error",
        "variance_rel_error",
        "total_count_rel_error",
    ]

    for (method, n_points), rows_group in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        row = {
            "method": method,
            "n_points": int(n_points),
            "status": "ok",
            "num_seeds": len(rows_group),
        }
        for metric in metrics:
            values = np.array([float(r[metric]) for r in rows_group], dtype=np.float64)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
        if method == "gpcounts_sparse":
            inducing = np.array([float(r["inducing_points"]) for r in rows_group], dtype=np.float64)
            row["inducing_points_mean"] = float(inducing.mean())
        summary.append(row)
    return summary


def _make_plots(summary_rows: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    methods = sorted({row["method"] for row in summary_rows})
    colors = {
        "pg_nb": "#1982c4",
        "gpcounts_sparse": "#ff595e",
        "gpcounts_full": "#6a4c93",
    }

    fig, axes = plt.subplots(1, 4, figsize=(21, 4.5), constrained_layout=True)
    for method in methods:
        rows = sorted(
            [row for row in summary_rows if row["method"] == method],
            key=lambda row: row["n_points"],
        )
        x = np.array([row["n_points"] for row in rows], dtype=np.float64)
        runtime = np.array([row["runtime_sec_mean"] for row in rows], dtype=np.float64)
        mae = np.array([row["test_mean_mae_mean"] for row in rows], dtype=np.float64)
        zero_prob_mae = np.array([row["test_zero_prob_mae_mean"] for row in rows], dtype=np.float64)
        ls = np.array([row["lengthscale_hat_mean"] for row in rows], dtype=np.float64)
        rhat = np.array([row["total_count_hat_mean"] for row in rows], dtype=np.float64)

        axes[0].plot(x, runtime, marker="o", linewidth=2, color=colors.get(method, None), label=method)
        axes[1].plot(x, mae, marker="o", linewidth=2, color=colors.get(method, None), label=method)
        axes[2].plot(x, zero_prob_mae, marker="o", linewidth=2, color=colors.get(method, None), label=method)
        axes[3].plot(x, ls, marker="o", linewidth=2, color=colors.get(method, None), label=f"{method} lengthscale")
        axes[3].plot(x, rhat, marker="s", linewidth=2, linestyle="--", color=colors.get(method, None), alpha=0.75, label=f"{method} r")

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_title("Scaling")
    axes[0].set_xlabel("n")
    axes[0].set_ylabel("Fit time (s)")

    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_title("Learning Curve")
    axes[1].set_xlabel("n")
    axes[1].set_ylabel("Test mean-count MAE")

    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_title("Zero-Prob Sanity")
    axes[2].set_xlabel("n")
    axes[2].set_ylabel("Test zero-prob MAE")

    axes[3].set_xscale("log")
    axes[3].set_title("Hyperparameter Recovery")
    axes[3].set_xlabel("n")
    axes[3].set_ylabel("Estimated value")

    true_lengthscale = summary_rows[0].get("truth_lengthscale", 0.01)
    true_total_count = summary_rows[0].get("truth_total_count", 3.0)
    axes[3].axhline(true_lengthscale, color="black", linewidth=1.5, linestyle=":", label="true lengthscale")
    axes[3].axhline(true_total_count, color="black", linewidth=1.5, linestyle="-.", label="true r")

    for ax in axes:
        ax.legend(loc="best", fontsize=8)

    path = output_dir / "synthetic_nb_scaling_summary.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _make_sanity_plots(summary_rows: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    methods = sorted({row["method"] for row in summary_rows})
    colors = {
        "pg_nb": "#1982c4",
        "gpcounts_sparse": "#ff595e",
        "gpcounts_full": "#6a4c93",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for method in methods:
        rows = sorted(
            [row for row in summary_rows if row["method"] == method],
            key=lambda row: row["n_points"],
        )
        x = np.array([row["n_points"] for row in rows], dtype=np.float64)
        log_mean_rmse = np.array([row["test_log_mean_rmse_mean"] for row in rows], dtype=np.float64)
        nll = np.array([row["test_count_nll_per_point_mean"] for row in rows], dtype=np.float64)

        axes[0].plot(x, log_mean_rmse, marker="o", linewidth=2, color=colors.get(method, None), label=method)
        axes[1].plot(x, nll, marker="o", linewidth=2, color=colors.get(method, None), label=method)

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_title("Log-Mean Truth Recovery")
    axes[0].set_xlabel("n")
    axes[0].set_ylabel("Test log-mean RMSE")

    axes[1].set_xscale("log")
    axes[1].set_title("Predictive Count NLL")
    axes[1].set_xlabel("n")
    axes[1].set_ylabel("Test NB NLL / point")

    for ax in axes:
        ax.legend(loc="best", fontsize=8)

    path = output_dir / "synthetic_nb_scaling_sanity.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _run_parent(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    methods = ["pg_nb", "gpcounts_sparse"]
    sizes = [int(size) for size in args.sizes]
    seeds = [int(seed) for seed in args.seeds]
    max_n = max(sizes)

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        dataset_path = args.output_dir / "datasets" / f"seed_{seed}_nmax_{max_n}.npz"
        if not dataset_path.exists():
            print(f"generating dataset seed={seed} n_max={max_n}", flush=True)
            _generate_dataset_via_pg_python(
                dataset_path,
                max_n=max_n,
                seed=seed,
                args=args,
            )
        for n_points in sizes:
            for method in methods:
                print(f"running method={method} seed={seed} n={n_points}", flush=True)
                row = _launch_child(method, n_points, seed, args, dataset_path=dataset_path)
                row["truth_lengthscale"] = float(args.true_lengthscale)
                row["truth_variance"] = float(args.true_variance)
                row["truth_total_count"] = float(args.true_total_count)
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)

    summary_rows = _aggregate_rows(rows)
    for row in summary_rows:
        row["truth_lengthscale"] = float(args.true_lengthscale)
        row["truth_variance"] = float(args.true_variance)
        row["truth_total_count"] = float(args.true_total_count)

    raw_csv = args.output_dir / "benchmark_raw.csv"
    raw_json = args.output_dir / "benchmark_raw.json"
    summary_csv = args.output_dir / "benchmark_summary.csv"
    summary_json = args.output_dir / "benchmark_summary.json"
    _write_csv(raw_csv, rows)
    raw_json.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    _write_csv(summary_csv, summary_rows)
    summary_json.write_text(json.dumps(summary_rows, indent=2, sort_keys=True) + "\n")
    _make_plots(summary_rows, args.output_dir)
    _make_sanity_plots(summary_rows, args.output_dir)

    print(f"wrote {raw_csv}")
    print(f"wrote {summary_csv}")
    print(f"wrote {args.output_dir / 'synthetic_nb_scaling_summary.png'}")
    print(f"wrote {args.output_dir / 'synthetic_nb_scaling_sanity.png'}")
    return 0


def main() -> int:
    args = parse_args()
    if args.child_generate_dataset:
        if args.dataset_path is None:
            raise ValueError("--dataset-path is required with --child-generate-dataset.")
        max_n = int(args.max_n) if args.max_n is not None else int(args.n_points)
        _generate_notebook_dataset_file(
            args.dataset_path,
            max_n=max_n,
            seed=args.seed,
            true_lengthscale=args.true_lengthscale,
            true_variance=args.true_variance,
            true_total_count=args.true_total_count,
            grid_size=args.grid_size,
        )
        return 0
    if args.child_method is not None:
        return _run_child(args)
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
