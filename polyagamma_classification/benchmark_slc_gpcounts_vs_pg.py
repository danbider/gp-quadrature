from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
DEFAULT_OUTPUT_DIR = ROOT / "data" / "slc_gpcounts_speed_benchmark"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the local PG negative-binomial regressor against GPcounts "
            "on the SLC17A7 spatial transcriptomics example."
        )
    )
    parser.add_argument(
        "--child-method",
        choices=["pg_nb", "gpcounts_full", "gpcounts_sparse"],
        help="Internal mode: run a single benchmark job and emit one JSON record.",
    )
    parser.add_argument(
        "--n-points",
        default="1000",
        help="Number of training points to sample, or 'full' for all SLC spots.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["1000", "2500", "5000"],
        help="Benchmark sizes to run in parent mode.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["pg_nb", "gpcounts_full", "gpcounts_sparse"],
        choices=["pg_nb", "gpcounts_full", "gpcounts_sparse"],
        help="Methods to benchmark in parent mode.",
    )
    parser.add_argument(
        "--pg-python",
        type=Path,
        default=PARENT / "venv" / "bin" / "python",
        help="Python interpreter used for the local PG benchmark.",
    )
    parser.add_argument(
        "--gpcounts-python",
        type=Path,
        default=Path("/opt/anaconda3/envs/ssm310/bin/python"),
        help="Python interpreter used for GPcounts.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=1800,
        help="Per-job timeout for parent-mode subprocess launches.",
    )
    parser.add_argument(
        "--gpcounts-sparse-m",
        type=int,
        default=0,
        help=(
            "Number of GPcounts inducing points in sparse mode. "
            "Use 0 to let GPcounts choose its default 5%% of n."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where parent-mode CSV/JSON summaries are written.",
    )
    return parser.parse_args()


def _normalize_size_arg(size_arg: str) -> str:
    if size_arg.lower() == "full":
        return "full"
    return str(int(size_arg))


def _set_runtime_env() -> None:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def _load_slc_subset(n_points: str, seed: int) -> tuple[Any, Any]:
    _set_runtime_env()
    import numpy as np
    import torch

    x = torch.load(PARENT / "x.pt", map_location="cpu").numpy().astype("float64")
    y_log = torch.load(PARENT / "y_slc17a7.pt", map_location="cpu").numpy().astype("float64")
    counts = np.rint(np.expm1(y_log)).astype("int64")

    if n_points != "full":
        n = int(n_points)
        if n <= 0:
            raise ValueError("n_points must be positive.")
        if n > x.shape[0]:
            raise ValueError(f"Requested n_points={n} but only {x.shape[0]} spots are available.")
        rng = np.random.default_rng(seed)
        subset_idx = np.sort(rng.choice(x.shape[0], size=n, replace=False))
        x = x[subset_idx]
        counts = counts[subset_idx]

    return x, counts


def run_pg_nb_child(n_points: str, seed: int) -> dict[str, Any]:
    _set_runtime_env()
    import numpy as np

    from pg_classifier import PolyagammaGPNegativeBinomialRegressor

    x, counts = _load_slc_subset(n_points=n_points, seed=seed)
    started = time.perf_counter()
    reg = PolyagammaGPNegativeBinomialRegressor(
        total_count=1.0,
        learn_total_count=True,
        total_count_lr=0.05,
        total_count_update_frequency=1,
        total_count_quadrature_nodes=16,
        lengthscale_init=0.20,
        variance_init=1.0,
        max_iter=50,
        e_step_iters=1,
        final_e_step_iters=2,
        rho0=0.7,
        gamma=1e-3,
        lr=0.05,
        n_e_probes=4,
        n_m_probes=8,
        cg_tol=1e-6,
        nufft_eps=1e-7,
        spectral_eps=1e-4,
        trunc_eps=1e-4,
        prediction_batch_size=256,
        predictive_variance_method="chebyshev",
        predictive_variance_chebyshev_nodes=7,
        use_exact_weighted_toeplitz_operator=True,
        random_state=seed,
        device="cpu",
        store_history=False,
        verbose=0,
    )
    reg.fit(x, counts)
    fit_time = time.perf_counter() - started
    mean_count = reg.predict_response_mean(x)

    return {
        "method": "pg_nb",
        "status": "ok",
        "n_points": int(x.shape[0]),
        "seed": int(seed),
        "runtime_sec": float(fit_time),
        "training_mae": float(np.mean(np.abs(mean_count - counts))),
        "training_mae_reported": float(reg.training_mean_absolute_error_),
        "lengthscale": float(reg.lengthscale_),
        "variance": float(reg.variance_),
        "total_count": float(reg.total_count_),
        "mean_count": float(np.mean(counts)),
        "zero_fraction": float(np.mean(counts == 0)),
    }


def run_gpcounts_child(n_points: str, seed: int, sparse: bool, sparse_m: int) -> dict[str, Any]:
    _set_runtime_env()
    import numpy as np
    import pandas as pd

    gpcounts_root = ROOT / "GPcounts"
    if str(gpcounts_root) not in sys.path:
        sys.path.insert(0, str(gpcounts_root))

    from GPcounts.GP_NB_ZINB import GP_nb_zinb

    x, counts = _load_slc_subset(n_points=n_points, seed=seed)
    x_df = pd.DataFrame(x, columns=["x1", "x2"])
    y_df = pd.DataFrame([counts], index=["SLC17A7"])

    gp = GP_nb_zinb(
        X=x_df,
        y=y_df,
        sparse=sparse,
        M=int(sparse_m),
        safe_mode=False,
        scale=None,
        save=False,
    )
    started = time.perf_counter()
    log_likelihood = gp.model_log_likelihood(
        lik_name="Negative_binomial",
        transform=True,
        txt="slc17a7_benchmark",
        kernel_type="RBF",
        models_number=1,
    )
    fit_time = time.perf_counter() - started

    f_mean, f_var = gp.model.predict_f(gp.X)
    mean_count = np.exp(
        np.asarray(f_mean.numpy()).reshape(-1) + 0.5 * np.asarray(f_var.numpy()).reshape(-1)
    )
    kernel = gp.model.kernel
    lengthscale = getattr(kernel, "lengthscales", None)
    variance = getattr(kernel, "variance", None)
    inducing_count = 0
    if sparse:
        inducing = getattr(gp.model, "inducing_variable", None)
        if inducing is not None and getattr(inducing, "Z", None) is not None:
            inducing_count = int(np.asarray(inducing.Z.numpy()).shape[0])

    return {
        "method": "gpcounts_sparse" if sparse else "gpcounts_full",
        "status": "ok",
        "n_points": int(x.shape[0]),
        "seed": int(seed),
        "runtime_sec": float(fit_time),
        "training_mae": float(np.mean(np.abs(mean_count - counts))),
        "log_posterior_density": float(log_likelihood),
        "lengthscale": float(np.asarray(lengthscale.numpy()).reshape(-1)[0]) if lengthscale is not None else None,
        "variance": float(np.asarray(variance.numpy()).reshape(-1)[0]) if variance is not None else None,
        "alpha": float(gp.model.likelihood.alpha.numpy()),
        "inducing_points": int(inducing_count),
        "mean_count": float(np.mean(counts)),
        "zero_fraction": float(np.mean(counts == 0)),
    }


def _run_child_from_args(args: argparse.Namespace) -> int:
    size = _normalize_size_arg(args.n_points)
    if args.child_method == "pg_nb":
        result = run_pg_nb_child(n_points=size, seed=args.seed)
    elif args.child_method == "gpcounts_full":
        result = run_gpcounts_child(
            n_points=size,
            seed=args.seed,
            sparse=False,
            sparse_m=0,
        )
    else:
        result = run_gpcounts_child(
            n_points=size,
            seed=args.seed,
            sparse=True,
            sparse_m=args.gpcounts_sparse_m,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


def _python_for_method(method: str, args: argparse.Namespace) -> Path:
    if method == "pg_nb":
        return args.pg_python
    return args.gpcounts_python


def _launch_child(method: str, size: str, args: argparse.Namespace) -> dict[str, Any]:
    python = _python_for_method(method, args)
    if not python.exists():
        return {
            "method": method,
            "n_points": size,
            "status": "missing_python",
            "python": str(python),
        }

    cmd = [
        str(python),
        str(Path(__file__).resolve()),
        "--child-method",
        method,
        "--n-points",
        size,
        "--seed",
        str(args.seed),
        "--gpcounts-sparse-m",
        str(args.gpcounts_sparse_m),
    ]
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
            "n_points": size,
            "status": "timeout",
            "timeout_sec": int(args.timeout_sec),
            "python": str(python),
        }

    wall_time = time.perf_counter() - started
    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()

    if proc.returncode != 0:
        return {
            "method": method,
            "n_points": size,
            "status": "error",
            "returncode": int(proc.returncode),
            "wall_time_sec": float(wall_time),
            "python": str(python),
            "stdout_tail": stdout.splitlines()[-20:],
            "stderr_tail": stderr.splitlines()[-20:],
        }

    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        return {
            "method": method,
            "n_points": size,
            "status": "empty_output",
            "wall_time_sec": float(wall_time),
            "python": str(python),
            "stderr_tail": stderr.splitlines()[-20:],
        }

    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError:
        return {
            "method": method,
            "n_points": size,
            "status": "bad_json",
            "wall_time_sec": float(wall_time),
            "python": str(python),
            "stdout_tail": lines[-20:],
            "stderr_tail": stderr.splitlines()[-20:],
        }

    payload["python"] = str(python)
    payload["subprocess_wall_time_sec"] = float(wall_time)
    return payload


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def _run_parent(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    sizes = [_normalize_size_arg(size) for size in args.sizes]

    for size in sizes:
        for method in args.methods:
            print(f"running method={method} n_points={size}", flush=True)
            row = _launch_child(method=method, size=size, args=args)
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    csv_path = args.output_dir / "benchmark_results.csv"
    json_path = args.output_dir / "benchmark_results.json"
    _write_rows_csv(csv_path, rows)
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    return 0


def main() -> int:
    args = parse_args()
    if args.child_method is not None:
        return _run_child_from_args(args)
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
