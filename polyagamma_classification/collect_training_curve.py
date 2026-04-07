from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def _set_runtime_env() -> None:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def _load_synthetic_dataset(dataset_path: Path, n_points: int) -> dict[str, Any]:
    import numpy as np

    data = np.load(dataset_path)
    if n_points > data["x_train_pool"].shape[0]:
        raise ValueError(f"Requested n_points={n_points} but dataset only has {data['x_train_pool'].shape[0]} samples.")
    return {
        "x_train": data["x_train_pool"][:n_points].astype(np.float64),
        "y_train": data["y_train_pool"][:n_points].astype(np.float64),
    }


def run_pg(args: argparse.Namespace, data: dict[str, Any]) -> dict[str, Any]:
    from pg_classifier import PolyagammaGPNegativeBinomialRegressor

    reg = PolyagammaGPNegativeBinomialRegressor(
        total_count=args.total_count,
        learn_total_count=True,
        total_count_lr=0.05,
        total_count_update_frequency=1,
        total_count_quadrature_nodes=16,
        lengthscale_init=args.init_lengthscale,
        variance_init=args.init_variance,
        max_iter=args.pg_max_iter,
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
        store_history=True,
        verbose=0,
    )
    reg.fit(data["x_train"], data["y_train"])
    history = [
        {
            "step": -1,
            "elapsed_sec": 0.0,
            "lengthscale": float(args.init_lengthscale),
            "variance": float(args.init_variance),
            "total_count": float(args.total_count),
        }
    ]
    history.extend(
        [
        {
            "step": int(record["iter"]),
            "elapsed_sec": float(record.get("elapsed_sec", 0.0)),
            "lengthscale": float(record["lengthscale"]),
            "variance": float(record["variance"]),
            "total_count": float(record.get("total_count", args.total_count)),
        }
        for record in reg.history_
        ]
    )
    return {
        "method": "pg",
        "label": "PG",
        "history": history,
        "truth": {
            "lengthscale": float(args.truth_lengthscale),
            "variance": float(args.truth_variance),
            "total_count": float(args.truth_total_count),
        },
    }


def run_gpcounts(args: argparse.Namespace, data: dict[str, Any]) -> dict[str, Any]:
    import pandas as pd

    gpcounts_root = ROOT / "GPcounts"
    if str(gpcounts_root) not in sys.path:
        sys.path.insert(0, str(gpcounts_root))
    from GPcounts.GP_NB_ZINB import GP_nb_zinb

    x_train_df = pd.DataFrame(data["x_train"], columns=["x1", "x2"])
    y_train_df = pd.DataFrame([data["y_train"]], index=["synthetic_gene"])

    gp = GP_nb_zinb(
        X=x_train_df,
        y=y_train_df,
        sparse=args.sparse,
        M=args.inducing_points,
        monitor_history=True,
        allow_random_restarts=args.allow_random_restarts,
        safe_mode=args.safe_mode,
        save=False,
    )
    gp.scipy_max_iter = args.scipy_max_iter
    gp.initialize_hyper_parameters(
        length_scale=args.init_lengthscale,
        variance=args.init_variance,
        alpha=max(1e-12, 1.0 / float(args.total_count)),
        km=1.0,
    )

    gp.model_log_likelihood(
        lik_name="Negative_binomial",
        transform=True,
        txt="training_curve",
        kernel_type="RBF",
        models_number=1,
    )
    history = [
        {
            "step": int(record["step"]),
            "elapsed_sec": float(record.get("elapsed_sec", 0.0)),
            "lengthscale": float(record["lengthscale"]),
            "variance": float(record["variance"]),
            "total_count": float(record.get("total_count", args.total_count)),
        }
        for record in gp.training_history
    ]
    return {
        "method": "gpcounts",
        "label": f"GPcounts M={args.inducing_points}" if args.sparse else "GPcounts Full",
        "history": history,
        "truth": {
            "lengthscale": float(args.truth_lengthscale),
            "variance": float(args.truth_variance),
            "total_count": float(args.truth_total_count),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect PG vs GPcounts training history.")
    parser.add_argument("--method", choices=["pg", "gpcounts"], required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--n-points", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init-lengthscale", type=float, default=0.1)
    parser.add_argument("--init-variance", type=float, default=1.0)
    parser.add_argument("--total-count", type=float, default=3.0)
    parser.add_argument("--truth-lengthscale", type=float, default=0.1)
    parser.add_argument("--truth-variance", type=float, default=1.0)
    parser.add_argument("--truth-total-count", type=float, default=3.0)
    parser.add_argument("--pg-max-iter", type=int, default=50)
    parser.add_argument("--inducing-points", type=int, default=256)
    parser.add_argument("--sparse", dest="sparse", action="store_true")
    parser.add_argument("--full", dest="sparse", action="store_false")
    parser.set_defaults(sparse=True)
    parser.add_argument("--scipy-max-iter", type=int, default=50)
    parser.add_argument("--allow-random-restarts", action="store_true")
    parser.add_argument("--safe-mode", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _set_runtime_env()
    dataset = _load_synthetic_dataset(args.dataset_path, args.n_points)
    if args.method == "pg":
        payload = run_pg(args, dataset)
    else:
        payload = run_gpcounts(args, dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
