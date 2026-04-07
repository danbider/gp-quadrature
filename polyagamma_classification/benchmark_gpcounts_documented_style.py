from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent


def _set_runtime_env() -> None:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def _load_dataset(dataset_path: Path, n_points: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    from benchmark_synthetic_nb_scaling import _load_dataset_from_file

    data = _load_dataset_from_file(dataset_path, n_points)
    cell_names = [f"cell_{i}" for i in range(n_points)]
    x_df = pd.DataFrame(data["x_train"], columns=["x", "y"], index=cell_names)
    y_df = pd.DataFrame([data["y_train"]], index=["synthetic_gene"], columns=cell_names)
    return x_df, y_df


def _fit_gpcounts(
    x_df: pd.DataFrame,
    y_df: pd.DataFrame,
    *,
    sparse: bool,
    m: int,
    safe_mode: bool,
) -> dict[str, float | int | str]:
    gpcounts_root = ROOT / "GPcounts"
    if str(gpcounts_root) not in sys.path:
        sys.path.insert(0, str(gpcounts_root))
    from GPcounts.GP_NB_ZINB import GP_nb_zinb

    gp = GP_nb_zinb(
        x_df,
        y_df,
        sparse=sparse,
        M=m,
        safe_mode=safe_mode,
        save=False,
    )
    started = time.perf_counter()
    ll = gp.model_log_likelihood(
        "Negative_binomial",
        True,
        txt="documented_style",
        kernel_type="RBF",
        models_number=1,
    )
    runtime = time.perf_counter() - started
    if str(ll) == "nan":
        return {
            "status": "nan",
            "runtime_sec": runtime,
            "sparse": sparse,
            "M": int(m),
        }
    kernel = gp.model.kernel
    alpha = float(gp.model.likelihood.alpha.numpy())
    return {
        "status": "ok",
        "runtime_sec": runtime,
        "sparse": sparse,
        "M": int(m),
        "log_likelihood": float(ll),
        "lengthscale": float(kernel.lengthscales.numpy().reshape(-1)[0]),
        "variance": float(kernel.variance.numpy().reshape(-1)[0]),
        "total_count": float(1.0 / max(alpha, 1e-12)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPcounts comparison using documented-style settings.")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--n-points", type=int, default=5000)
    parser.add_argument("--safe-mode", action="store_true", default=True)
    parser.add_argument("--sparse-m", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    _set_runtime_env()
    args = parse_args()
    x_df, y_df = _load_dataset(args.dataset_path, args.n_points)
    rows: list[dict[str, float | int | str]] = []
    rows.append(_fit_gpcounts(x_df, y_df, sparse=False, m=0, safe_mode=args.safe_mode))
    for m in args.sparse_m:
        rows.append(_fit_gpcounts(x_df, y_df, sparse=True, m=int(m), safe_mode=args.safe_mode))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
