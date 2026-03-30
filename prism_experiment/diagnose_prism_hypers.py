#!/usr/bin/env python3
"""
Diagnose whether large-N PRISM hyperparameter drift is driven by data geometry
or by the EFGP approximation.

The key comparison is not just "small random subsample" vs "large subsample".
For a dense spatial field, random subsampling changes the infill geometry. This
script supports three sampling modes:

1. random: uniform random valid pixels over the whole raster
2. stride: regular thinning over the whole raster, preserving domain coverage
3. window: contiguous crop at native density, preserving local spacing

For each sample, the script can:
- standardize x/y the same way as the notebook
- report nearest-neighbor spacing statistics
- fit an exact dense SE GP by Cholesky on manageable subsets
- optionally fit EFGPND on the same subset for exact-vs-approx comparison
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

# The NUFFT / PyTorch stack in this repo is sensitive to duplicate OpenMP
# runtimes and multi-threaded startup in subprocess-style runs.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
from PIL import Image
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.spatial import cKDTree, distance

torch.set_num_threads(1)

THIS_DIR = Path(__file__).resolve().parent
REPO_DIR = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from efgpnd import EFGPND  # noqa: E402
from kernels.squared_exponential import SquaredExponential  # noqa: E402
from load_prism import (  # noqa: E402
    _find_tif_path,
    _get_geotransform,
    _get_nodata_value,
    _resolve_dataset_dir,
    load_prism_dataset,
)


@dataclass
class RasterGrid:
    data: np.ndarray
    valid: np.ndarray
    nrows: int
    ncols: int
    origin_lon: float
    origin_lat: float
    pixel_width: float
    pixel_height: float
    valid_count: int

    def coords_from_rows_cols(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        lon = self.origin_lon + (cols.astype(np.float64) + 0.5) * self.pixel_width
        lat = self.origin_lat - (rows.astype(np.float64) + 0.5) * self.pixel_height
        return np.column_stack((lon, lat))


def load_raster_grid(dataset: str) -> RasterGrid:
    dataset_dir = _resolve_dataset_dir(dataset)
    tif_path = _find_tif_path(dataset_dir)
    img = Image.open(tif_path)
    data = np.array(img, dtype=np.float64)
    nrows, ncols = data.shape
    origin_lon, origin_lat, pixel_width, pixel_height = _get_geotransform(img)
    nodata = _get_nodata_value(img)
    valid = np.isfinite(data) if nodata is None else data != nodata
    return RasterGrid(
        data=data,
        valid=valid,
        nrows=nrows,
        ncols=ncols,
        origin_lon=origin_lon,
        origin_lat=origin_lat,
        pixel_width=pixel_width,
        pixel_height=pixel_height,
        valid_count=int(valid.sum()),
    )


def standardize_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x_min = x.min(axis=0)
    x_max = x.max(axis=0)
    span = np.where((x_max - x_min) > 0, x_max - x_min, 1.0)
    x_std = (x - x_min) / span
    y_std = (y - y.mean()) / y.std()
    return x_std, y_std


def nearest_neighbor_stats(x: np.ndarray, y: np.ndarray, max_points: int = 10_000) -> Dict[str, float]:
    n = x.shape[0]
    if n <= 1:
        return {
            "median_nn": float("nan"),
            "p10_nn": float("nan"),
            "p90_nn": float("nan"),
            "median_semivar_nn": float("nan"),
            "mean_semivar_nn": float("nan"),
        }
    probe_n = min(max_points, n)
    probe_idx = np.arange(probe_n)
    tree = cKDTree(x)
    dists, idxs = tree.query(x[probe_idx], k=2)
    nn = dists[:, 1]
    dy = y[probe_idx] - y[idxs[:, 1]]
    semivar = 0.5 * dy**2
    return {
        "median_nn": float(np.median(nn)),
        "p10_nn": float(np.quantile(nn, 0.10)),
        "p90_nn": float(np.quantile(nn, 0.90)),
        "median_semivar_nn": float(np.median(semivar)),
        "mean_semivar_nn": float(np.mean(semivar)),
    }


def sample_random(dataset: str, target_n: int, seed: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    x, y = load_prism_dataset(dataset, n_sub=target_n, seed=seed)
    meta = {"actual_n": float(len(y))}
    return x, y, meta


def sample_stride(raster: RasterGrid, target_n: int, seed: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    base_stride = max(1, int(round(math.sqrt(raster.valid_count / target_n))))
    candidates = sorted({max(1, base_stride + delta) for delta in (-1, 0, 1, 2)})

    best = None
    for stride in candidates:
        row_offset = int(rng.integers(0, stride))
        col_offset = int(rng.integers(0, stride))
        rows = np.arange(row_offset, raster.nrows, stride)
        cols = np.arange(col_offset, raster.ncols, stride)
        valid_sub = raster.valid[np.ix_(rows, cols)]
        actual_n = int(valid_sub.sum())
        score = abs(actual_n - target_n)
        if best is None or score < best["score"]:
            best = {
                "stride": stride,
                "row_offset": row_offset,
                "col_offset": col_offset,
                "rows": rows,
                "cols": cols,
                "valid_sub": valid_sub,
                "actual_n": actual_n,
                "score": score,
            }

    rows_2d, cols_2d = np.nonzero(best["valid_sub"])
    sel_rows = best["rows"][rows_2d]
    sel_cols = best["cols"][cols_2d]
    if sel_rows.size > target_n:
        keep = rng.choice(sel_rows.size, size=target_n, replace=False)
        sel_rows = sel_rows[keep]
        sel_cols = sel_cols[keep]
    x = raster.coords_from_rows_cols(sel_rows, sel_cols)
    y = raster.data[sel_rows, sel_cols]
    meta = {
        "actual_n": float(len(y)),
        "stride": float(best["stride"]),
        "row_offset": float(best["row_offset"]),
        "col_offset": float(best["col_offset"]),
    }
    return x, y, meta


def sample_window(raster: RasterGrid, target_n: int, seed: int, trials: int = 12) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    area_fraction = min(1.0, target_n / raster.valid_count)
    row_span = max(1, int(round(raster.nrows * math.sqrt(area_fraction))))
    col_span = max(1, int(round(raster.ncols * math.sqrt(area_fraction))))

    best = None
    for _ in range(trials):
        row0 = int(rng.integers(0, max(1, raster.nrows - row_span + 1)))
        col0 = int(rng.integers(0, max(1, raster.ncols - col_span + 1)))
        rows = np.arange(row0, row0 + row_span)
        cols = np.arange(col0, col0 + col_span)
        valid_sub = raster.valid[np.ix_(rows, cols)]
        actual_n = int(valid_sub.sum())
        score = abs(actual_n - target_n)
        if best is None or score < best["score"]:
            best = {
                "row0": row0,
                "col0": col0,
                "rows": rows,
                "cols": cols,
                "valid_sub": valid_sub,
                "actual_n": actual_n,
                "score": score,
            }

    rows_2d, cols_2d = np.nonzero(best["valid_sub"])
    sel_rows = best["rows"][rows_2d]
    sel_cols = best["cols"][cols_2d]
    if sel_rows.size > target_n:
        keep = rng.choice(sel_rows.size, size=target_n, replace=False)
        sel_rows = sel_rows[keep]
        sel_cols = sel_cols[keep]
    x = raster.coords_from_rows_cols(sel_rows, sel_cols)
    y = raster.data[sel_rows, sel_cols]
    meta = {
        "actual_n": float(len(y)),
        "window_rows": float(len(best["rows"])),
        "window_cols": float(len(best["cols"])),
        "row0": float(best["row0"]),
        "col0": float(best["col0"]),
    }
    return x, y, meta


def sample_dataset(dataset: str, raster: RasterGrid, mode: str, target_n: int, seed: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    if mode == "random":
        return sample_random(dataset, target_n=target_n, seed=seed)
    if mode == "stride":
        return sample_stride(raster, target_n=target_n, seed=seed)
    if mode == "window":
        return sample_window(raster, target_n=target_n, seed=seed)
    raise ValueError(f"Unsupported sampling mode: {mode}")


def exact_nll_and_grad(theta: np.ndarray, d2: np.ndarray, y: np.ndarray, jitter: float) -> Tuple[float, np.ndarray]:
    ell, sigf2, sign2 = np.exp(theta)
    k_signal = sigf2 * np.exp(-0.5 * d2 / (ell * ell))
    eye = np.eye(d2.shape[0], dtype=np.float64)

    jitter_now = jitter
    for _ in range(6):
        try:
            k = k_signal + (sign2 + jitter_now) * eye
            chol, lower = cho_factor(k, lower=True, check_finite=False)
            alpha = cho_solve((chol, lower), y, check_finite=False)
            kinv = cho_solve((chol, lower), eye, check_finite=False)
            break
        except np.linalg.LinAlgError:
            jitter_now *= 10.0
    else:
        return float("inf"), np.zeros_like(theta)

    n = y.shape[0]
    nll = 0.5 * float(y @ alpha) + float(np.log(np.diag(chol)).sum()) + 0.5 * n * math.log(2.0 * math.pi)

    common = kinv - np.outer(alpha, alpha)
    dlog_ell = k_signal * (d2 / (ell * ell))
    dlog_sigf2 = k_signal
    dlog_sign2 = sign2 * eye
    grad = np.empty_like(theta)
    grad[0] = 0.5 * np.sum(common * dlog_ell)
    grad[1] = 0.5 * np.sum(common * dlog_sigf2)
    grad[2] = 0.5 * np.sum(common * dlog_sign2)
    return nll, grad


def exact_fit_se(x: np.ndarray, y: np.ndarray, jitter: float, maxiter: int) -> Dict[str, float]:
    d2 = distance.cdist(x, x, metric="sqeuclidean")
    tree = cKDTree(x)
    nn, _ = tree.query(x[: min(len(x), 2048)], k=2)
    median_nn = max(float(np.median(nn[:, 1])), 1e-4)
    nonzero_d2 = d2[d2 > 0]
    median_pairwise = math.sqrt(float(np.median(nonzero_d2))) if nonzero_d2.size else 0.1

    starts = [
        np.log([min(0.5, max(5.0 * median_nn, 1e-3)), 1.0, 0.10]),
        np.log([min(0.5, max(20.0 * median_nn, 5e-3)), 1.0, 0.05]),
        np.log([min(0.5, max(0.25 * median_pairwise, 1e-2)), 1.0, 0.01]),
    ]
    bounds = [
        (math.log(1e-4), math.log(1.0)),
        (math.log(1e-4), math.log(100.0)),
        (math.log(1e-6), math.log(10.0)),
    ]

    best = None
    for theta0 in starts:
        result = minimize(
            fun=lambda th: exact_nll_and_grad(th, d2, y, jitter),
            x0=theta0,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter},
        )
        if best is None or result.fun < best.fun:
            best = result

    ell, sigf2, sign2 = np.exp(best.x)
    return {
        "exact_lengthscale": float(ell),
        "exact_variance": float(sigf2),
        "exact_sigmasq": float(sign2),
        "exact_nll": float(best.fun),
        "exact_success": float(bool(best.success)),
        "exact_iters": float(best.nit),
    }


def approx_fit_efgp(
    x: np.ndarray,
    y: np.ndarray,
    *,
    epsilon: float,
    trace_samples: int,
    cg_tol: float,
    lr: float,
    max_iters: int,
) -> Dict[str, float]:
    x_t = torch.from_numpy(x.astype(np.float64))
    y_t = torch.from_numpy(y.astype(np.float64))
    nn = cKDTree(x).query(x[: min(len(x), 2048)], k=2)[0][:, 1]
    init_ls = min(0.5, max(20.0 * float(np.median(nn)), 5e-3))
    kernel = SquaredExponential(dimension=x.shape[1], init_lengthscale=init_ls, init_variance=1.0)
    model = EFGPND(
        x_t,
        y_t,
        kernel=kernel,
        sigmasq=0.10,
        eps=epsilon,
        estimate_params=False,
    )
    model.optimize_hyperparameters(
        optimizer="Adam",
        lr=lr,
        max_iters=max_iters,
        trace_samples=trace_samples,
        compute_log_marginal=False,
        verbose=False,
        cg_tol=cg_tol,
    )
    return {
        "approx_lengthscale": float(model.kernel.get_hyper("lengthscale")),
        "approx_variance": float(model.kernel.get_hyper("variance")),
        "approx_sigmasq": float(model.sigmasq.item()),
    }


def iter_sizes(values: Sequence[int]) -> Iterable[int]:
    for value in values:
        if value <= 0:
            raise ValueError("All sample sizes must be positive.")
        yield int(value)


def write_csv(rows: List[Dict[str, float]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="prism_tmean_us_30s_2020_avg_30y")
    parser.add_argument("--modes", nargs="+", default=["random", "stride", "window"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000, 2500, 5000])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exact-jitter", type=float, default=1e-8)
    parser.add_argument("--exact-maxiter", type=int, default=50)
    parser.add_argument("--with-approx", action="store_true")
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--trace-samples", type=int, default=1)
    parser.add_argument("--cg-tol", type=float, default=1e-4)
    parser.add_argument("--approx-lr", type=float, default=0.1)
    parser.add_argument("--approx-maxiters", type=int, default=25)
    parser.add_argument(
        "--csv",
        type=Path,
        default=REPO_DIR / "experiments" / "prism_hyper_diagnostics.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raster = load_raster_grid(args.dataset)
    rows: List[Dict[str, float]] = []

    print(
        f"Loaded {args.dataset}: shape=({raster.nrows}, {raster.ncols}), "
        f"valid_count={raster.valid_count}"
    )

    for mode in args.modes:
        for i, target_n in enumerate(iter_sizes(args.sizes)):
            sample_seed = args.seed + 10_000 * i + 1_000_000 * args.modes.index(mode)
            x_raw, y_raw, meta = sample_dataset(
                args.dataset,
                raster=raster,
                mode=mode,
                target_n=target_n,
                seed=sample_seed,
            )
            x, y = standardize_xy(x_raw, y_raw)
            row: Dict[str, float] = {
                "mode": mode,
                "target_n": float(target_n),
                "seed": float(sample_seed),
            }
            row.update(meta)
            row.update(nearest_neighbor_stats(x, y))
            row.update(exact_fit_se(x, y, jitter=args.exact_jitter, maxiter=args.exact_maxiter))
            if args.with_approx:
                row.update(
                    approx_fit_efgp(
                        x,
                        y,
                        epsilon=args.epsilon,
                        trace_samples=args.trace_samples,
                        cg_tol=args.cg_tol,
                        lr=args.approx_lr,
                        max_iters=args.approx_maxiters,
                    )
                )
                row["lengthscale_ratio_approx_to_exact"] = row["approx_lengthscale"] / row["exact_lengthscale"]
                row["variance_ratio_approx_to_exact"] = row["approx_variance"] / row["exact_variance"]
                row["sigmasq_ratio_approx_to_exact"] = row["approx_sigmasq"] / row["exact_sigmasq"]

            rows.append(row)
            summary = (
                f"mode={mode:>6} target_n={target_n:>6} actual_n={int(row['actual_n']):>6} "
                f"median_nn={row['median_nn']:.3e} "
                f"exact(ls={row['exact_lengthscale']:.3e}, var={row['exact_variance']:.3e}, "
                f"noise={row['exact_sigmasq']:.3e})"
            )
            if args.with_approx:
                summary += (
                    f" approx(ls={row['approx_lengthscale']:.3e}, var={row['approx_variance']:.3e}, "
                    f"noise={row['approx_sigmasq']:.3e})"
                )
            print(summary)

    write_csv(rows, args.csv)
    print(f"CSV written to {args.csv}")


if __name__ == "__main__":
    main()
