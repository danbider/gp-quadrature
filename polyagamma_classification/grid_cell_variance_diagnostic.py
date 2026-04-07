from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from pg_classifier import PolyagammaGPNegativeBinomialRegressor


REPO_DIR = Path(__file__).resolve().parent
DEFAULT_NWB_PATH = REPO_DIR / "data" / "dandi_000582" / "sub-11265_ses-07020602_behavior+ecephys.nwb"


@dataclass
class DatasetSlice:
    X: np.ndarray
    y: np.ndarray
    start_bin: int
    stop_bin: int
    bin_size: float
    count_mean: float
    count_variance: float
    zero_fraction: float
    max_count: int


@dataclass
class VariantSummary:
    name: str
    fit_seconds: float
    final_lengthscale: float
    final_variance: float
    final_total_count: float
    final_mae: float
    mean_grad_variance: float
    positive_grad_fraction: float
    first_variance_ratio: float
    mean_variance_ratio: float
    exp_lr: float
    variance_path: list[float]
    lengthscale_path: list[float]
    total_count_path: list[float]
    grad_variance_path: list[float]


def extract_unit_spike_times(
    spike_times: np.ndarray,
    spike_times_index: np.ndarray,
    unit_index: int,
) -> np.ndarray:
    start = 0 if unit_index == 0 else int(spike_times_index[unit_index - 1])
    stop = int(spike_times_index[unit_index])
    return spike_times[start:stop]


def load_grid_cell_slice(
    *,
    nwb_path: Path,
    neuron: int,
    bin_size: float,
    window_bins: int,
) -> DatasetSlice:
    with h5py.File(nwb_path, "r") as f:
        position = f["processing/behavior/Position/SpatialSeriesLED1/data"][:]
        position_t = f["processing/behavior/Position/SpatialSeriesLED1/timestamps"][:]
        spike_times = f["units/spike_times"][:]
        spike_times_index = f["units/spike_times_index"][:]

    unit_spikes = extract_unit_spike_times(spike_times, spike_times_index, neuron)
    t0 = float(position_t[0])
    t1 = float(position_t[-1])
    n_bins = int(np.floor((t1 - t0) / bin_size))
    edges = t0 + np.arange(n_bins + 1) * bin_size
    centers = edges[:-1] + 0.5 * bin_size

    counts_all = np.histogram(unit_spikes, bins=edges)[0].astype(np.float64)
    position_interp = np.column_stack(
        [
            np.interp(centers, position_t, position[:, 0]),
            np.interp(centers, position_t, position[:, 1]),
        ]
    )
    coord_mins = position_interp.min(axis=0)
    coord_maxs = position_interp.max(axis=0)
    coord_span = np.where(coord_maxs > coord_mins, coord_maxs - coord_mins, 1.0)
    X_all = 2.0 * (position_interp - coord_mins) / coord_span - 1.0

    if window_bins <= 0 or window_bins >= counts_all.size:
        start_bin = 0
        stop_bin = counts_all.size
    else:
        rolling = np.convolve(counts_all, np.ones(window_bins, dtype=np.float64), mode="valid")
        start_bin = int(np.argmax(rolling))
        stop_bin = start_bin + window_bins

    y = counts_all[start_bin:stop_bin]
    X = X_all[start_bin:stop_bin]
    return DatasetSlice(
        X=X,
        y=y,
        start_bin=start_bin,
        stop_bin=stop_bin,
        bin_size=bin_size,
        count_mean=float(y.mean()),
        count_variance=float(y.var()),
        zero_fraction=float(np.mean(y == 0)),
        max_count=int(y.max()),
    )


def notebook_like_config(max_iter: int, random_state: int) -> dict[str, object]:
    return {
        "total_count": 2.0,
        "learn_total_count": True,
        "total_count_lr": 0.05,
        "total_count_update_frequency": 1,
        "total_count_quadrature_nodes": 16,
        "lengthscale_init": 0.12,
        "variance_init": 1.0,
        "max_iter": max_iter,
        "e_step_iters": 1,
        "final_e_step_iters": 2,
        "rho0": 1.0,
        "gamma": 1e-3,
        "lr": 0.04,
        "n_e_probes": 1,
        "n_m_probes": 1,
        "cg_tol": 1e-5,
        "nufft_eps": 1e-4,
        "spectral_eps": 1e-4,
        "trunc_eps": 1e-4,
        "prediction_batch_size": 128,
        "predictive_variance_method": "chebyshev",
        "predictive_variance_chebyshev_nodes": 7,
        "use_exact_weighted_toeplitz_operator": True,
        "random_state": random_state,
        "device": "cpu",
        "store_history": True,
        "verbose": 0,
    }


def diagnostic_variants(max_iter: int, random_state: int) -> dict[str, dict[str, object]]:
    baseline = notebook_like_config(max_iter=max_iter, random_state=random_state)
    return {
        "baseline": baseline,
        "lower_lr": {**baseline, "lr": 0.01},
        "better_probes_same_lr": {
            **baseline,
            "e_step_iters": 3,
            "final_e_step_iters": 3,
            "rho0": 0.7,
            "n_e_probes": 16,
            "n_m_probes": 64,
        },
        "better_probes_lower_lr": {
            **baseline,
            "lr": 0.01,
            "e_step_iters": 3,
            "final_e_step_iters": 3,
            "rho0": 0.7,
            "n_e_probes": 16,
            "n_m_probes": 64,
        },
        "slower_r_updates": {**baseline, "total_count_update_frequency": 5},
        "fixed_r": {**baseline, "learn_total_count": False},
    }


def summarize_variant(name: str, reg: PolyagammaGPNegativeBinomialRegressor, fit_seconds: float) -> VariantSummary:
    history = reg.history_[:-1]
    variance_path = [float(row["variance"]) for row in history]
    lengthscale_path = [float(row["lengthscale"]) for row in history]
    total_count_path = [float(row.get("total_count", reg.total_count_)) for row in history]
    grad_variance_path = [float(row["grad_variance"]) for row in history]

    prev_variance = float(reg.variance_init)
    variance_ratios: list[float] = []
    for variance in variance_path:
        variance_ratios.append(float(variance / prev_variance))
        prev_variance = variance

    positive_grad_fraction = float(np.mean(np.asarray(grad_variance_path) > 0.0)) if grad_variance_path else float("nan")
    return VariantSummary(
        name=name,
        fit_seconds=float(fit_seconds),
        final_lengthscale=float(reg.lengthscale_),
        final_variance=float(reg.variance_),
        final_total_count=float(reg.total_count_),
        final_mae=float(reg.training_metric_),
        mean_grad_variance=float(np.mean(grad_variance_path)) if grad_variance_path else float("nan"),
        positive_grad_fraction=positive_grad_fraction,
        first_variance_ratio=float(variance_ratios[0]) if variance_ratios else float("nan"),
        mean_variance_ratio=float(np.mean(variance_ratios)) if variance_ratios else float("nan"),
        exp_lr=float(np.exp(reg.lr)),
        variance_path=variance_path,
        lengthscale_path=lengthscale_path,
        total_count_path=total_count_path,
        grad_variance_path=grad_variance_path,
    )


def run_variant(
    name: str,
    config: dict[str, object],
    data: DatasetSlice,
) -> VariantSummary:
    reg = PolyagammaGPNegativeBinomialRegressor(**config)
    started = time.time()
    reg.fit(data.X, data.y)
    return summarize_variant(name=name, reg=reg, fit_seconds=time.time() - started)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose grid-cell variance growth under notebook and alternative settings.")
    parser.add_argument("--nwb-path", type=Path, default=DEFAULT_NWB_PATH)
    parser.add_argument("--neuron", type=int, default=7)
    parser.add_argument("--bin-size", type=float, default=0.01)
    parser.add_argument("--window-bins", type=int, default=1200, help="Use the highest-count contiguous window of this size. Use <=0 for all bins.")
    parser.add_argument("--max-iter", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None, help="Optional path for JSON output.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = load_grid_cell_slice(
        nwb_path=args.nwb_path,
        neuron=args.neuron,
        bin_size=args.bin_size,
        window_bins=args.window_bins,
    )
    variants = diagnostic_variants(max_iter=args.max_iter, random_state=args.random_state)

    print("Dataset slice")
    print(
        json.dumps(
            {
                "nwb_path": str(args.nwb_path),
                "neuron": args.neuron,
                "bin_size": args.bin_size,
                "start_bin": data.start_bin,
                "stop_bin": data.stop_bin,
                "n_bins": int(data.y.size),
                "count_mean": data.count_mean,
                "count_variance": data.count_variance,
                "zero_fraction": data.zero_fraction,
                "max_count": data.max_count,
            },
            indent=2,
        )
    )

    summaries: list[VariantSummary] = []
    for name, config in variants.items():
        summary = run_variant(name=name, config=config, data=data)
        summaries.append(summary)
        print()
        print(name)
        print(
            json.dumps(
                {
                    "fit_seconds": round(summary.fit_seconds, 3),
                    "final_lengthscale": round(summary.final_lengthscale, 6),
                    "final_variance": round(summary.final_variance, 6),
                    "final_total_count": round(summary.final_total_count, 6),
                    "final_mae": round(summary.final_mae, 6),
                    "mean_grad_variance": round(summary.mean_grad_variance, 6),
                    "positive_grad_fraction": round(summary.positive_grad_fraction, 6),
                    "first_variance_ratio": round(summary.first_variance_ratio, 6),
                    "mean_variance_ratio": round(summary.mean_variance_ratio, 6),
                    "exp_lr": round(summary.exp_lr, 6),
                    "variance_path": [round(v, 6) for v in summary.variance_path],
                    "total_count_path": [round(v, 6) for v in summary.total_count_path],
                    "grad_variance_path": [round(v, 6) for v in summary.grad_variance_path],
                },
                indent=2,
            )
        )

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "dataset": {
                        "start_bin": data.start_bin,
                        "stop_bin": data.stop_bin,
                        "bin_size": data.bin_size,
                        "count_mean": data.count_mean,
                        "count_variance": data.count_variance,
                        "zero_fraction": data.zero_fraction,
                        "max_count": data.max_count,
                        "n_bins": int(data.y.size),
                    },
                    "variants": [asdict(summary) for summary in summaries],
                },
                indent=2,
            )
        )
        print()
        print(f"Wrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
