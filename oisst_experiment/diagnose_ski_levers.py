"""
Diagnose which SKI levers matter most on a small OISST subset.

The goal is to separate:
1. optimization / estimator budget (more steps, more Hutchinson probes, more CG)
2. fast iterative SKI training artifacts (CG/Lanczos/Hutchinson)
3. interpolation bias from the SKI grid itself
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.append(str(Path(__file__).resolve().parent))

from diagnose_oisst_ski_vs_efgp import (
    _exact_log_marginal,
    _fit_dense_exact,
    _fit_ski,
    _normalize_split,
)
from kernels.squared_exponential import SquaredExponential
from load_oisst import load_oisst_torch


def _load_split(train_n: int, val_n: int, seed: int, dtype: torch.dtype):
    x_all, y_all = load_oisst_torch(
        n_sub=train_n + val_n,
        seed=seed,
        path=str(REPO_ROOT.parent / "oisst-avhrr-v02r01.20260315_preliminary.nc"),
    )
    x_train = x_all[:train_n]
    y_train = y_all[:train_n]
    x_val = x_all[train_n:]
    y_val = y_all[train_n:]
    return _normalize_split(x_train, y_train, x_val, y_val, dtype=dtype)


def _best_entry(history: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    return max(history, key=lambda item: item[key])


def _summarize_run(name: str, result: Dict[str, Any]) -> Dict[str, Any]:
    best_exact = _best_entry(result["history"], "exact_log_marginal")
    best_rmse = min(result["history"], key=lambda item: item["val_rmse"])
    return {
        "name": name,
        "final": result["final"],
        "best_exact_log_marginal": result["best_exact_log_marginal"],
        "best_exact_entry": best_exact,
        "best_val_rmse": best_rmse["val_rmse"],
        "best_val_rmse_entry": best_rmse,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose SKI levers on a small OISST subset.")
    parser.add_argument("--train-n", type=int, default=400)
    parser.add_argument("--val-n", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)

    x_train, y_train, x_val, y_val = _load_split(
        train_n=args.train_n,
        val_n=args.val_n,
        seed=args.seed,
        dtype=dtype,
    )

    init_kernel = SquaredExponential(dimension=x_train.size(1))
    init_lengthscale, init_variance, init_noise = init_kernel.estimate_hyperparameters(x_train, y_train)

    init_summary = {
        "lengthscale": init_lengthscale,
        "variance": init_variance,
        "noise": init_noise,
        "exact_log_marginal": _exact_log_marginal(
            x_train,
            y_train,
            lengthscale=init_lengthscale,
            variance=init_variance,
            noise=init_noise,
        ),
    }
    print(json.dumps({"train_n": args.train_n, "val_n": args.val_n, "init": init_summary}, indent=2), flush=True)

    dense = _fit_dense_exact(
        x_train,
        y_train,
        x_val,
        y_val,
        init_lengthscale=init_lengthscale,
        init_variance=init_variance,
        init_noise=init_noise,
        lr=0.1,
        iters=60,
        noise_floor=1e-6,
        print_every=20,
    )
    dense_summary = _summarize_run("dense_exact_60_lr0.1", dense)

    variants = [
        {
            "name": "ski_fast_baseline_48x24_20_lr0.1",
            "iters": 20,
            "lr": 0.1,
            "grid": (48, 24),
            "trace_samples": 4,
            "cg_tol": 1e-3,
            "max_cg_iterations": 100,
            "max_cholesky_size": 0,
            "use_toeplitz": True,
            "memory_efficient": True,
            "fast_computations": True,
        },
        {
            "name": "ski_fast_betterbudget_48x24_60_lr0.1",
            "iters": 60,
            "lr": 0.1,
            "grid": (48, 24),
            "trace_samples": 16,
            "cg_tol": 1e-3,
            "max_cg_iterations": 200,
            "max_cholesky_size": 0,
            "use_toeplitz": True,
            "memory_efficient": True,
            "fast_computations": True,
        },
        {
            "name": "ski_exact_coarse_48x24_60_lr0.1",
            "iters": 60,
            "lr": 0.1,
            "grid": (48, 24),
            "trace_samples": 64,
            "cg_tol": 1e-6,
            "max_cg_iterations": 1000,
            "max_cholesky_size": 2000,
            "use_toeplitz": False,
            "memory_efficient": False,
            "fast_computations": False,
        },
        {
            "name": "ski_exact_fine_96x48_60_lr0.1",
            "iters": 60,
            "lr": 0.1,
            "grid": (96, 48),
            "trace_samples": 64,
            "cg_tol": 1e-6,
            "max_cg_iterations": 1000,
            "max_cholesky_size": 2000,
            "use_toeplitz": False,
            "memory_efficient": False,
            "fast_computations": False,
        },
        {
            "name": "ski_exact_finer_192x96_60_lr0.1",
            "iters": 60,
            "lr": 0.1,
            "grid": (192, 96),
            "trace_samples": 64,
            "cg_tol": 1e-6,
            "max_cg_iterations": 1000,
            "max_cholesky_size": 2000,
            "use_toeplitz": False,
            "memory_efficient": False,
            "fast_computations": False,
        },
    ]

    ski_summaries = []
    for variant in variants:
        print(f"\n=== {variant['name']} ===", flush=True)
        ski = _fit_ski(
            x_train,
            y_train,
            x_val,
            y_val,
            init_lengthscale=init_lengthscale,
            init_variance=init_variance,
            init_noise=init_noise,
            lr=variant["lr"],
            iters=variant["iters"],
            grid_size=variant["grid"],
            noise_floor=1e-6,
            cg_tol=variant["cg_tol"],
            trace_samples=variant["trace_samples"],
            max_cg_iterations=variant["max_cg_iterations"],
            max_cholesky_size=variant["max_cholesky_size"],
            use_toeplitz=variant["use_toeplitz"],
            memory_efficient=variant["memory_efficient"],
            fast_computations=variant["fast_computations"],
            print_every=20,
        )
        ski_summaries.append(_summarize_run(variant["name"], ski))

    summary = {
        "init": init_summary,
        "dense_anchor": dense_summary,
        "ski_variants": ski_summaries,
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2))

    print("\n=== SKI Lever Summary ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
