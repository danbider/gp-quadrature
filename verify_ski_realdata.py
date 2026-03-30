"""
Run a lightweight SKI hyper-learning check on the real-data loaders.

This is meant to catch obvious memory/runtime regressions before using the
same helper from the notebooks.
"""

from __future__ import annotations

import argparse

import torch

from oisst_experiment.load_oisst import load_oisst_torch
from prism_experiment.load_prism import load_prism_dataset_torch
from utils.ski import fit_ski_gp


def _normalize_xy(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.to(torch.float32)
    y = y.to(torch.float32)

    x_min = x.min(dim=0).values
    x_max = x.max(dim=0).values
    x = (x - x_min) / (x_max - x_min)

    y = (y - y.mean()) / y.std()
    return x, y


def _run_case(name: str, x: torch.Tensor, y: torch.Tensor, *, grid_size, max_iters: int) -> None:
    x, y = _normalize_xy(x, y)
    print(f"\n=== {name} ===", flush=True)
    print(f"points={x.shape[0]:,} dims={x.shape[1]}", flush=True)

    result = fit_ski_gp(
        x,
        y,
        kernel="SE",
        grid_size=grid_size,
        max_iters=max_iters,
        lr=0.05,
        noise_floor=1e-4,
        cg_tolerance=1e-3,
        max_cg_iterations=100,
        max_preconditioner_size=10,
        max_lanczos_quadrature_iterations=10,
        num_trace_samples=2,
        checkpoint_size=None,
        use_toeplitz=True,
        memory_efficient=True,
        verbose=True,
    )

    print(
        "summary:",
        {
            "best_iteration": result["best_iteration"],
            "best_loss": round(result["best_loss"], 6),
            "num_train": result["num_train"],
            "grid_size": result["grid_size"],
            "fit_time_sec": round(result["fit_time_sec"], 3),
        },
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify real-data SKI hyper-learning.")
    parser.add_argument("--oisst-n", type=int, default=250_000, help="Number of OISST points to load.")
    parser.add_argument("--prism-n", type=int, default=500_000, help="Number of PRISM points to load.")
    parser.add_argument("--max-iters", type=int, default=1, help="Number of SKI optimizer steps per dataset.")
    args = parser.parse_args()

    oisst_x, oisst_y = load_oisst_torch(
        n_sub=args.oisst_n,
        seed=0,
        path="oisst-avhrr-v02r01.20260315_preliminary.nc",
    )
    _run_case("OISST", oisst_x, oisst_y, grid_size=(256, 128), max_iters=args.max_iters)

    prism_x, prism_y = load_prism_dataset_torch(
        "prism_tmean_us_30s_2020_avg_30y",
        n_sub=args.prism_n,
        seed=0,
    )
    _run_case("PRISM 30y", prism_x, prism_y, grid_size=(288, 128), max_iters=args.max_iters)


if __name__ == "__main__":
    main()
