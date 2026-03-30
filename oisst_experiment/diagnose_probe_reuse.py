#!/usr/bin/env python3
"""
Quick OISST study: fresh vs reused Hutchinson probes in EFGP training.

Cases:
1. fresh_j1   : 1 fresh probe per iteration
2. fresh_j10  : 10 fresh probes per iteration
3. fixed_j10  : the same 10 probes reused every iteration

The training loop mirrors diagnose_oisst_ski_vs_efgp.py for the EFGP arm.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.optim import Adam

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from efgpnd import EFGPND
from kernels.squared_exponential import SquaredExponential
from load_oisst import load_oisst_torch
from oisst_experiment.diagnose_oisst_ski_vs_efgp import _exact_log_marginal, _normalize_split, _rmse


def _total_variation(values):
    return float(sum(abs(values[i] - values[i - 1]) for i in range(1, len(values))))


def _fit_efgp_probe_mode(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    init_lengthscale: float,
    init_variance: float,
    init_noise: float,
    lr: float,
    iters: int,
    trace_samples: int,
    cg_tol: float,
    noise_floor: float,
    eps: float,
    base_seed: int,
    probe_mode: str,
    probe_seed: Optional[int],
    print_every: int,
) -> Dict:
    torch.manual_seed(base_seed)

    kernel = SquaredExponential(
        dimension=x_train.size(1),
        init_lengthscale=init_lengthscale,
        init_variance=init_variance,
    )
    model = EFGPND(
        x_train,
        y_train,
        kernel=kernel,
        sigmasq=init_noise,
        eps=eps,
        estimate_params=False,
    )
    optimizer = Adam(model.parameters(), lr=lr)

    history = []
    best_state = None
    best_log_marginal = float("-inf")
    t0 = time.perf_counter()

    for iteration in range(1, iters + 1):
        optimizer.zero_grad()
        if probe_mode == "fixed":
            if probe_seed is None:
                raise ValueError("probe_seed is required for fixed probe mode")
            torch.manual_seed(probe_seed)
        model.compute_gradients(
            trace_samples=trace_samples,
            cg_tol=cg_tol,
            noise_floor=noise_floor,
        )
        optimizer.step()

        ls = float(model.kernel.get_hyper("lengthscale"))
        var = float(model.kernel.get_hyper("variance"))
        noise = float(model.sigmasq.item())
        exact_log_marginal = _exact_log_marginal(
            x_train,
            y_train,
            lengthscale=ls,
            variance=var,
            noise=noise,
        )
        mean_val = model.predict(x_val, return_variance=False, force_recompute=True)
        mean_val = mean_val[0] if isinstance(mean_val, tuple) else mean_val
        val_rmse = _rmse(mean_val.detach(), y_val)
        stats = dict(model.last_gradient_stats)
        history.append(
            {
                "iteration": iteration,
                "lengthscale": ls,
                "variance": var,
                "noise": noise,
                "exact_log_marginal": exact_log_marginal,
                "val_rmse": val_rmse,
                "mean_cg_iters": stats.get("mean_cg_iters"),
                "trace_cg_iters": stats.get("trace_cg_iters"),
                "mtot": stats.get("mtot"),
                "M": stats.get("feature_count"),
            }
        )
        if exact_log_marginal > best_log_marginal:
            best_log_marginal = exact_log_marginal
            best_state = copy.deepcopy(model.state_dict())
        if iteration == 1 or iteration % print_every == 0 or iteration == iters:
            print(
                f"[{probe_mode:>5} J={trace_samples:>2}] iter {iteration:>3}/{iters}  "
                f"ls={ls:.6g} var={var:.6g} noise={noise:.6g}  "
                f"exact_log_marg={exact_log_marginal:.6f}  "
                f"val_rmse={val_rmse:.6f}  "
                f"cg(mean/trace)={stats.get('mean_cg_iters')}/{stats.get('trace_cg_iters')}",
                flush=True,
            )

    elapsed = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    ls_hist = [row["lengthscale"] for row in history]
    var_hist = [row["variance"] for row in history]
    exact_hist = [row["exact_log_marginal"] for row in history]
    rmse_hist = [row["val_rmse"] for row in history]

    summary = {
        "probe_mode": probe_mode,
        "trace_samples": trace_samples,
        "probe_seed": probe_seed,
        "elapsed_sec": elapsed,
        "final": history[-1],
        "best_exact_log_marginal": max(exact_hist),
        "best_val_rmse": min(rmse_hist),
        "lengthscale_total_variation": _total_variation(ls_hist),
        "variance_total_variation": _total_variation(var_hist),
        "mean_trace_cg_iters": float(sum(row["trace_cg_iters"] for row in history) / len(history)),
        "mean_mean_cg_iters": float(sum(row["mean_cg_iters"] for row in history) / len(history)),
        "history": history,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose fresh vs reused probes on OISST EFGP training.")
    parser.add_argument("--train-n", type=int, default=1_500)
    parser.add_argument("--val-n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--cg-tol", type=float, default=1e-5)
    parser.add_argument("--noise-floor", type=float, default=1e-6)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--probe-seed", type=int, default=1234)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(__file__).resolve().parent / "diagnostics" / "oisst_probe_reuse_summary.json",
    )
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)

    total_n = args.train_n + args.val_n
    x_all, y_all = load_oisst_torch(
        n_sub=total_n,
        seed=args.seed,
        path=str(REPO_ROOT.parent / "oisst-avhrr-v02r01.20260315_preliminary.nc"),
    )
    x_train = x_all[: args.train_n]
    y_train = y_all[: args.train_n]
    x_val = x_all[args.train_n :]
    y_val = y_all[args.train_n :]
    x_train, y_train, x_val, y_val = _normalize_split(
        x_train,
        y_train,
        x_val,
        y_val,
        dtype=dtype,
    )

    init_kernel = SquaredExponential(dimension=x_train.size(1))
    init_lengthscale, init_variance, init_noise = init_kernel.estimate_hyperparameters(x_train, y_train)

    print(
        f"Initial hypers: ls={init_lengthscale:.6g} var={init_variance:.6g} noise={init_noise:.6g}",
        flush=True,
    )

    cases = [
        ("fresh", 1, None),
        ("fresh", 10, None),
        ("fixed", 10, args.probe_seed),
    ]
    results = {}
    for probe_mode, trace_samples, probe_seed in cases:
        key = f"{probe_mode}_j{trace_samples}"
        print(f"\n=== {key} ===", flush=True)
        results[key] = _fit_efgp_probe_mode(
            x_train,
            y_train,
            x_val,
            y_val,
            init_lengthscale=init_lengthscale,
            init_variance=init_variance,
            init_noise=init_noise,
            lr=args.lr,
            iters=args.iters,
            trace_samples=trace_samples,
            cg_tol=args.cg_tol,
            noise_floor=args.noise_floor,
            eps=args.eps,
            base_seed=args.seed,
            probe_mode=probe_mode,
            probe_seed=probe_seed,
            print_every=args.print_every,
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2))
    print(f"\nWrote summary to {args.output_json}", flush=True)

    print("\n=== Summary ===", flush=True)
    for key, result in results.items():
        final = result["final"]
        print(
            f"{key:>10}  final ls={final['lengthscale']:.6g} var={final['variance']:.6g} noise={final['noise']:.6g}  "
            f"best_exact_log_marg={result['best_exact_log_marginal']:.6f}  "
            f"best_val_rmse={result['best_val_rmse']:.6f}  "
            f"TV(ls)={result['lengthscale_total_variation']:.3e}  "
            f"TV(var)={result['variance_total_variation']:.3e}  "
            f"time={result['elapsed_sec']:.2f}s"
        )


if __name__ == "__main__":
    main()
