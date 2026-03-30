"""
Diagnose SKI vs EFGP hyper-learning on a small OISST subset with a dense exact-GP
reference.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import torch
from torch.optim import Adam

import gpytorch

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from efgpnd import EFGPND
from kernels.squared_exponential import SquaredExponential
from load_oisst import load_oisst_torch


def _exact_log_marginal(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    lengthscale: float,
    variance: float,
    noise: float,
) -> float:
    kernel = SquaredExponential(
        dimension=x.size(1),
        init_lengthscale=lengthscale,
        init_variance=variance,
    )
    kernel.set_hyper("lengthscale", lengthscale)
    kernel.set_hyper("variance", variance)
    return kernel.log_marginal(x, y, sigmasq=noise)


def _normalize_split(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    dtype: torch.dtype,
):
    x_train = x_train.to(dtype=dtype)
    y_train = y_train.to(dtype=dtype)
    x_val = x_val.to(dtype=dtype)
    y_val = y_val.to(dtype=dtype)

    x_min = x_train.min(dim=0).values
    x_max = x_train.max(dim=0).values
    x_train = (x_train - x_min) / (x_max - x_min)
    x_val = (x_val - x_min) / (x_max - x_min)

    y_mean = y_train.mean()
    y_std = y_train.std()
    y_train = (y_train - y_mean) / y_std
    y_val = (y_val - y_mean) / y_std
    return x_train, y_train, x_val, y_val


def _rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def _positive_sqrt(value: float) -> float:
    return max(value, 0.0) ** 0.5


def _plot_hyper_trajectories(
    *,
    efgp_history,
    ski_history,
    dense_history,
    out_dir: Path,
    prefix: str,
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)

    method_specs = [
        ("EFGP", efgp_history, "tab:blue"),
        ("SKI", ski_history, "tab:orange"),
        ("Dense", dense_history, "tab:green"),
    ]

    def _plot_one(y_key: str, y_label: str, filename_suffix: str) -> str:
        fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
        for label, history, color in method_specs:
            x_vals = [entry["lengthscale"] for entry in history]
            if y_key == "sigma_f":
                y_vals = [_positive_sqrt(entry["variance"]) for entry in history]
            else:
                y_vals = [_positive_sqrt(entry["noise"]) for entry in history]

            ax.plot(
                x_vals,
                y_vals,
                "-o",
                linewidth=2.0,
                markersize=3.5,
                color=color,
                label=label,
                alpha=0.95,
            )
            ax.scatter(
                [x_vals[0]],
                [y_vals[0]],
                color=color,
                marker="s",
                s=45,
                zorder=3,
            )
            ax.scatter(
                [x_vals[-1]],
                [y_vals[-1]],
                color=color,
                marker="*",
                s=100,
                zorder=3,
            )

        ax.set_xlabel("ell")
        ax.set_ylabel(y_label)
        ax.set_title(f"OISST hyperparameter trajectory: ell vs {y_label}")
        ax.grid(alpha=0.3)
        ax.legend()

        out_path = out_dir / f"{prefix}_{filename_suffix}.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        return str(out_path)

    sigma_f_path = _plot_one("sigma_f", "sigma_f", "ell_vs_sigma_f")
    sigma_noise_path = _plot_one("sigma_noise", "sigma_noise", "ell_vs_sigma_noise")
    return {
        "ell_vs_sigma_f": sigma_f_path,
        "ell_vs_sigma_noise": sigma_noise_path,
    }


class _SKIZeroMeanGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, *, grid_size):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.base_kernel = gpytorch.kernels.RBFKernel()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.GridInterpolationKernel(
                self.base_kernel,
                grid_size=grid_size,
                num_dims=train_x.size(-1),
                grid_bounds=[(0.0, 1.0)] * train_x.size(-1),
            )
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x),
            self.covar_module(x),
        )


class _DenseZeroMeanGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.base_kernel = gpytorch.kernels.RBFKernel()
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x),
            self.covar_module(x),
        )


def _fit_efgp(
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
    print_every: int,
) -> Dict:
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
        eps=1e-5,
        estimate_params=False,
    )
    optimizer = Adam(model.parameters(), lr=lr)

    history = []
    best_state = None
    best_log_marginal = float("-inf")
    for iteration in range(1, iters + 1):
        optimizer.zero_grad()
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
        history.append(
            {
                "iteration": iteration,
                "lengthscale": ls,
                "variance": var,
                "noise": noise,
                "exact_log_marginal": exact_log_marginal,
                "val_rmse": val_rmse,
            }
        )
        if exact_log_marginal > best_log_marginal:
            best_log_marginal = exact_log_marginal
            best_state = copy.deepcopy(model.state_dict())
        if iteration == 1 or iteration % print_every == 0 or iteration == iters:
            print(
                f"[EFGP] iter {iteration:>3}/{iters}  "
                f"ls={ls:.6g} var={var:.6g} noise={noise:.6g}  "
                f"exact_log_marg={exact_log_marginal:.6f}  "
                f"val_rmse={val_rmse:.6f}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "model": model,
        "history": history,
        "final": history[-1],
        "best_exact_log_marginal": max(item["exact_log_marginal"] for item in history),
    }


def _fit_ski(
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
    grid_size,
    noise_floor: float,
    cg_tol: float,
    trace_samples: int,
    max_cg_iterations: int,
    max_cholesky_size: int,
    use_toeplitz: bool,
    memory_efficient: bool,
    fast_computations: bool,
    print_every: int,
) -> Dict:
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(noise_floor)
    ).to(dtype=x_train.dtype)
    model = _SKIZeroMeanGP(
        x_train,
        y_train,
        likelihood,
        grid_size=grid_size,
    ).to(dtype=x_train.dtype)

    with torch.no_grad():
        model.base_kernel.lengthscale = init_lengthscale
        model.covar_module.outputscale = init_variance
        likelihood.noise = max(init_noise, noise_floor)

    optimizer = Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    history = []
    best_model_state = None
    best_likelihood_state = None
    best_log_marginal = float("-inf")

    contexts = [
        gpytorch.settings.max_cholesky_size(max_cholesky_size),
        gpytorch.settings.cg_tolerance(cg_tol),
        gpytorch.settings.eval_cg_tolerance(cg_tol),
        gpytorch.settings.max_cg_iterations(max_cg_iterations),
        gpytorch.settings.max_preconditioner_size(10),
        gpytorch.settings.max_lanczos_quadrature_iterations(15),
        gpytorch.settings.num_trace_samples(trace_samples),
        gpytorch.settings.memory_efficient(memory_efficient),
        gpytorch.settings.use_toeplitz(use_toeplitz),
        gpytorch.settings.fast_computations(
            covar_root_decomposition=fast_computations,
            log_prob=fast_computations,
            solves=fast_computations,
        ),
    ]
    with ExitStack() as stack:
        for context in contexts:
            stack.enter_context(context)

        for iteration in range(1, iters + 1):
            model.train()
            likelihood.train()
            optimizer.zero_grad(set_to_none=True)
            loss = -mll(model(x_train), y_train)
            loss.backward()
            optimizer.step()

            ls = float(model.base_kernel.lengthscale.detach().item())
            var = float(model.covar_module.outputscale.detach().item())
            noise = float(likelihood.noise.detach().item())
            exact_log_marginal = _exact_log_marginal(
                x_train,
                y_train,
                lengthscale=ls,
                variance=var,
                noise=noise,
            )

            model.eval()
            likelihood.eval()
            with torch.no_grad():
                val_pred = model(x_val).mean
            val_rmse = _rmse(val_pred.detach(), y_val)

            history.append(
                {
                    "iteration": iteration,
                    "lengthscale": ls,
                    "variance": var,
                    "noise": noise,
                    "loss": float(loss.detach().item()),
                    "exact_log_marginal": exact_log_marginal,
                    "val_rmse": val_rmse,
                }
            )
            if exact_log_marginal > best_log_marginal:
                best_log_marginal = exact_log_marginal
                best_model_state = copy.deepcopy(model.state_dict())
                best_likelihood_state = copy.deepcopy(likelihood.state_dict())
            if iteration == 1 or iteration % print_every == 0 or iteration == iters:
                print(
                    f"[SKI ] iter {iteration:>3}/{iters}  "
                    f"ls={ls:.6g} var={var:.6g} noise={noise:.6g}  "
                    f"loss={float(loss.detach().item()):.6f}  "
                    f"exact_log_marg={exact_log_marginal:.6f}  "
                    f"val_rmse={val_rmse:.6f}",
                    flush=True,
                )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    if best_likelihood_state is not None:
        likelihood.load_state_dict(best_likelihood_state)
    return {
        "model": model,
        "likelihood": likelihood,
        "history": history,
        "final": history[-1],
        "best_exact_log_marginal": max(item["exact_log_marginal"] for item in history),
    }


def _fit_dense_exact(
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
    noise_floor: float,
    print_every: int,
) -> Dict:
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(noise_floor)
    ).to(dtype=x_train.dtype)
    model = _DenseZeroMeanGP(x_train, y_train, likelihood).to(dtype=x_train.dtype)

    with torch.no_grad():
        model.base_kernel.lengthscale = init_lengthscale
        model.covar_module.outputscale = init_variance
        likelihood.noise = max(init_noise, noise_floor)

    optimizer = Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    history = []
    best_model_state = None
    best_likelihood_state = None
    best_log_marginal = float("-inf")

    for iteration in range(1, iters + 1):
        model.train()
        likelihood.train()
        optimizer.zero_grad(set_to_none=True)
        loss = -mll(model(x_train), y_train)
        loss.backward()
        optimizer.step()

        ls = float(model.base_kernel.lengthscale.detach().item())
        var = float(model.covar_module.outputscale.detach().item())
        noise = float(likelihood.noise.detach().item())
        exact_log_marginal = _exact_log_marginal(
            x_train,
            y_train,
            lengthscale=ls,
            variance=var,
            noise=noise,
        )

        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            val_pred = model(x_val).mean
        val_rmse = _rmse(val_pred.detach(), y_val)

        history.append(
            {
                "iteration": iteration,
                "lengthscale": ls,
                "variance": var,
                "noise": noise,
                "loss": float(loss.detach().item()),
                "exact_log_marginal": exact_log_marginal,
                "val_rmse": val_rmse,
            }
        )
        if exact_log_marginal > best_log_marginal:
            best_log_marginal = exact_log_marginal
            best_model_state = copy.deepcopy(model.state_dict())
            best_likelihood_state = copy.deepcopy(likelihood.state_dict())
        if iteration == 1 or iteration % print_every == 0 or iteration == iters:
            print(
                f"[DENSE] iter {iteration:>3}/{iters}  "
                f"ls={ls:.6g} var={var:.6g} noise={noise:.6g}  "
                f"loss={float(loss.detach().item()):.6f}  "
                f"exact_log_marg={exact_log_marginal:.6f}  "
                f"val_rmse={val_rmse:.6f}",
                flush=True,
            )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    if best_likelihood_state is not None:
        likelihood.load_state_dict(best_likelihood_state)
    return {
        "model": model,
        "likelihood": likelihood,
        "history": history,
        "final": history[-1],
        "best_exact_log_marginal": max(item["exact_log_marginal"] for item in history),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose OISST SKI vs EFGP hyper-learning.")
    parser.add_argument("--train-n", type=int, default=1_500)
    parser.add_argument("--val-n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--efgp-iters", type=int, default=None)
    parser.add_argument("--ski-iters", type=int, default=None)
    parser.add_argument("--dense-iters", type=int, default=None)
    parser.add_argument("--efgp-lr", type=float, default=0.05)
    parser.add_argument("--ski-lr", type=float, default=0.1)
    parser.add_argument("--dense-lr", type=float, default=0.05)
    parser.add_argument("--efgp-trace-samples", type=int, default=4)
    parser.add_argument("--ski-trace-samples", type=int, default=4)
    parser.add_argument("--ski-cg-tol", type=float, default=1e-3)
    parser.add_argument("--grid-lon", type=int, default=128)
    parser.add_argument("--grid-lat", type=int, default=64)
    parser.add_argument("--noise-floor", type=float, default=1e-6)
    parser.add_argument("--cg-tol", type=float, default=1e-5)
    parser.add_argument("--ski-max-cg-iters", type=int, default=100)
    parser.add_argument("--ski-max-cholesky-size", type=int, default=0)
    parser.add_argument("--ski-exact-training", action="store_true")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "diagnostics",
    )
    parser.add_argument("--output-prefix", type=str, default="oisst_ski_vs_efgp")
    args = parser.parse_args()

    efgp_iters = args.iters if args.efgp_iters is None else args.efgp_iters
    ski_iters = args.iters if args.ski_iters is None else args.ski_iters
    dense_iters = args.iters if args.dense_iters is None else args.dense_iters

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
    init_exact_log_marginal = _exact_log_marginal(
        x_train,
        y_train,
        lengthscale=init_lengthscale,
        variance=init_variance,
        noise=init_noise,
    )

    print(
        json.dumps(
            {
                "train_n": args.train_n,
                "val_n": args.val_n,
                "dtype": args.dtype,
                "init": {
                    "lengthscale": init_lengthscale,
                    "variance": init_variance,
                    "noise": init_noise,
                    "exact_log_marginal": init_exact_log_marginal,
                },
            },
            indent=2,
        ),
        flush=True,
    )

    start = time.time()
    efgp_result = _fit_efgp(
        x_train,
        y_train,
        x_val,
        y_val,
        init_lengthscale=init_lengthscale,
        init_variance=init_variance,
        init_noise=init_noise,
        lr=args.efgp_lr,
        iters=efgp_iters,
        trace_samples=args.efgp_trace_samples,
        cg_tol=args.cg_tol,
        noise_floor=args.noise_floor,
        print_every=args.print_every,
    )
    ski_result = _fit_ski(
        x_train,
        y_train,
        x_val,
        y_val,
        init_lengthscale=init_lengthscale,
        init_variance=init_variance,
        init_noise=init_noise,
        lr=args.ski_lr,
        iters=ski_iters,
        grid_size=(args.grid_lon, args.grid_lat),
        noise_floor=args.noise_floor,
        cg_tol=args.ski_cg_tol,
        trace_samples=args.ski_trace_samples,
        max_cg_iterations=args.ski_max_cg_iters,
        max_cholesky_size=args.ski_max_cholesky_size,
        use_toeplitz=not args.ski_exact_training,
        memory_efficient=not args.ski_exact_training,
        fast_computations=not args.ski_exact_training,
        print_every=args.print_every,
    )
    dense_result = _fit_dense_exact(
        x_train,
        y_train,
        x_val,
        y_val,
        init_lengthscale=init_lengthscale,
        init_variance=init_variance,
        init_noise=init_noise,
        lr=args.dense_lr,
        iters=dense_iters,
        noise_floor=args.noise_floor,
        print_every=args.print_every,
    )

    plot_paths = _plot_hyper_trajectories(
        efgp_history=efgp_result["history"],
        ski_history=ski_result["history"],
        dense_history=dense_result["history"],
        out_dir=args.output_dir,
        prefix=args.output_prefix,
    )

    summary = {
        "elapsed_sec": time.time() - start,
        "init_exact_log_marginal": init_exact_log_marginal,
        "efgp_final": efgp_result["final"],
        "efgp_best_exact_log_marginal": efgp_result["best_exact_log_marginal"],
        "ski_final": ski_result["final"],
        "ski_best_exact_log_marginal": ski_result["best_exact_log_marginal"],
        "dense_final": dense_result["final"],
        "dense_best_exact_log_marginal": dense_result["best_exact_log_marginal"],
        "plots": plot_paths,
    }
    print("\n=== Summary ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
