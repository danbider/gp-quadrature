"""
Utilities for fitting memory-conscious GPyTorch SKI baselines.
"""

from __future__ import annotations

import copy
import math
import time
from contextlib import ExitStack
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch

try:
    import gpytorch
except ImportError as exc:  # pragma: no cover - handled at runtime
    gpytorch = None
    _GPYTORCH_IMPORT_ERROR = exc
else:
    _GPYTORCH_IMPORT_ERROR = None

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None

GridSizeLike = Union[int, Sequence[int]]
GridBoundsLike = Sequence[Tuple[float, float]]


def _require_gpytorch() -> None:
    if gpytorch is None:
        raise ImportError(
            "utils.ski requires gpytorch in the active environment."
        ) from _GPYTORCH_IMPORT_ERROR


def _canonicalize_kernel_name(kernel: str) -> str:
    key = kernel.lower().replace("-", "").replace("_", "")
    if key in {"se", "squaredexponential", "rbf", "gaussian"}:
        return "rbf"
    if key in {"matern", "matern32", "mat32"}:
        return "matern32"
    if key in {"matern52", "mat52"}:
        return "matern52"
    raise ValueError(
        f"Unsupported SKI kernel '{kernel}'. "
        "Expected one of: SE, SquaredExponential, RBF, Matern32, Matern52."
    )


def _build_base_kernel(kernel: str):
    kernel_name = _canonicalize_kernel_name(kernel)
    if kernel_name == "rbf":
        return gpytorch.kernels.RBFKernel()
    if kernel_name == "matern32":
        return gpytorch.kernels.MaternKernel(nu=1.5)
    if kernel_name == "matern52":
        return gpytorch.kernels.MaternKernel(nu=2.5)
    raise AssertionError(f"Unhandled kernel name '{kernel_name}'")


def _resolve_grid_bounds(train_x: torch.Tensor, grid_bounds: Optional[GridBoundsLike]) -> Tuple[Tuple[float, float], ...]:
    num_dims = train_x.size(-1)
    if grid_bounds is not None:
        if len(grid_bounds) != num_dims:
            raise ValueError(
                f"grid_bounds has {len(grid_bounds)} dims, expected {num_dims}"
            )
        resolved = []
        for lo, hi in grid_bounds:
            lo_f = float(lo)
            hi_f = float(hi)
            if not hi_f > lo_f:
                raise ValueError(f"Each grid bound must satisfy hi > lo, got {(lo, hi)}")
            resolved.append((lo_f, hi_f))
        return tuple(resolved)

    mins = train_x.min(dim=0).values
    maxs = train_x.max(dim=0).values
    spans = torch.clamp(maxs - mins, min=1e-6)
    padding = 0.01 * spans
    return tuple(
        (float((mins[i] - padding[i]).item()), float((maxs[i] + padding[i]).item()))
        for i in range(num_dims)
    )


def _resolve_grid_size(
    *,
    grid_size: Optional[GridSizeLike],
    num_dims: int,
    target_grid_points: int,
    grid_bounds: GridBoundsLike,
) -> Union[int, Tuple[int, ...]]:
    if isinstance(grid_size, int):
        return int(grid_size)

    if grid_size is not None:
        resolved = tuple(int(v) for v in grid_size)
        if len(resolved) != num_dims:
            raise ValueError(
                f"grid_size has {len(resolved)} dims, expected {num_dims}"
            )
        if any(v <= 1 for v in resolved):
            raise ValueError("Each entry in grid_size must be > 1")
        return resolved

    base_side = max(16, int(round(target_grid_points ** (1.0 / num_dims))))
    spans = [max(hi - lo, 1e-6) for lo, hi in grid_bounds]
    geom_mean = math.prod(spans) ** (1.0 / num_dims)
    scaled = [max(16, int(round(base_side * (span / geom_mean)))) for span in spans]

    total = math.prod(scaled)
    if total > target_grid_points:
        shrink = (target_grid_points / total) ** (1.0 / num_dims)
        scaled = [max(16, int(math.floor(v * shrink))) for v in scaled]

    return tuple(scaled)


def _maybe_subsample_training_data(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    max_train_n: Optional[int],
    subsample_seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if max_train_n is None or x.size(0) <= max_train_n:
        return x, y, None

    generator = torch.Generator(device="cpu")
    generator.manual_seed(subsample_seed)
    train_indices = torch.randperm(x.size(0), generator=generator)[:max_train_n]
    train_indices, _ = torch.sort(train_indices)
    device_indices = train_indices.to(device=x.device)
    return (
        x.index_select(0, device_indices),
        y.index_select(0, device_indices),
        train_indices,
    )


def _rss_gb() -> Optional[float]:
    if psutil is None:
        return None
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 3)


class _SKIExactGPModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood,
        *,
        kernel: str,
        grid_size: Union[int, Tuple[int, ...]],
        grid_bounds: GridBoundsLike,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.base_kernel = _build_base_kernel(kernel)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.GridInterpolationKernel(
                self.base_kernel,
                grid_size=grid_size,
                num_dims=train_x.size(-1),
                grid_bounds=grid_bounds,
            )
        )

    def forward(self, x: torch.Tensor):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def fit_ski_gp(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    kernel: str = "SE",
    grid_size: Optional[GridSizeLike] = None,
    target_grid_points: int = 32_768,
    grid_bounds: Optional[GridBoundsLike] = None,
    max_iters: int = 50,
    lr: float = 0.05,
    noise_floor: float = 1e-4,
    dtype: torch.dtype = torch.float32,
    device: Optional[Union[str, torch.device]] = None,
    max_train_n: Optional[int] = None,
    subsample_seed: int = 0,
    init_lengthscale: Optional[float] = None,
    init_outputscale: Optional[float] = None,
    init_noise: Optional[float] = None,
    cg_tolerance: float = 1e-3,
    eval_cg_tolerance: Optional[float] = None,
    max_cg_iterations: int = 100,
    max_preconditioner_size: int = 10,
    max_lanczos_quadrature_iterations: int = 10,
    num_trace_samples: int = 2,
    checkpoint_size: Optional[int] = None,
    use_toeplitz: bool = True,
    memory_efficient: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Fit a GPyTorch exact GP with SKI interpolation and return training logs.

    Notes
    -----
    Hyper-learning is done with the standard GPyTorch route:
    ``loss = -ExactMarginalLogLikelihood(...); loss.backward(); optimizer.step()``.
    """
    _require_gpytorch()

    if x.ndim != 2:
        raise ValueError(f"x must have shape (N, d), got {tuple(x.shape)}")
    if y.ndim != 1:
        y = y.reshape(-1)
    if x.size(0) != y.size(0):
        raise ValueError(f"x and y must have matching first dims, got {x.size(0)} and {y.size(0)}")
    if max_iters < 1:
        raise ValueError("max_iters must be >= 1")

    total_n = x.size(0)
    train_x, train_y, train_indices = _maybe_subsample_training_data(
        x, y, max_train_n=max_train_n, subsample_seed=subsample_seed
    )

    target_device = torch.device(device) if device is not None else train_x.device
    train_x = train_x.to(device=target_device, dtype=dtype).contiguous()
    train_y = train_y.to(device=target_device, dtype=dtype).contiguous()

    resolved_grid_bounds = _resolve_grid_bounds(train_x, grid_bounds)
    resolved_grid_size = _resolve_grid_size(
        grid_size=grid_size,
        num_dims=train_x.size(-1),
        target_grid_points=target_grid_points,
        grid_bounds=resolved_grid_bounds,
    )

    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(noise_floor)
    ).to(device=target_device, dtype=dtype)

    model = _SKIExactGPModel(
        train_x,
        train_y,
        likelihood,
        kernel=kernel,
        grid_size=resolved_grid_size,
        grid_bounds=resolved_grid_bounds,
    ).to(device=target_device, dtype=dtype)

    with torch.no_grad():
        if init_lengthscale is not None:
            model.base_kernel.lengthscale = float(init_lengthscale)
        if init_outputscale is not None:
            model.covar_module.outputscale = float(init_outputscale)
        if init_noise is not None:
            likelihood.noise = max(float(init_noise), noise_floor)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    history: Dict[str, list] = {
        "iteration": [],
        "loss": [],
        "lengthscale": [],
        "outputscale": [],
        "noise": [],
        "forward_sec": [],
        "backward_sec": [],
        "elapsed_sec": [],
        "rss_gb": [],
    }

    best_loss = float("inf")
    best_model_state = None
    best_likelihood_state = None
    best_iteration = None
    start_time = time.time()

    context_managers = [
        gpytorch.settings.max_cholesky_size(0),
        gpytorch.settings.cg_tolerance(cg_tolerance),
        gpytorch.settings.eval_cg_tolerance(eval_cg_tolerance or cg_tolerance),
        gpytorch.settings.max_cg_iterations(max_cg_iterations),
        gpytorch.settings.max_preconditioner_size(max_preconditioner_size),
        gpytorch.settings.max_lanczos_quadrature_iterations(max_lanczos_quadrature_iterations),
        gpytorch.settings.num_trace_samples(num_trace_samples),
        gpytorch.settings.memory_efficient(memory_efficient),
        gpytorch.settings.use_toeplitz(use_toeplitz),
        gpytorch.settings.fast_computations(
            covar_root_decomposition=True,
            log_prob=True,
            solves=True,
        ),
    ]
    if checkpoint_size is not None:
        context_managers.append(gpytorch.beta_features.checkpoint_kernel(checkpoint_size))

    with ExitStack() as stack:
        for context_manager in context_managers:
            stack.enter_context(context_manager)

        for iteration in range(max_iters):
            optimizer.zero_grad(set_to_none=True)

            forward_start = time.time()
            output = model(train_x)
            loss = -mll(output, train_y)
            forward_sec = time.time() - forward_start

            backward_start = time.time()
            loss.backward()
            backward_sec = time.time() - backward_start

            optimizer.step()

            loss_value = float(loss.detach().item())
            lengthscale_value = float(model.base_kernel.lengthscale.detach().reshape(-1).mean().item())
            outputscale_value = float(model.covar_module.outputscale.detach().item())
            noise_value = float(likelihood.noise.detach().item())
            rss_value = _rss_gb()
            elapsed_sec = time.time() - start_time

            history["iteration"].append(iteration + 1)
            history["loss"].append(loss_value)
            history["lengthscale"].append(lengthscale_value)
            history["outputscale"].append(outputscale_value)
            history["noise"].append(noise_value)
            history["forward_sec"].append(forward_sec)
            history["backward_sec"].append(backward_sec)
            history["elapsed_sec"].append(elapsed_sec)
            history["rss_gb"].append(rss_value)

            if loss_value < best_loss:
                best_loss = loss_value
                best_model_state = copy.deepcopy(model.state_dict())
                best_likelihood_state = copy.deepcopy(likelihood.state_dict())
                best_iteration = iteration + 1

            if verbose:
                rss_text = f"{rss_value:.3f} GB" if rss_value is not None else "n/a"
                print(
                    f"[SKI] iter {iteration + 1:>3}/{max_iters}  "
                    f"loss={loss_value:.6g}  "
                    f"ls={lengthscale_value:.6g}  "
                    f"os={outputscale_value:.6g}  "
                    f"noise={noise_value:.6g}  "
                    f"time(fwd/bwd)={forward_sec:.2f}/{backward_sec:.2f}s  "
                    f"rss={rss_text}"
                )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    if best_likelihood_state is not None:
        likelihood.load_state_dict(best_likelihood_state)

    model.eval()
    likelihood.eval()

    return {
        "model": model,
        "likelihood": likelihood,
        "history": history,
        "train_x": train_x,
        "train_y": train_y,
        "train_indices": train_indices,
        "num_train": train_x.size(0),
        "num_total": total_n,
        "grid_size": resolved_grid_size,
        "grid_bounds": resolved_grid_bounds,
        "best_iteration": best_iteration,
        "best_loss": best_loss,
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(target_device),
        "fit_time_sec": time.time() - start_time,
        "settings": {
            "kernel": _canonicalize_kernel_name(kernel),
            "lr": lr,
            "noise_floor": noise_floor,
            "cg_tolerance": cg_tolerance,
            "eval_cg_tolerance": eval_cg_tolerance or cg_tolerance,
            "max_cg_iterations": max_cg_iterations,
            "max_preconditioner_size": max_preconditioner_size,
            "max_lanczos_quadrature_iterations": max_lanczos_quadrature_iterations,
            "num_trace_samples": num_trace_samples,
            "checkpoint_size": checkpoint_size,
            "use_toeplitz": use_toeplitz,
            "memory_efficient": memory_efficient,
        },
    }


__all__ = ["fit_ski_gp"]
