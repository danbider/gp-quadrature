"""
GPyTorch SVGP (Stochastic Variational GP) baseline for hyperparameter learning.

Uses ApproximateGP + VariationalStrategy + VariationalELBO.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, Optional, Union

import torch

try:
    import gpytorch
except ImportError as exc:
    gpytorch = None
    _GPYTORCH_IMPORT_ERROR = exc
else:
    _GPYTORCH_IMPORT_ERROR = None


def _require_gpytorch() -> None:
    if gpytorch is None:
        raise ImportError(
            "utils.svgp requires gpytorch in the active environment."
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
        f"Unsupported kernel '{kernel}'. "
        "Expected one of: SE, RBF, Matern32, Matern52."
    )


def _build_base_kernel(kernel: str):
    name = _canonicalize_kernel_name(kernel)
    if name == "rbf":
        return gpytorch.kernels.RBFKernel()
    if name == "matern32":
        return gpytorch.kernels.MaternKernel(nu=1.5)
    if name == "matern52":
        return gpytorch.kernels.MaternKernel(nu=2.5)
    raise AssertionError(f"Unhandled kernel name '{name}'")


class _SVGPModel(gpytorch.models.ApproximateGP):
    def __init__(
        self,
        inducing_points: torch.Tensor,
        kernel: str,
        learn_inducing_locations: bool = True,
    ):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.base_kernel = _build_base_kernel(kernel)
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)

    def forward(self, x: torch.Tensor):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def fit_svgp(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    kernel: str = "SE",
    num_inducing: int = 100,
    max_iters: int = 50,
    lr: float = 0.05,
    noise_floor: float = 1e-4,
    dtype: torch.dtype = torch.float32,
    device: Optional[Union[str, torch.device]] = None,
    init_lengthscale: Optional[float] = None,
    init_outputscale: Optional[float] = None,
    init_noise: Optional[float] = None,
    learn_inducing_locations: bool = True,
    inducing_seed: int = 0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Fit a GPyTorch SVGP model and return training logs.

    Uses ApproximateGP with VariationalStrategy and VariationalELBO.
    """
    _require_gpytorch()

    if x.ndim != 2:
        raise ValueError(f"x must have shape (N, d), got {tuple(x.shape)}")
    if y.ndim != 1:
        y = y.reshape(-1)

    target_device = torch.device(device) if device is not None else x.device
    train_x = x.to(device=target_device, dtype=dtype).contiguous()
    train_y = y.to(device=target_device, dtype=dtype).contiguous()

    n = train_x.size(0)
    m = min(num_inducing, n)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(inducing_seed)
    idx = torch.randperm(n, generator=gen)[:m]
    inducing_points = train_x[idx].clone()

    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(noise_floor)
    ).to(device=target_device, dtype=dtype)

    model = _SVGPModel(
        inducing_points=inducing_points,
        kernel=kernel,
        learn_inducing_locations=learn_inducing_locations,
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

    all_params = set(model.parameters()) | set(likelihood.parameters())
    optimizer = torch.optim.Adam(list(all_params), lr=lr)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=n)

    history: Dict[str, list] = {
        "iteration": [],
        "loss": [],
        "lengthscale": [],
        "outputscale": [],
        "noise": [],
        "elapsed_sec": [],
    }

    # Log initial hyperparameters at iter 0
    history["iteration"].append(0)
    history["loss"].append(float("nan"))
    history["lengthscale"].append(float(model.base_kernel.lengthscale.detach().reshape(-1).mean().item()))
    history["outputscale"].append(float(model.covar_module.outputscale.detach().item()))
    history["noise"].append(float(likelihood.noise.detach().item()))
    history["elapsed_sec"].append(0.0)

    best_loss = float("inf")
    best_model_state = None
    best_likelihood_state = None
    best_iteration = None
    start_time = time.time()

    for iteration in range(max_iters):
        optimizer.zero_grad(set_to_none=True)
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

        loss_val = float(loss.detach().item())
        ls_val = float(model.base_kernel.lengthscale.detach().reshape(-1).mean().item())
        os_val = float(model.covar_module.outputscale.detach().item())
        noise_val = float(likelihood.noise.detach().item())
        elapsed = time.time() - start_time

        history["iteration"].append(iteration + 1)
        history["loss"].append(loss_val)
        history["lengthscale"].append(ls_val)
        history["outputscale"].append(os_val)
        history["noise"].append(noise_val)
        history["elapsed_sec"].append(elapsed)

        if loss_val < best_loss:
            best_loss = loss_val
            best_model_state = copy.deepcopy(model.state_dict())
            best_likelihood_state = copy.deepcopy(likelihood.state_dict())
            best_iteration = iteration + 1

        if verbose:
            print(
                f"[SVGP M={m}] iter {iteration + 1:>3}/{max_iters}  "
                f"loss={loss_val:.6g}  ls={ls_val:.6g}  "
                f"os={os_val:.6g}  noise={noise_val:.6g}"
            )

    fit_time = time.time() - start_time

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
        "num_inducing": m,
        "fit_time_sec": fit_time,
        "best_iteration": best_iteration,
        "best_loss": best_loss,
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(target_device),
    }


__all__ = ["fit_svgp"]
