"""
GPyTorch SVGP baseline for negative-binomial count regression.

Uses ApproximateGP + VariationalStrategy (inducing points) + VariationalELBO with a
Negative-Binomial likelihood, so it can be compared head-to-head against the
Polya-Gamma EFGP regressor (``PolyagammaGPNegativeBinomialRegressor``).

Parameterization
----------------
EFGP's PG augmentation is locked to the exp/logit link: ``y ~ NB(r, p=sigmoid(f))``,
i.e. ``E[y] = r * exp(f)``. To fit the *identical* generative model with an
inducing-point SVGP we use the matching link here via
``torch.distributions.NegativeBinomial(total_count=r, logits=f)`` and learn ``r``.

This differs from gpytorch's built-in ``NegativeBinomialLikelihood`` only in the link
inside ``forward`` (that one uses a softplus mean link and learns ``probs``). Crucially,
this class mirrors the built-in's structure exactly -- it extends
``_OneDimensionalLikelihood`` and overrides **only** ``forward``, inheriting the same
Gauss-Hermite ``expected_log_prob``/``log_marginal`` quadrature -- so per-ELBO-step cost
is identical (see ``scratch/synth_nb_gp_compare.py`` for the parity micro-benchmark).
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, Optional, Union

import torch

try:
    import gpytorch
except ImportError as exc:  # pragma: no cover - environment guard
    gpytorch = None
    _GPYTORCH_IMPORT_ERROR = exc
else:
    _GPYTORCH_IMPORT_ERROR = None

# Reuse the exact SVGP scaffolding from the Gaussian baseline: ApproximateGP with
# CholeskyVariationalDistribution + VariationalStrategy, ZeroMean, ScaleKernel(RBF/Matern).
from utils.svgp import _SVGPModel, _require_gpytorch


class ExpLinkNegativeBinomialLikelihood(
    gpytorch.likelihoods._OneDimensionalLikelihood if gpytorch is not None else object
):
    """Negative-Binomial likelihood with the exp/logit link, matching EFGP.

    ``forward(f)`` returns ``NegativeBinomial(total_count=r, logits=f)`` so the latent
    GP is the log-mean: ``E[y] = r * exp(f)``. The dispersion ``r`` (``total_count``) is a
    learnable, strictly-positive parameter optimized jointly through the ELBO.

    Structurally identical to ``gpytorch.likelihoods.NegativeBinomialLikelihood``:
    extends ``_OneDimensionalLikelihood`` and overrides only ``forward`` (inheriting the
    Gauss-Hermite quadrature), so the per-step cost matches the built-in.
    """

    def __init__(
        self,
        init_total_count: float = 1.0,
        batch_shape: "torch.Size" = torch.Size([]),
        total_count_prior=None,
        total_count_constraint=None,
    ):
        _require_gpytorch()
        super().__init__()
        if total_count_constraint is None:
            total_count_constraint = gpytorch.constraints.Positive()

        self.register_parameter(
            name="raw_total_count",
            parameter=torch.nn.Parameter(torch.zeros(*batch_shape, 1)),
        )
        if total_count_prior is not None:
            self.register_prior(
                "total_count_prior",
                total_count_prior,
                lambda m: m.total_count,
                lambda m, v: m._set_total_count(v),
            )
        self.register_constraint("raw_total_count", total_count_constraint)
        self.total_count = init_total_count

    @property
    def total_count(self) -> torch.Tensor:
        return self.raw_total_count_constraint.transform(self.raw_total_count)

    @total_count.setter
    def total_count(self, value) -> None:
        self._set_total_count(value)

    def _set_total_count(self, value) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_total_count)
        self.initialize(
            raw_total_count=self.raw_total_count_constraint.inverse_transform(value)
        )

    def forward(self, function_samples: torch.Tensor, *args, **kwargs):
        # Squeeze the trailing singleton dispersion dim so it broadcasts against the
        # (..., n) function samples exactly like the built-in NB likelihood does.
        total_count = self.total_count.squeeze(-1)
        return torch.distributions.NegativeBinomial(
            total_count=total_count, logits=function_samples
        )


def _response_mean_from_latent(
    mean_f: torch.Tensor, var_f: torch.Tensor, total_count: float
) -> torch.Tensor:
    """E[y] under a Gaussian latent: r * exp(m + 0.5 v).

    Same formula as ``pg_classifier.negative_binomial_gaussian_mean`` (inlined to keep
    this module dependency-light -- no finufft/pg_classifier import).
    """
    return total_count * torch.exp(mean_f + 0.5 * var_f.clamp_min(0.0))


def fit_svgp_nb(
    x: torch.Tensor,
    y_counts: torch.Tensor,
    *,
    kernel: str = "SE",
    num_inducing: int = 200,
    max_iters: int = 100,
    lr: float = 0.05,
    batch_size: Optional[int] = None,
    init_total_count: float = 1.0,
    init_lengthscale: Optional[float] = 0.20,
    init_outputscale: Optional[float] = 1.0,
    dtype: torch.dtype = torch.float64,
    device: Optional[Union[str, torch.device]] = None,
    learn_inducing_locations: bool = True,
    inducing_seed: int = 0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Fit an SVGP negative-binomial regressor and return training logs + predictors.

    Uses ApproximateGP + VariationalStrategy (inducing points) + VariationalELBO with the
    exp-link :class:`ExpLinkNegativeBinomialLikelihood` (learns the dispersion ``r``).

    ``y_counts`` are raw integer counts (NOT standardized). If ``batch_size`` is set,
    training uses minibatch SGD in the standard SVGP fashion (Hensman et al. 2013), with
    ``VariationalELBO(num_data=n)`` keeping the ELBO unbiased; ``max_iters`` is then the
    number of epochs. If ``batch_size is None`` training is full-batch and ``max_iters``
    is the number of optimizer steps.
    """
    _require_gpytorch()

    if x.ndim != 2:
        raise ValueError(f"x must have shape (N, d), got {tuple(x.shape)}")
    if y_counts.ndim != 1:
        y_counts = y_counts.reshape(-1)

    target_device = torch.device(device) if device is not None else x.device
    train_x = x.to(device=target_device, dtype=dtype).contiguous()
    train_y = y_counts.to(device=target_device, dtype=dtype).contiguous()

    n = train_x.size(0)
    m = min(num_inducing, n)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(inducing_seed)
    idx = torch.randperm(n, generator=gen)[:m]
    inducing_points = train_x[idx].clone()

    likelihood = ExpLinkNegativeBinomialLikelihood(
        init_total_count=init_total_count
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

    model.train()
    likelihood.train()

    all_params = set(model.parameters()) | set(likelihood.parameters())
    optimizer = torch.optim.Adam(list(all_params), lr=lr)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=n)

    if batch_size is not None and batch_size < n:
        dataset = torch.utils.data.TensorDataset(train_x, train_y)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
    else:
        loader = None

    def _cur():
        ls = float(model.base_kernel.lengthscale.detach().reshape(-1).mean().item())
        os_ = float(model.covar_module.outputscale.detach().item())
        r = float(likelihood.total_count.detach().reshape(-1).mean().item())
        return ls, os_, r

    history: Dict[str, list] = {
        "iteration": [], "loss": [], "lengthscale": [],
        "outputscale": [], "total_count": [], "elapsed_sec": [],
    }
    ls0, os0, r0 = _cur()
    history["iteration"].append(0)
    history["loss"].append(float("nan"))
    history["lengthscale"].append(ls0)
    history["outputscale"].append(os0)
    history["total_count"].append(r0)
    history["elapsed_sec"].append(0.0)

    best_loss = float("inf")
    best_model_state = None
    best_likelihood_state = None
    best_iteration = None
    start_time = time.time()

    for iteration in range(max_iters):
        if loader is None:
            optimizer.zero_grad(set_to_none=True)
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            loss_val = float(loss.detach().item())
        else:
            batch_losses = []
            for xb, yb in loader:
                optimizer.zero_grad(set_to_none=True)
                output = model(xb)
                loss = -mll(output, yb)
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.detach().item()))
            loss_val = sum(batch_losses) / max(len(batch_losses), 1)

        ls_val, os_val, r_val = _cur()
        elapsed = time.time() - start_time
        history["iteration"].append(iteration + 1)
        history["loss"].append(loss_val)
        history["lengthscale"].append(ls_val)
        history["outputscale"].append(os_val)
        history["total_count"].append(r_val)
        history["elapsed_sec"].append(elapsed)

        if loss_val < best_loss:
            best_loss = loss_val
            best_model_state = copy.deepcopy(model.state_dict())
            best_likelihood_state = copy.deepcopy(likelihood.state_dict())
            best_iteration = iteration + 1

        if verbose:
            unit = "epoch" if loader is not None else "iter"
            print(
                f"[SVGP-NB M={m}] {unit} {iteration + 1:>3}/{max_iters}  "
                f"loss={loss_val:.6g}  ls={ls_val:.6g}  os={os_val:.6g}  r={r_val:.6g}"
            )

    fit_time = time.time() - start_time

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    if best_likelihood_state is not None:
        likelihood.load_state_dict(best_likelihood_state)

    model.eval()
    likelihood.eval()

    learned_total_count = float(likelihood.total_count.detach().reshape(-1).mean().item())

    def _latent_posterior(x_new: torch.Tensor):
        x_t = x_new.to(device=target_device, dtype=dtype)
        if x_t.ndim != 2:
            x_t = x_t.reshape(-1, train_x.size(1))
        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(1):
            latent = model(x_t)
            return latent.mean, latent.variance

    def predict_latent_mean(x_new: torch.Tensor) -> torch.Tensor:
        return _latent_posterior(x_new)[0]

    def predict_response_mean(x_new: torch.Tensor) -> torch.Tensor:
        mean_f, var_f = _latent_posterior(x_new)
        return _response_mean_from_latent(mean_f, var_f, learned_total_count)

    return {
        "model": model,
        "likelihood": likelihood,
        "history": history,
        "num_inducing": m,
        "fit_time_sec": fit_time,
        "best_iteration": best_iteration,
        "best_loss": best_loss,
        "total_count": learned_total_count,
        "predict_latent_mean": predict_latent_mean,
        "predict_response_mean": predict_response_mean,
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(target_device),
    }


__all__ = ["ExpLinkNegativeBinomialLikelihood", "fit_svgp_nb"]
