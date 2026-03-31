from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import pytorch_finufft.functional as pff
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from torch import vmap
from torch.fft import fftn, ifftn
from torch.optim import Adam

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from cg import ConjugateGradients
from efgpnd import NUFFT, ToeplitzND, compute_convolution_vector_vectorized_dD
from kernels import SquaredExponential
from utils.kernels import get_xis

@dataclass
class qVariationalParams:
    """Compatibility container for the diagonal PG variational parameters."""

    n: int
    device: torch.device | None = None
    dtype: torch.dtype = torch.float64
    init_delta: float | torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.init_delta is None:
            self.Delta = torch.full((self.n,), 0.25, device=self.device, dtype=self.dtype)
            return

        if torch.is_tensor(self.init_delta):
            delta = self.init_delta.to(device=self.device, dtype=self.dtype)
            if delta.shape != (self.n,):
                raise ValueError("init_delta tensor must have shape (n,).")
            self.Delta = delta.clone()
            return

        self.Delta = torch.full((self.n,), float(self.init_delta), device=self.device, dtype=self.dtype)


@dataclass
class _VariationalState:
    delta: torch.Tensor
    mean: torch.Tensor | None = None
    sigma_diag: torch.Tensor | None = None
    probes: torch.Tensor | None = None


@dataclass
class _SpectralState:
    xis: torch.Tensor
    h: float
    mtot: int
    ws: torch.Tensor
    ws2: torch.Tensor
    Dprime: torch.Tensor
    toeplitz: ToeplitzND
    nufft_op: NUFFT
    fadj: Callable[[torch.Tensor], torch.Tensor]
    fwd: Callable[[torch.Tensor], torch.Tensor]
    fadj_batched: Callable[[torch.Tensor], torch.Tensor]
    fwd_batched: Callable[[torch.Tensor], torch.Tensor]
    out_shape: tuple[int, ...]


@dataclass(frozen=True)
class _PreparedTargets:
    values: np.ndarray
    metadata: dict[str, object]


class _PGLikelihood:
    history_key = "fit_metric"
    history_label = "fit_metric"
    training_attr = "training_metric_"

    def prepare_targets(self, y_arr: np.ndarray) -> _PreparedTargets:
        raise NotImplementedError

    def kappa(self, targets: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def pg_b(self, targets: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def response_mean(self, mean: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def fit_metric(self, mean: torch.Tensor, variance: torch.Tensor, targets: torch.Tensor) -> float:
        return float("nan")


@dataclass(frozen=True)
class _PGBernoulliLikelihood(_PGLikelihood):
    history_key: str = "approx_accuracy"
    history_label: str = "approx_acc"
    training_attr: str = "training_accuracy_"

    def prepare_targets(self, y_arr: np.ndarray) -> _PreparedTargets:
        classes = np.unique(y_arr)
        if classes.size != 2:
            raise ValueError("PolyagammaGPClassifier only supports binary classification.")
        return _PreparedTargets(
            values=(y_arr == classes[1]).astype(np.float64),
            metadata={"classes_": classes},
        )

    def kappa(self, targets: torch.Tensor) -> torch.Tensor:
        return targets - 0.5

    def pg_b(self, targets: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(targets)

    def response_mean(self, mean: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
        return approximate_logistic_gaussian_prob(mean, variance)

    def fit_metric(self, mean: torch.Tensor, variance: torch.Tensor, targets: torch.Tensor) -> float:
        return float(
            (
                self.response_mean(mean, variance).gt(0.5)
                == targets.bool()
            )
            .float()
            .mean()
            .item()
        )


@dataclass(frozen=True)
class _PGNegativeBinomialLikelihood(_PGLikelihood):
    total_count: float
    history_key: str = "mean_count_mae"
    history_label: str = "count_mae"
    training_attr: str = "training_mean_absolute_error_"

    def __post_init__(self) -> None:
        if self.total_count <= 0:
            raise ValueError("total_count must be positive.")

    def prepare_targets(self, y_arr: np.ndarray) -> _PreparedTargets:
        if np.any(y_arr < 0):
            raise ValueError("Negative binomial targets must be nonnegative.")
        if not np.allclose(y_arr, np.round(y_arr)):
            raise ValueError("Negative binomial targets must be integer-valued.")
        return _PreparedTargets(values=np.round(y_arr).astype(np.float64), metadata={})

    def kappa(self, targets: torch.Tensor) -> torch.Tensor:
        return 0.5 * (targets - self.total_count)

    def pg_b(self, targets: torch.Tensor) -> torch.Tensor:
        return targets + self.total_count

    def response_mean(self, mean: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
        return negative_binomial_gaussian_mean(mean, variance, total_count=self.total_count)

    def fit_metric(self, mean: torch.Tensor, variance: torch.Tensor, targets: torch.Tensor) -> float:
        predicted = self.response_mean(mean, variance)
        return float(torch.mean(torch.abs(predicted - targets)).item())


def approximate_logistic_gaussian_prob(
    mean: torch.Tensor,
    variance: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Approximate E[sigmoid(F)] when F is Gaussian.

    The notebook uses
      sigmoid(mean / sqrt(1 + pi * variance / 8)),
    which is a standard logistic-Gaussian moment approximation. When no
    variance is provided, this falls back to sigmoid(mean).
    """

    if variance is None:
        return torch.sigmoid(mean)

    safe_var = variance.clamp_min(0.0)
    denom = torch.sqrt(1.0 + (math.pi / 8.0) * safe_var)
    return torch.sigmoid(mean / denom)


def negative_binomial_gaussian_mean(
    mean: torch.Tensor,
    variance: torch.Tensor,
    *,
    total_count: float,
) -> torch.Tensor:
    safe_var = variance.clamp_min(0.0)
    return total_count * torch.exp(mean + 0.5 * safe_var)


@lru_cache(maxsize=None)
def _gauss_hermite_normal_rule(num_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive.")
    base_nodes, base_weights = np.polynomial.hermite.hermgauss(num_nodes)
    nodes = np.sqrt(2.0) * base_nodes
    weights = base_weights / np.sqrt(np.pi)
    return nodes.astype(np.float64), weights.astype(np.float64)


def _expected_log_sigmoid_negative_gaussian(
    mean: torch.Tensor,
    variance: torch.Tensor,
    *,
    quadrature_nodes: int,
) -> torch.Tensor:
    mean_flat = mean.reshape(-1)
    safe_var = variance.reshape(-1).clamp_min(0.0)
    std_flat = torch.sqrt(safe_var)
    node_np, weight_np = _gauss_hermite_normal_rule(quadrature_nodes)
    nodes = torch.as_tensor(node_np, device=mean.device, dtype=mean.dtype)
    weights = torch.as_tensor(weight_np, device=mean.device, dtype=mean.dtype)
    eval_points = mean_flat[:, None] + std_flat[:, None] * nodes[None, :]
    expectations = (F.logsigmoid(-eval_points) * weights[None, :]).sum(dim=1)
    return expectations.reshape(mean.shape)


def _negative_binomial_total_count_gradient(
    targets: torch.Tensor,
    mean: torch.Tensor,
    variance: torch.Tensor,
    *,
    total_count: float | torch.Tensor,
    quadrature_nodes: int,
) -> torch.Tensor:
    total_count_t = torch.as_tensor(total_count, device=mean.device, dtype=mean.dtype)
    expected_log_sigmoid_neg = _expected_log_sigmoid_negative_gaussian(
        mean,
        variance,
        quadrature_nodes=quadrature_nodes,
    )
    return torch.sum(
        torch.special.digamma(targets + total_count_t)
        - torch.special.digamma(total_count_t)
        + expected_log_sigmoid_neg
    )


def _pg_omega_expectation(
    c: torch.Tensor,
    pg_b: torch.Tensor,
) -> torch.Tensor:
    safe_c = c.clamp_min(1e-12)
    mean = 0.5 * pg_b * torch.tanh(0.5 * safe_c) / safe_c
    return torch.where(c > 1e-8, mean, 0.25 * pg_b)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype == "float64":
        return torch.float64
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def _sample_rademacher(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int | None,
) -> torch.Tensor:
    if seed is None:
        samples = torch.rand(shape, dtype=dtype)
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        samples = torch.rand(shape, generator=generator, dtype=dtype)
    return samples.mul_(2.0).floor_().mul_(2.0).sub_(1.0).to(device=device)


def _make_kernel(
    kernel: str,
    *,
    dimension: int,
    lengthscale: float,
    variance: float,
) -> SquaredExponential:
    kernel_name = kernel.lower()
    if kernel_name not in {"squared_exponential", "se", "rbf"}:
        raise ValueError("Only the squared exponential kernel is supported in v1.")
    return SquaredExponential(
        dimension=dimension,
        init_lengthscale=lengthscale,
        init_variance=variance,
    )


def _build_spectral_state(
    X: torch.Tensor,
    kernel: SquaredExponential,
    *,
    spectral_eps: float,
    trunc_eps: float,
    nufft_eps: float,
    rdtype: torch.dtype,
    cdtype: torch.dtype,
    device: torch.device,
) -> _SpectralState:
    x0 = X.min(dim=0).values
    x1 = X.max(dim=0).values
    L = (x1 - x0).max()
    d = X.shape[1]

    xis_1d, h, mtot = get_xis(
        kernel_obj=kernel,
        eps=spectral_eps,
        L=L,
        use_integral=True,
        l2scaled=False,
        trunc_eps=trunc_eps,
    )
    grids = torch.meshgrid(*(xis_1d for _ in range(d)), indexing="ij")
    xis = torch.stack(grids, dim=-1).view(-1, d)

    spec_density = kernel.spectral_density(xis).to(dtype=rdtype)
    ws2 = (spec_density * h**d).to(device=device, dtype=cdtype)
    ws = torch.sqrt(ws2)

    m_conv = (mtot - 1) // 2
    v_kernel = compute_convolution_vector_vectorized_dD(m_conv, X, h).to(dtype=cdtype)
    toeplitz = ToeplitzND(v_kernel, force_pow2=True)
    Dprime = (h**d * kernel.spectral_grad(xis)).to(device=device, dtype=cdtype)

    out_shape = (mtot,) * d
    nufft_op = NUFFT(
        X,
        torch.zeros_like(X),
        h,
        nufft_eps,
        cdtype=cdtype,
        device=device,
    )
    fadj = lambda v: nufft_op.type1(v, out_shape=out_shape).reshape(-1)
    fwd = lambda fk: nufft_op.type2(fk, out_shape=out_shape)
    fadj_batched = vmap(fadj, in_dims=0, out_dims=0)
    fwd_batched = vmap(fwd, in_dims=0, out_dims=0)

    return _SpectralState(
        xis=xis.to(device=device, dtype=rdtype),
        h=h,
        mtot=mtot,
        ws=ws,
        ws2=ws2,
        Dprime=Dprime,
        toeplitz=toeplitz,
        nufft_op=nufft_op,
        fadj=fadj,
        fwd=fwd,
        fadj_batched=fadj_batched,
        fwd_batched=fwd_batched,
        out_shape=out_shape,
    )


def _build_weighted_toeplitz(
    delta: torch.Tensor,
    spectral: _SpectralState,
) -> ToeplitzND:
    weights = delta.to(dtype=spectral.ws.dtype, device=delta.device).flatten()
    conv_shape = tuple(2 * n - 1 for n in spectral.out_shape)
    v_weighted = spectral.nufft_op.type1(weights, out_shape=conv_shape)
    return ToeplitzND(v_weighted.to(dtype=spectral.ws.dtype), force_pow2=True)


def _make_sigma_apply(
    spectral: _SpectralState,
    delta: torch.Tensor,
    *,
    cg_tol: float,
    use_exact_weighted_toeplitz_operator: bool = False,
    weighted_toeplitz: ToeplitzND | None = None,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], dict[str, int]]:
    info = {"cg_iters": 0}
    delta_complex = delta.to(dtype=spectral.ws.dtype, device=delta.device)
    weighted_toeplitz_op = weighted_toeplitz
    if use_exact_weighted_toeplitz_operator and weighted_toeplitz_op is None:
        weighted_toeplitz_op = _build_weighted_toeplitz(delta_complex, spectral)

    def sigma_apply(z: torch.Tensor) -> torch.Tensor:
        vector_input = z.dim() == 1
        if vector_input:
            z = z.unsqueeze(0)
        z = z.to(dtype=spectral.ws.dtype)

        Delta = delta_complex.view(1, -1)
        rhs = spectral.ws * spectral.fadj_batched(z)

        def A_feat(u: torch.Tensor) -> torch.Tensor:
            if weighted_toeplitz_op is not None:
                if u.dim() == 1:
                    t = spectral.ws * u
                    return u + spectral.ws * weighted_toeplitz_op(t)
                t = u * spectral.ws[None, :]
                return u + spectral.ws[None, :] * weighted_toeplitz_op(t)
            psi_u = spectral.fwd_batched(spectral.ws * u)
            return u + spectral.ws * spectral.fadj_batched(Delta * psi_u)

        cg = ConjugateGradients(
            A_feat,
            rhs,
            x0=torch.zeros_like(rhs),
            tol=cg_tol,
            max_iter=2000,
            early_stopping=True,
        )
        x = cg.solve()
        info["cg_iters"] = cg.iters_completed

        result = spectral.fwd_batched(spectral.ws * x).real
        if vector_input:
            return result.squeeze(0)
        return result

    return sigma_apply, info


def _make_feature_space_solver(
    delta: torch.Tensor,
    spectral: _SpectralState,
    *,
    cg_tol: float,
    use_exact_weighted_toeplitz_operator: bool = False,
    weighted_toeplitz: ToeplitzND | None = None,
) -> tuple[
    Callable[[torch.Tensor], tuple[torch.Tensor, int]],
    Callable[[torch.Tensor], torch.Tensor],
    dict[str, int],
]:
    omega = delta.to(dtype=spectral.ws.dtype, device=delta.device).flatten()
    D2_real = spectral.ws2.real
    eps_d = max(float(D2_real.mean()) * 1e-14, 1e-14)
    Ds = torch.sqrt(torch.clamp(D2_real, min=eps_d)).to(dtype=spectral.ws.dtype)
    Dsinv = 1.0 / Ds
    info = {"cg_iters": 0}
    weighted_toeplitz_op = weighted_toeplitz
    if use_exact_weighted_toeplitz_operator and weighted_toeplitz_op is None:
        weighted_toeplitz_op = _build_weighted_toeplitz(omega, spectral)

    def apply_omega(v: torch.Tensor) -> torch.Tensor:
        if v.dim() == 2:
            return omega[:, None] * v
        return omega * v

    def apply_S(Y: torch.Tensor) -> torch.Tensor:
        if weighted_toeplitz_op is not None:
            if Y.dim() == 1:
                t = Ds * Y
                return Ds * weighted_toeplitz_op(t)
            t = Y * Ds[None, :]
            return Ds[None, :] * weighted_toeplitz_op(t)
        if Y.dim() == 1:
            t = Ds * Y
            u = spectral.fwd(t)
            u = apply_omega(u)
            v = spectral.fadj(u)
            return Ds * v
        t = Y * Ds[None, :]
        u = spectral.fwd_batched(t).mT
        u = apply_omega(u)
        v = spectral.fadj_batched(u.T).T
        return (Ds[:, None] * v).T

    def apply_IpS(Y: torch.Tensor) -> torch.Tensor:
        return Y + apply_S(Y)

    def solve_A_beta(q: torch.Tensor) -> tuple[torch.Tensor, int]:
        rhs = Ds * q if q.dim() == 1 else q * Ds[None, :]

        cg = ConjugateGradients(
            apply_IpS,
            rhs,
            x0=torch.zeros_like(rhs),
            tol=cg_tol,
            max_iter=2000,
            early_stopping=True,
        )
        y = cg.solve()
        info["cg_iters"] = int(cg.iters_completed)
        beta = Dsinv * y if q.dim() == 1 else y * Dsinv[None, :]
        return beta, cg.iters_completed

    return solve_A_beta, apply_omega, info


def _run_estep(
    targets: torch.Tensor,
    kappa: torch.Tensor,
    pg_b: torch.Tensor,
    likelihood: _PGLikelihood,
    variational: _VariationalState,
    spectral: _SpectralState,
    *,
    max_iters: int,
    rho0: float,
    gamma: float,
    tol: float,
    n_probes: int,
    cg_tol: float,
    reuse_probes: bool,
    use_exact_weighted_toeplitz_operator: bool = False,
    seed: int | None,
    verbose: int,
) -> tuple[_VariationalState, dict[str, float]]:
    sigma_apply, sigma_info = _make_sigma_apply(
        spectral,
        variational.delta,
        cg_tol=cg_tol,
        use_exact_weighted_toeplitz_operator=use_exact_weighted_toeplitz_operator,
    )
    probes = variational.probes
    fit_metric = float("nan")
    residual = float("inf")

    with torch.no_grad():
        for it in range(max_iters):
            if n_probes > 0 and (probes is None or probes.shape[0] != n_probes or not reuse_probes or it == 0):
                probe_seed = None if seed is None else seed + 17 * (it + 1)
                probes = _sample_rademacher(
                    (n_probes, targets.numel()),
                    device=targets.device,
                    dtype=targets.dtype,
                    seed=probe_seed,
                )

            if n_probes > 0:
                Z = torch.cat([kappa[None, :], probes], dim=0)
            else:
                Z = kappa[None, :]

            S_all = sigma_apply(Z)
            mean = S_all[0]
            Sz = S_all[1:] if n_probes > 0 else torch.empty((0, targets.numel()), device=targets.device, dtype=targets.dtype)
            sigma_diag = (probes * Sz).mean(dim=0) if n_probes > 0 else torch.zeros_like(mean)

            c2 = (sigma_diag + mean.pow(2)).clamp_min(1e-12)
            c = torch.sqrt(c2)
            Lambda = _pg_omega_expectation(c, pg_b)

            rho = rho0 / (1.0 + gamma * it)
            variational.delta.mul_(1.0 - rho).add_(rho * Lambda)
            variational.delta.clamp_(min=0.0)

            residual = float((variational.delta - Lambda).abs().max().item())
            variational.mean = mean
            variational.sigma_diag = sigma_diag
            variational.probes = probes
            fit_metric = likelihood.fit_metric(mean, sigma_diag, targets)
            if verbose > 1:
                print(
                    f"E-step it {it:3d} rho={rho:.3f} max|Delta-Lambda|={residual:.3e} "
                    f"{likelihood.history_label}={fit_metric:.4f}"
                )
            if residual < tol:
                break

    return variational, {
        "residual": residual,
        "metric": fit_metric,
        "cg_iters": float(sigma_info["cg_iters"]),
    }


def _compute_mstep_gradient(
    kappa: torch.Tensor,
    delta: torch.Tensor,
    spectral: _SpectralState,
    *,
    n_probes: int,
    cg_tol: float,
    use_exact_weighted_toeplitz_operator: bool = False,
    seed: int | None,
) -> dict[str, torch.Tensor]:
    solve_A_beta, apply_omega, solve_info = _make_feature_space_solver(
        delta,
        spectral,
        cg_tol=cg_tol,
        use_exact_weighted_toeplitz_operator=use_exact_weighted_toeplitz_operator,
    )

    probes = _sample_rademacher(
        (n_probes, kappa.numel()),
        device=kappa.device,
        dtype=kappa.dtype,
        seed=None if seed is None else seed + 10_000,
    ).to(dtype=spectral.ws.dtype)
    Q_block = spectral.fadj_batched(probes)
    q_y = spectral.fadj_batched(kappa.to(dtype=spectral.ws.dtype).unsqueeze(0))

    Q_all = torch.cat([Q_block, q_y], dim=0)
    beta_all, cg_iters = solve_A_beta(Q_all)
    beta_probes = beta_all[:-1, :]
    beta_x = beta_all[-1, :]

    Rfeat = spectral.fadj_batched(apply_omega(probes.mT).T).T
    X = Rfeat.conj() * beta_probes.T
    vals = (X.mT @ spectral.Dprime).real
    term2 = vals.mean(dim=0)

    abs2 = (beta_x.conj() * beta_x).real
    term1 = spectral.Dprime.real.T @ abs2
    grad = 0.5 * (term1 - term2)

    return {
        "grad": grad,
        "term1": term1,
        "term2": term2,
        "beta_mean": beta_x,
        "cg_iters": torch.tensor(float(cg_iters)),
    }


def _solve_beta_mean(
    kappa: torch.Tensor,
    delta: torch.Tensor,
    spectral: _SpectralState,
    *,
    cg_tol: float,
    use_exact_weighted_toeplitz_operator: bool = False,
) -> tuple[torch.Tensor, int]:
    solve_A_beta, _, solve_info = _make_feature_space_solver(
        delta,
        spectral,
        cg_tol=cg_tol,
        use_exact_weighted_toeplitz_operator=use_exact_weighted_toeplitz_operator,
    )
    q_y = spectral.fadj(kappa.to(dtype=spectral.ws.dtype))
    beta, cg_iters = solve_A_beta(q_y)
    return beta, cg_iters


def _predictive_mean(
    X_new: torch.Tensor,
    beta_mean: torch.Tensor,
    spectral: _SpectralState,
    *,
    nufft_eps: float,
) -> torch.Tensor:
    eval_op = NUFFT(
        X_new,
        torch.zeros_like(X_new),
        spectral.h,
        nufft_eps,
        cdtype=spectral.ws.dtype,
        device=X_new.device,
    )
    return eval_op.type2(spectral.ws2 * beta_mean, out_shape=spectral.out_shape).real


def _predictive_latent_moments(
    X_new: torch.Tensor,
    beta_mean: torch.Tensor | None,
    delta: torch.Tensor,
    spectral: _SpectralState,
    *,
    cg_tol: float,
    nufft_eps: float,
    use_exact_weighted_toeplitz_operator: bool = False,
    batch_size: int | None = None,
    weighted_toeplitz: ToeplitzND | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if X_new.ndim != 2:
        raise ValueError("X_new must have shape (n_samples, n_features).")

    n_test = X_new.shape[0]
    if n_test == 0:
        empty = torch.empty((0,), device=X_new.device, dtype=delta.dtype)
        return empty, empty

    solve_A_beta, _, _ = _make_feature_space_solver(
        delta,
        spectral,
        cg_tol=cg_tol,
        use_exact_weighted_toeplitz_operator=use_exact_weighted_toeplitz_operator,
        weighted_toeplitz=weighted_toeplitz,
    )
    block_size = n_test if batch_size is None else max(1, min(batch_size, n_test))

    means: list[torch.Tensor] = []
    variances: list[torch.Tensor] = []
    ws2_row = spectral.ws2.unsqueeze(0)
    for start in range(0, n_test, block_size):
        stop = min(start + block_size, n_test)
        X_block = X_new[start:stop]
        block_op = NUFFT(
            X_block,
            torch.zeros_like(X_block),
            spectral.h,
            nufft_eps,
            cdtype=spectral.ws.dtype,
            device=X_block.device,
        )

        eye_block = torch.eye(
            X_block.shape[0],
            device=X_block.device,
            dtype=spectral.ws.dtype,
        )
        phi_block = block_op.type1(eye_block, out_shape=spectral.out_shape).reshape(X_block.shape[0], -1)

        if beta_mean is None:
            mean_block = torch.empty((X_block.shape[0],), device=X_block.device, dtype=delta.dtype)
        else:
            mean_block = torch.sum(
                phi_block.conj() * (ws2_row * beta_mean.unsqueeze(0)),
                dim=1,
            ).real

        beta_block, _ = solve_A_beta(phi_block)
        variance_block = torch.sum(
            phi_block.conj() * (ws2_row * beta_block),
            dim=1,
        ).real.clamp_min(0.0)

        means.append(mean_block)
        variances.append(variance_block)

    return torch.cat(means, dim=0), torch.cat(variances, dim=0)


def _predictive_variance(
    X_new: torch.Tensor,
    delta: torch.Tensor,
    spectral: _SpectralState,
    *,
    cg_tol: float,
    nufft_eps: float,
    use_exact_weighted_toeplitz_operator: bool = False,
    batch_size: int | None = None,
    weighted_toeplitz: ToeplitzND | None = None,
) -> torch.Tensor:
    _, variance = _predictive_latent_moments(
        X_new,
        None,
        delta,
        spectral,
        cg_tol=cg_tol,
        nufft_eps=nufft_eps,
        use_exact_weighted_toeplitz_operator=use_exact_weighted_toeplitz_operator,
        batch_size=batch_size,
        weighted_toeplitz=weighted_toeplitz,
    )
    return variance


def _estimate_stochastic_variance_sums(
    delta: torch.Tensor,
    spectral: _SpectralState,
    *,
    cg_tol: float,
    n_probes: int,
    use_exact_weighted_toeplitz_operator: bool = False,
    seed: int | None = None,
    weighted_toeplitz: ToeplitzND | None = None,
) -> tuple[torch.Tensor, dict[str, int]]:
    if n_probes <= 0:
        raise ValueError("n_probes must be positive for stochastic predictive variance.")

    solve_A_beta, _, solve_info = _make_feature_space_solver(
        delta,
        spectral,
        cg_tol=cg_tol,
        use_exact_weighted_toeplitz_operator=use_exact_weighted_toeplitz_operator,
        weighted_toeplitz=weighted_toeplitz,
    )
    n_features = spectral.ws.numel()
    eta_real = _sample_rademacher(
        (n_probes, n_features),
        device=delta.device,
        dtype=delta.dtype,
        seed=seed,
    )
    etas = eta_real.to(dtype=spectral.ws.dtype)
    beta_probes, cg_iters = solve_A_beta(etas)
    gammas = spectral.ws2.unsqueeze(0) * beta_probes

    grid_shape = spectral.out_shape
    corr_shape = tuple(2 * m - 1 for m in grid_shape)
    fft_dims = tuple(range(1, len(grid_shape) + 1))
    gamma_grid = gammas.reshape((n_probes,) + grid_shape)
    eta_grid = etas.reshape((n_probes,) + grid_shape)
    gamma_fft = fftn(gamma_grid, s=corr_shape, dim=fft_dims)
    eta_fft = fftn(eta_grid, s=corr_shape, dim=fft_dims)
    est_sums = ifftn(gamma_fft * torch.conj(eta_fft), s=corr_shape, dim=fft_dims).mean(dim=0)

    info = {
        "cg_iters": int(cg_iters),
        "n_probes": int(n_probes),
    }
    return est_sums, info


def _evaluate_stochastic_variance_sums(
    est_sums: torch.Tensor,
    X_new: torch.Tensor,
    *,
    h: float,
    nufft_eps: float,
    cdtype: torch.dtype,
) -> torch.Tensor:
    eval_op = NUFFT(
        X_new,
        torch.zeros_like(X_new),
        h,
        nufft_eps,
        cdtype=cdtype,
        device=X_new.device,
    )
    variance = pff.finufft_type2(
        eval_op.phi,
        est_sums.contiguous(),
        eps=nufft_eps,
        isign=+1,
        modeord=True,
    ).real
    return variance.clamp_min(0.0)


def _predictive_variance_stochastic(
    X_new: torch.Tensor,
    delta: torch.Tensor,
    spectral: _SpectralState,
    *,
    cg_tol: float,
    nufft_eps: float,
    n_probes: int,
    use_exact_weighted_toeplitz_operator: bool = False,
    seed: int | None = None,
    est_sums: torch.Tensor | None = None,
    weighted_toeplitz: ToeplitzND | None = None,
) -> tuple[torch.Tensor, dict[str, int]]:
    if est_sums is None:
        est_sums, info = _estimate_stochastic_variance_sums(
            delta,
            spectral,
            cg_tol=cg_tol,
            n_probes=n_probes,
            use_exact_weighted_toeplitz_operator=use_exact_weighted_toeplitz_operator,
            seed=seed,
            weighted_toeplitz=weighted_toeplitz,
        )
    else:
        info = {
            "cg_iters": 0,
            "n_probes": int(n_probes),
        }

    variance = _evaluate_stochastic_variance_sums(
        est_sums.to(device=X_new.device, dtype=spectral.ws.dtype),
        X_new,
        h=spectral.h,
        nufft_eps=nufft_eps,
        cdtype=spectral.ws.dtype,
    )
    return variance, info


def _chebyshev_lobatto_nodes(a: float, b: float, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    if n_nodes < 2:
        raise ValueError("predictive_variance_chebyshev_nodes must be at least 2.")
    k = np.arange(n_nodes, dtype=np.float64)
    nodes_std = np.cos(np.pi * k / (n_nodes - 1))
    weights = np.ones(n_nodes, dtype=np.float64)
    weights[0] = 0.5
    weights[-1] = 0.5
    weights *= (-1.0) ** k
    nodes = 0.5 * (a + b) + 0.5 * (b - a) * nodes_std
    scale = 2.0 / (b - a) if b > a else 1.0
    order = np.argsort(nodes)
    return nodes[order], (weights * scale)[order]


def _barycentric_interpolation_matrix(
    nodes: np.ndarray,
    weights: np.ndarray,
    targets: np.ndarray,
    *,
    atol: float = 1e-14,
) -> np.ndarray:
    nodes = np.asarray(nodes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    diff = targets[:, None] - nodes[None, :]
    mat = np.empty((targets.size, nodes.size), dtype=np.float64)

    close = np.isclose(diff, 0.0, atol=atol, rtol=0.0)
    matched_rows = close.any(axis=1)
    if np.any(matched_rows):
        matched_idx = np.argmax(close[matched_rows], axis=1)
        mat[matched_rows] = 0.0
        mat[np.where(matched_rows)[0], matched_idx] = 1.0

    unmatched_rows = ~matched_rows
    if np.any(unmatched_rows):
        diff_u = diff[unmatched_rows]
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = weights[None, :] / diff_u
        mat[unmatched_rows] = raw / raw.sum(axis=1, keepdims=True)
    return mat


def _tensor_product_barycentric_interpolate(
    node_values: torch.Tensor,
    interpolation_mats: list[torch.Tensor],
) -> torch.Tensor:
    dimension = len(interpolation_mats)
    if dimension == 0:
        raise ValueError("At least one interpolation matrix is required.")
    letters = "abcdefghijklmnopqrstuvwxyz"
    if dimension > len(letters):
        raise ValueError("Chebyshev interpolation currently supports at most 26 dimensions.")
    node_letters = letters[:dimension]
    operands: list[torch.Tensor] = []
    input_terms: list[str] = []
    for idx, letter in enumerate(node_letters):
        operands.append(interpolation_mats[idx])
        input_terms.append(f"n{letter}")
    operands.append(node_values)
    input_terms.append(node_letters)
    expr = ",".join(input_terms) + "->n"
    return torch.einsum(expr, *operands)


def _predictive_variance_chebyshev(
    X_new: torch.Tensor,
    delta: torch.Tensor,
    spectral: _SpectralState,
    *,
    cg_tol: float,
    nufft_eps: float,
    n_nodes_per_dim: int,
    use_exact_weighted_toeplitz_operator: bool = False,
    batch_size: int | None = None,
    weighted_toeplitz: ToeplitzND | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if X_new.ndim != 2:
        raise ValueError("X_new must have shape (n_samples, n_features).")
    if X_new.shape[0] == 0:
        empty = torch.empty((0,), device=X_new.device, dtype=delta.dtype)
        return empty, {
            "cg_iters": 0.0,
            "n_nodes_total": 0.0,
        }

    dimension = X_new.shape[1]
    node_axes_np: list[np.ndarray] = []
    barycentric_mats: list[torch.Tensor] = []
    for dim in range(dimension):
        coord = X_new[:, dim].detach().cpu().numpy()
        lower = float(coord.min())
        upper = float(coord.max())
        if np.isclose(lower, upper):
            pad = max(abs(lower), 1.0) * 1e-6
            lower -= pad
            upper += pad
        nodes_np, weights_np = _chebyshev_lobatto_nodes(lower, upper, n_nodes_per_dim)
        interp_np = _barycentric_interpolation_matrix(nodes_np, weights_np, coord)
        node_axes_np.append(nodes_np)
        barycentric_mats.append(
            torch.as_tensor(interp_np, device=X_new.device, dtype=delta.dtype)
        )

    node_axes_t = [
        torch.as_tensor(axis, device=X_new.device, dtype=X_new.dtype) for axis in node_axes_np
    ]
    mesh = torch.meshgrid(*node_axes_t, indexing="ij")
    node_points = torch.stack([g.reshape(-1) for g in mesh], dim=1)
    node_variance = _predictive_variance(
        node_points,
        delta,
        spectral,
        cg_tol=cg_tol,
        nufft_eps=nufft_eps,
        use_exact_weighted_toeplitz_operator=use_exact_weighted_toeplitz_operator,
        batch_size=batch_size,
        weighted_toeplitz=weighted_toeplitz,
    )
    node_shape = (n_nodes_per_dim,) * dimension
    interpolated = _tensor_product_barycentric_interpolate(
        node_variance.reshape(node_shape).to(dtype=delta.dtype),
        barycentric_mats,
    ).clamp_min(0.0)

    info = {
        "cg_iters": float(node_points.shape[0]),
        "n_nodes_total": float(node_points.shape[0]),
    }
    return interpolated, info


def _dense_pg_reference_gradient(
    X: torch.Tensor,
    y: torch.Tensor,
    mean: torch.Tensor,
    delta: torch.Tensor,
    spectral: _SpectralState,
    *,
    jitter: float = 1e-8,
) -> torch.Tensor:
    cdtype = spectral.ws.dtype
    rdtype = mean.dtype
    F_train = torch.exp(2.0 * math.pi * 1j * (X @ spectral.xis.T)).to(dtype=cdtype)
    Kff = F_train @ torch.diag(spectral.ws2.to(dtype=cdtype)) @ F_train.T.conj()
    Kff = Kff.real.to(dtype=rdtype)

    K = Kff + jitter * torch.eye(X.shape[0], device=X.device, dtype=rdtype)
    L = torch.linalg.cholesky(K, upper=False)
    I = torch.eye(X.shape[0], device=X.device, dtype=rdtype)
    K_inv = torch.cholesky_solve(I, L, upper=False)

    S_inv = K_inv + torch.diag(delta.to(dtype=rdtype))
    LS = torch.linalg.cholesky(S_inv, upper=False)
    S = torch.cholesky_inverse(LS, upper=False)

    dK_dls = F_train @ torch.diag(spectral.Dprime[:, 0].to(dtype=cdtype)) @ F_train.T.conj()
    dK_dvar = F_train @ torch.diag(spectral.Dprime[:, 1].to(dtype=cdtype)) @ F_train.T.conj()
    dK_dls_r = dK_dls.real.to(dtype=rdtype)
    dK_dvar_r = dK_dvar.real.to(dtype=rdtype)

    mean_col = mean.flatten().unsqueeze(-1)
    v = torch.cholesky_solve(mean_col, L, upper=False).squeeze(-1)
    KinvS = torch.cholesky_solve(S, L, upper=False)

    t1_ls = v @ (dK_dls_r @ v)
    t2_ls = torch.sum(KinvS * (K_inv @ dK_dls_r))
    t3_ls = torch.sum(K_inv * dK_dls_r)

    t1_var = v @ (dK_dvar_r @ v)
    t2_var = torch.sum(KinvS * (K_inv @ dK_dvar_r))
    t3_var = torch.sum(K_inv * dK_dvar_r)

    grad_ls = 0.5 * (t1_ls + t2_ls - t3_ls)
    grad_var = 0.5 * (t1_var + t2_var - t3_var)
    return torch.stack([grad_ls, grad_var])


class _BasePolyagammaGPEstimator(BaseEstimator):
    """
    Shared PG-augmented GP estimator implementation.

    Subclasses provide the likelihood-specific PG scalars and response map.
    """

    def __init__(
        self,
        *,
        kernel: str = "squared_exponential",
        lengthscale_init: float = 0.3,
        variance_init: float = 1.0,
        max_iter: int = 50,
        e_step_iters: int = 1,
        final_e_step_iters: int = 1,
        e_step_tol: float = 1e-4,
        rho0: float = 0.7,
        gamma: float = 1e-3,
        lr: float = 0.05,
        n_e_probes: int = 10,
        n_m_probes: int = 10,
        cg_tol: float = 1e-6,
        nufft_eps: float = 1e-7,
        spectral_eps: float = 1e-4,
        trunc_eps: float = 1e-4,
        jitter: float = 1e-8,
        use_exact_weighted_toeplitz_operator: bool = True,
        reuse_e_probes: bool = True,
        prediction_batch_size: int | None = 64,
        predictive_variance_method: str = "exact",
        predictive_variance_probes: int = 16,
        predictive_variance_chebyshev_nodes: int = 7,
        warm_start: bool = False,
        random_state: int | None = None,
        device: str = "auto",
        dtype: str | torch.dtype = "float64",
        verbose: int = 0,
        store_history: bool = False,
    ):
        self.kernel = kernel
        self.lengthscale_init = lengthscale_init
        self.variance_init = variance_init
        self.max_iter = max_iter
        self.e_step_iters = e_step_iters
        self.final_e_step_iters = final_e_step_iters
        self.e_step_tol = e_step_tol
        self.rho0 = rho0
        self.gamma = gamma
        self.lr = lr
        self.n_e_probes = n_e_probes
        self.n_m_probes = n_m_probes
        self.cg_tol = cg_tol
        self.nufft_eps = nufft_eps
        self.spectral_eps = spectral_eps
        self.trunc_eps = trunc_eps
        self.jitter = jitter
        self.use_exact_weighted_toeplitz_operator = use_exact_weighted_toeplitz_operator
        self.reuse_e_probes = reuse_e_probes
        self.prediction_batch_size = prediction_batch_size
        self.predictive_variance_method = predictive_variance_method
        self.predictive_variance_probes = predictive_variance_probes
        self.predictive_variance_chebyshev_nodes = predictive_variance_chebyshev_nodes
        self.warm_start = warm_start
        self.random_state = random_state
        self.device = device
        self.dtype = dtype
        self.verbose = verbose
        self.store_history = store_history

    def _predictive_variance_seed(self) -> int | None:
        if self.random_state is None:
            return None
        return int(self.random_state) + 2_000_000

    def _predictive_variance_method_normalized(self) -> str:
        method = str(self.predictive_variance_method).lower()
        if method not in {"exact", "stochastic", "stochastic_diag_sums", "chebyshev"}:
            raise ValueError(
                "predictive_variance_method must be one of "
                "{'exact', 'stochastic', 'stochastic_diag_sums', 'chebyshev'}."
            )
        return "stochastic" if method == "stochastic_diag_sums" else method

    def _get_predictive_weighted_toeplitz(self) -> ToeplitzND | None:
        if not self.use_exact_weighted_toeplitz_operator:
            return None
        cached = getattr(self, "_predictive_weighted_toeplitz_", None)
        if cached is None:
            cached = _build_weighted_toeplitz(
                self._variational_state_.delta.to(
                    dtype=self._spectral_state_.ws.dtype,
                    device=self._device_,
                ),
                self._spectral_state_,
            )
            self._predictive_weighted_toeplitz_ = cached
        return cached

    def _predictive_variance_off_train(self, X_t: torch.Tensor) -> torch.Tensor:
        method = self._predictive_variance_method_normalized()
        weighted_toeplitz = self._get_predictive_weighted_toeplitz()
        if method == "exact":
            variance = _predictive_variance(
                X_t,
                self._variational_state_.delta,
                self._spectral_state_,
                cg_tol=self.cg_tol,
                nufft_eps=self.nufft_eps,
                use_exact_weighted_toeplitz_operator=self.use_exact_weighted_toeplitz_operator,
                batch_size=self.prediction_batch_size,
                weighted_toeplitz=weighted_toeplitz,
            )
        elif method == "stochastic":
            variance, _ = _predictive_variance_stochastic(
                X_t,
                self._variational_state_.delta,
                self._spectral_state_,
                cg_tol=self.cg_tol,
                nufft_eps=self.nufft_eps,
                n_probes=self.predictive_variance_probes,
                use_exact_weighted_toeplitz_operator=self.use_exact_weighted_toeplitz_operator,
                seed=self._predictive_variance_seed(),
                est_sums=self._get_stochastic_predictive_variance_sums(),
                weighted_toeplitz=weighted_toeplitz,
            )
        else:
            variance, _ = _predictive_variance_chebyshev(
                X_t,
                self._variational_state_.delta,
                self._spectral_state_,
                cg_tol=self.cg_tol,
                nufft_eps=self.nufft_eps,
                n_nodes_per_dim=self.predictive_variance_chebyshev_nodes,
                use_exact_weighted_toeplitz_operator=self.use_exact_weighted_toeplitz_operator,
                batch_size=self.prediction_batch_size,
                weighted_toeplitz=weighted_toeplitz,
            )
        return variance

    def _get_stochastic_predictive_variance_sums(self) -> torch.Tensor:
        method = self._predictive_variance_method_normalized()
        if method != "stochastic":
            raise ValueError("Stochastic predictive variance sums requested while exact mode is active.")
        if self.predictive_variance_probes <= 0:
            raise ValueError("predictive_variance_probes must be positive.")
        if getattr(self, "_stochastic_predictive_variance_sums_", None) is None:
            sums, info = _estimate_stochastic_variance_sums(
                self._variational_state_.delta,
                self._spectral_state_,
                cg_tol=self.cg_tol,
                n_probes=self.predictive_variance_probes,
                use_exact_weighted_toeplitz_operator=self.use_exact_weighted_toeplitz_operator,
                seed=self._predictive_variance_seed(),
                weighted_toeplitz=self._get_predictive_weighted_toeplitz(),
            )
            self._stochastic_predictive_variance_sums_ = sums
            self._stochastic_predictive_variance_info_ = info
        return self._stochastic_predictive_variance_sums_

    def _make_likelihood(self) -> _PGLikelihood:
        raise NotImplementedError

    def _initialize_likelihood_state(self, y_t: torch.Tensor) -> None:
        return None

    def _step_auxiliary_parameters(
        self,
        *,
        targets: torch.Tensor,
        outer: int,
    ) -> dict[str, float]:
        return {}

    def _history_parameter_record(self) -> dict[str, float]:
        return {}

    def _initialize_fit_state(self, X: torch.Tensor, initial_delta: torch.Tensor) -> None:
        self._device_ = _resolve_device(self.device)
        self._rdtype_ = _resolve_dtype(self.dtype)
        self._cdtype_ = torch.complex128 if self._rdtype_ == torch.float64 else torch.complex64

        if not self.warm_start or not hasattr(self, "_variational_state_"):
            self.kernel_ = _make_kernel(
                self.kernel,
                dimension=X.shape[1],
                lengthscale=self.lengthscale_init,
                variance=self.variance_init,
            )
            delta = initial_delta.to(device=self._device_, dtype=self._rdtype_)
            self._variational_state_ = _VariationalState(delta=delta)
        else:
            if self._variational_state_.delta.shape[0] != X.shape[0]:
                delta = initial_delta.to(device=self._device_, dtype=self._rdtype_)
                self._variational_state_ = _VariationalState(delta=delta)

    def fit(self, X, y):
        X_arr, y_arr = check_X_y(X, y, ensure_2d=True, dtype=np.float64)
        likelihood = self._make_likelihood()
        prepared = likelihood.prepare_targets(y_arr)
        y_model_arr = prepared.values
        for key, value in prepared.metadata.items():
            setattr(self, key, value)

        self.n_features_in_ = X_arr.shape[1]
        self._X_train_np_ = X_arr.copy()

        self._device_ = _resolve_device(self.device)
        self._rdtype_ = _resolve_dtype(self.dtype)
        self._cdtype_ = torch.complex128 if self._rdtype_ == torch.float64 else torch.complex64

        X_t = torch.as_tensor(X_arr, device=self._device_, dtype=self._rdtype_)
        y_t = torch.as_tensor(y_model_arr, device=self._device_, dtype=self._rdtype_)
        self._initialize_likelihood_state(y_t)
        likelihood = self._make_likelihood()
        kappa_t = likelihood.kappa(y_t)
        pg_b_t = likelihood.pg_b(y_t)

        self._initialize_fit_state(X_t, 0.25 * pg_b_t)
        self._X_train_t_ = X_t
        self._y_train_t_ = y_t
        self._stochastic_predictive_variance_sums_ = None
        self._stochastic_predictive_variance_info_ = None
        self._predictive_weighted_toeplitz_ = None

        optimizer = Adam(self.kernel_._gp_params_ref.parameters(), lr=self.lr, maximize=True)
        history: list[dict[str, float]] = []

        for outer in range(self.max_iter):
            likelihood = self._make_likelihood()
            kappa_t = likelihood.kappa(y_t)
            pg_b_t = likelihood.pg_b(y_t)
            spectral = _build_spectral_state(
                X_t,
                self.kernel_,
                spectral_eps=self.spectral_eps,
                trunc_eps=self.trunc_eps,
                nufft_eps=self.nufft_eps,
                rdtype=self._rdtype_,
                cdtype=self._cdtype_,
                device=self._device_,
            )
            self._variational_state_, estep_info = _run_estep(
                y_t,
                kappa_t,
                pg_b_t,
                likelihood,
                self._variational_state_,
                spectral,
                max_iters=self.e_step_iters,
                rho0=self.rho0,
                gamma=self.gamma,
                tol=self.e_step_tol,
                n_probes=self.n_e_probes,
                cg_tol=self.cg_tol,
                reuse_probes=self.reuse_e_probes,
                use_exact_weighted_toeplitz_operator=self.use_exact_weighted_toeplitz_operator,
                seed=None if self.random_state is None else self.random_state + 1000 * outer,
                verbose=self.verbose,
            )
            mstep_out = _compute_mstep_gradient(
                kappa_t,
                self._variational_state_.delta,
                spectral,
                n_probes=self.n_m_probes,
                cg_tol=self.cg_tol,
                use_exact_weighted_toeplitz_operator=self.use_exact_weighted_toeplitz_operator,
                seed=None if self.random_state is None else self.random_state + 1000 * outer,
            )
            grad = mstep_out["grad"].real

            raw = self.kernel_._gp_params_ref.raw
            raw.grad = torch.stack(
                [
                    grad[0].to(dtype=raw.dtype, device=raw.device) * self.kernel_.lengthscale,
                    grad[1].to(dtype=raw.dtype, device=raw.device) * self.kernel_.variance,
                    torch.tensor(0.0, dtype=raw.dtype, device=raw.device),
                ]
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            aux_record = self._step_auxiliary_parameters(targets=y_t, outer=outer)

            record = {
                "iter": float(outer),
                "lengthscale": float(self.kernel_.lengthscale),
                "variance": float(self.kernel_.variance),
                "grad_lengthscale": float(grad[0].item()),
                "grad_variance": float(grad[1].item()),
                "e_residual": estep_info["residual"],
                "e_cg_iters": estep_info["cg_iters"],
                "m_cg_iters": float(mstep_out["cg_iters"].item()),
            }
            record.update(aux_record)
            record[likelihood.history_key] = estep_info["metric"]
            history.append(record)
            if self.verbose:
                print(
                    f"outer {outer:3d} lengthscale={record['lengthscale']:.5f} "
                    f"variance={record['variance']:.5f} "
                    f"grad=({record['grad_lengthscale']:+.3e}, {record['grad_variance']:+.3e}) "
                    f"{likelihood.history_label}={record[likelihood.history_key]:.4f}"
                )

        self._spectral_state_ = _build_spectral_state(
            X_t,
            self.kernel_,
            spectral_eps=self.spectral_eps,
            trunc_eps=self.trunc_eps,
            nufft_eps=self.nufft_eps,
            rdtype=self._rdtype_,
            cdtype=self._cdtype_,
            device=self._device_,
        )
        likelihood = self._make_likelihood()
        kappa_t = likelihood.kappa(y_t)
        pg_b_t = likelihood.pg_b(y_t)
        self._variational_state_, final_estep_info = _run_estep(
            y_t,
            kappa_t,
            pg_b_t,
            likelihood,
            self._variational_state_,
            self._spectral_state_,
            max_iters=self.final_e_step_iters,
            rho0=self.rho0,
            gamma=self.gamma,
            tol=self.e_step_tol,
            n_probes=self.n_e_probes,
            cg_tol=self.cg_tol,
            reuse_probes=self.reuse_e_probes,
            use_exact_weighted_toeplitz_operator=self.use_exact_weighted_toeplitz_operator,
            seed=None if self.random_state is None else self.random_state + 999_999,
            verbose=self.verbose,
        )
        beta_mean, beta_cg_iters = _solve_beta_mean(
            kappa_t,
            self._variational_state_.delta,
            self._spectral_state_,
            cg_tol=self.cg_tol,
            use_exact_weighted_toeplitz_operator=self.use_exact_weighted_toeplitz_operator,
        )
        self._beta_mean_ = beta_mean
        self._likelihood_ = likelihood
        self._predictive_weighted_toeplitz_ = self._get_predictive_weighted_toeplitz()

        self.delta_ = self._variational_state_.delta.detach().cpu().numpy()
        self.posterior_mean_ = self._variational_state_.mean.detach().cpu().numpy()
        self.posterior_var_diag_ = self._variational_state_.sigma_diag.detach().cpu().numpy()
        self.lengthscale_ = float(self.kernel_.lengthscale)
        self.variance_ = float(self.kernel_.variance)
        self.n_iter_ = self.max_iter
        self.training_metric_ = final_estep_info["metric"]
        setattr(self, likelihood.training_attr, self.training_metric_)
        self.m_step_gradient_ = mstep_out["grad"].detach().cpu().numpy()
        self.beta_mean_ = beta_mean.detach().cpu().numpy()

        if self.store_history:
            self.history_ = history
        else:
            self.history_ = []
        self.history_.append(
            {
                "iter": float(self.max_iter),
                "lengthscale": self.lengthscale_,
                "variance": self.variance_,
                "grad_lengthscale": float(self.m_step_gradient_[0]),
                "grad_variance": float(self.m_step_gradient_[1]),
                "e_residual": final_estep_info["residual"],
                "e_cg_iters": final_estep_info["cg_iters"],
                "m_cg_iters": float(beta_cg_iters),
            }
        )
        self.history_[-1].update(self._history_parameter_record())
        self.history_[-1][likelihood.history_key] = final_estep_info["metric"]
        return self

    def _is_training_input(self, X_arr: np.ndarray) -> bool:
        return (
            hasattr(self, "_X_train_np_")
            and X_arr.shape == self._X_train_np_.shape
            and np.allclose(X_arr, self._X_train_np_)
        )

    def decision_function(self, X):
        '''
        For training data, return the posterior mean.
        For test data, return the predictive mean.
        '''
        check_is_fitted(self, ["posterior_mean_", "beta_mean_"])
        X_arr = check_array(X, ensure_2d=True, dtype=np.float64)
        if self._is_training_input(X_arr):
            return self.posterior_mean_.copy()

        X_t = torch.as_tensor(X_arr, device=self._device_, dtype=self._rdtype_)
        mean = _predictive_mean(
            X_t,
            self._beta_mean_,
            self._spectral_state_,
            nufft_eps=self.nufft_eps,
        )
        return mean.detach().cpu().numpy()

    def predictive_variance(self, X):
        check_is_fitted(self, ["posterior_var_diag_", "beta_mean_", "delta_"])
        X_arr = check_array(X, ensure_2d=True, dtype=np.float64)
        if self._is_training_input(X_arr):
            return self.posterior_var_diag_.copy()

        X_t = torch.as_tensor(X_arr, device=self._device_, dtype=self._rdtype_)
        variance = self._predictive_variance_off_train(X_t)
        return variance.detach().cpu().numpy()

    def predict_response_mean(self, X):
        check_is_fitted(self, ["posterior_mean_", "posterior_var_diag_", "beta_mean_"])
        X_arr = check_array(X, ensure_2d=True, dtype=np.float64)
        if self._is_training_input(X_arr):
            mean = torch.as_tensor(self.posterior_mean_, dtype=self._rdtype_)
            variance = torch.as_tensor(self.posterior_var_diag_, dtype=self._rdtype_)
        else:
            X_t = torch.as_tensor(X_arr, device=self._device_, dtype=self._rdtype_)
            mean = _predictive_mean(
                X_t,
                self._beta_mean_,
                self._spectral_state_,
                nufft_eps=self.nufft_eps,
            )
            variance = self._predictive_variance_off_train(X_t)

        response = self._likelihood_.response_mean(mean, variance)
        return response.detach().cpu().numpy()


class PolyagammaGPClassifier(_BasePolyagammaGPEstimator, ClassifierMixin):
    """
    Scikit-learn style PG-augmented GP classifier.

    Bernoulli likelihood with logistic link.
    """

    def _make_likelihood(self) -> _PGLikelihood:
        return _PGBernoulliLikelihood()

    def predict_proba(self, X):
        p1 = np.clip(self.predict_response_mean(X), 1e-8, 1.0 - 1e-8)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        labels = (proba >= 0.5).astype(int)
        return self.classes_[labels]


class PolyagammaGPNegativeBinomialRegressor(_BasePolyagammaGPEstimator, RegressorMixin):
    """
    PG-augmented GP regressor for negative binomial counts.

    By default `total_count` is fixed. When `learn_total_count=True`, the estimator
    updates `log(total_count)` during the M-step using the variational marginals
    from the E-step and Gauss-Hermite quadrature.
    """

    def __init__(
        self,
        *,
        total_count: float = 1.0,
        learn_total_count: bool = False,
        total_count_lr: float | None = None,
        total_count_update_frequency: int = 5,
        total_count_quadrature_nodes: int = 12,
        kernel: str = "squared_exponential",
        lengthscale_init: float = 0.3,
        variance_init: float = 1.0,
        max_iter: int = 50,
        e_step_iters: int = 1,
        final_e_step_iters: int = 1,
        e_step_tol: float = 1e-4,
        rho0: float = 0.7,
        gamma: float = 1e-3,
        lr: float = 0.05,
        n_e_probes: int = 10,
        n_m_probes: int = 10,
        cg_tol: float = 1e-6,
        nufft_eps: float = 1e-7,
        spectral_eps: float = 1e-4,
        trunc_eps: float = 1e-4,
        jitter: float = 1e-8,
        use_exact_weighted_toeplitz_operator: bool = True,
        reuse_e_probes: bool = True,
        prediction_batch_size: int | None = 64,
        predictive_variance_method: str = "exact",
        predictive_variance_probes: int = 16,
        predictive_variance_chebyshev_nodes: int = 7,
        warm_start: bool = False,
        random_state: int | None = None,
        device: str = "auto",
        dtype: str | torch.dtype = "float64",
        verbose: int = 0,
        store_history: bool = False,
    ):
        super().__init__(
            kernel=kernel,
            lengthscale_init=lengthscale_init,
            variance_init=variance_init,
            max_iter=max_iter,
            e_step_iters=e_step_iters,
            final_e_step_iters=final_e_step_iters,
            e_step_tol=e_step_tol,
            rho0=rho0,
            gamma=gamma,
            lr=lr,
            n_e_probes=n_e_probes,
            n_m_probes=n_m_probes,
            cg_tol=cg_tol,
            nufft_eps=nufft_eps,
            spectral_eps=spectral_eps,
            trunc_eps=trunc_eps,
            jitter=jitter,
            use_exact_weighted_toeplitz_operator=use_exact_weighted_toeplitz_operator,
            reuse_e_probes=reuse_e_probes,
            prediction_batch_size=prediction_batch_size,
            predictive_variance_method=predictive_variance_method,
            predictive_variance_probes=predictive_variance_probes,
            predictive_variance_chebyshev_nodes=predictive_variance_chebyshev_nodes,
            warm_start=warm_start,
            random_state=random_state,
            device=device,
            dtype=dtype,
            verbose=verbose,
            store_history=store_history,
        )
        self.total_count = total_count
        self.learn_total_count = learn_total_count
        self.total_count_lr = total_count_lr
        self.total_count_update_frequency = total_count_update_frequency
        self.total_count_quadrature_nodes = total_count_quadrature_nodes

    def _current_total_count_tensor(self) -> torch.Tensor:
        if hasattr(self, "_raw_total_count_"):
            return torch.exp(self._raw_total_count_)
        return torch.tensor(float(self.total_count), device=self._device_, dtype=self._rdtype_)

    def _current_total_count(self) -> float:
        return float(self._current_total_count_tensor().detach().cpu().item())

    def _make_likelihood(self) -> _PGLikelihood:
        total_count = self._current_total_count() if hasattr(self, "_device_") else float(self.total_count)
        return _PGNegativeBinomialLikelihood(total_count=total_count)

    def _initialize_likelihood_state(self, y_t: torch.Tensor) -> None:
        if self.total_count <= 0:
            raise ValueError("total_count must be positive.")
        if self.total_count_update_frequency <= 0:
            raise ValueError("total_count_update_frequency must be positive.")
        if self.total_count_quadrature_nodes <= 0:
            raise ValueError("total_count_quadrature_nodes must be positive.")

        if self.learn_total_count and self.warm_start and hasattr(self, "_raw_total_count_"):
            self._raw_total_count_ = torch.nn.Parameter(
                self._raw_total_count_.detach().to(device=self._device_, dtype=self._rdtype_)
            )
        elif self.learn_total_count:
            self._raw_total_count_ = torch.nn.Parameter(
                torch.tensor(math.log(float(self.total_count)), device=self._device_, dtype=self._rdtype_)
            )
        elif hasattr(self, "_raw_total_count_"):
            del self._raw_total_count_

        if self.learn_total_count:
            self._total_count_optimizer_ = Adam(
                [self._raw_total_count_],
                lr=self.lr if self.total_count_lr is None else self.total_count_lr,
                maximize=True,
            )
        elif hasattr(self, "_total_count_optimizer_"):
            del self._total_count_optimizer_

    def _step_auxiliary_parameters(
        self,
        *,
        targets: torch.Tensor,
        outer: int,
    ) -> dict[str, float]:
        current_total_count = self._current_total_count()
        record = {
            "total_count": current_total_count,
            "grad_total_count": 0.0,
            "total_count_updated": 0.0,
        }
        if not self.learn_total_count:
            return record

        grad_total_count = _negative_binomial_total_count_gradient(
            targets,
            self._variational_state_.mean,
            self._variational_state_.sigma_diag,
            total_count=current_total_count,
            quadrature_nodes=self.total_count_quadrature_nodes,
        )
        record["grad_total_count"] = float(grad_total_count.item())

        if (outer + 1) % self.total_count_update_frequency == 0:
            raw = self._raw_total_count_
            raw.grad = (
                grad_total_count.to(dtype=raw.dtype, device=raw.device)
                * self._current_total_count_tensor().to(dtype=raw.dtype, device=raw.device)
            ).detach()
            self._total_count_optimizer_.step()
            self._total_count_optimizer_.zero_grad(set_to_none=True)
            record["total_count"] = self._current_total_count()
            record["total_count_updated"] = 1.0

        return record

    def _history_parameter_record(self) -> dict[str, float]:
        return {
            "total_count": self._current_total_count(),
            "grad_total_count": 0.0,
            "total_count_updated": 0.0,
        }

    def predict_mean_count(self, X):
        return self.predict_response_mean(X)

    def predict(self, X):
        return self.predict_mean_count(X)

    def fit(self, X, y):
        fitted = super().fit(X, y)
        self.total_count_ = self._current_total_count()
        self.shape_parameter_ = self.total_count_
        return fitted
