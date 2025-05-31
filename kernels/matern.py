import math
import torch
from typing import List, Tuple, Optional
from pydantic import Field

from .kernel import Kernel

class Matern(Kernel):
    """
    Matérn kernel.

    k(r) = variance * f_nu(r/lengthscale)
    where f_nu is the Matérn function for nu = 1/2, 3/2, or 5/2.
    The spectral density and gradient use the ω = 2πξ convention, matching SquaredExponential.
    """
    # Define parameter names for this kernel
    hypers: List[str] = Field(default=['lengthscale', 'variance'], frozen=True)
    num_hypers: int = Field(default=3, frozen=True)  # includes noise variance

    # Initial hyperparameter values (used during initialization)
    init_lengthscale: float = Field(default=float('nan'), ge=1e-6)
    init_variance: float = Field(default=float('nan'), ge=1e-6)

    dimension: int = Field(..., ge=1)
    name: Optional[str] = Field(default=None, pattern='^(matern12|matern32|matern52)$')
    nu: Optional[float] = Field(default=None)

    def model_post_init(self, __context):
        nu_values = {
            'matern12': 0.5,
            'matern32': 1.5,
            'matern52': 2.5
        }
        if self.name is not None:
            self.nu = nu_values[self.name]
        super().model_post_init(__context)

    @property
    def lengthscale(self) -> float:
        """Property that uses get_hyper to retrieve the lengthscale."""
        return self.get_hyper('lengthscale')

    @lengthscale.setter
    def lengthscale(self, value: float) -> None:
        """Setter for lengthscale property that uses set_hyper."""
        self.set_hyper('lengthscale', value)

    @property
    def variance(self) -> float:
        """Property that uses get_hyper to retrieve the variance."""
        return self.get_hyper('variance')

    @variance.setter
    def variance(self, value: float) -> None:
        """Setter for variance property that uses set_hyper."""
        self.set_hyper('variance', value)

    def kernel(self, distance: torch.Tensor) -> torch.Tensor:
        """
        Compute the Matérn kernel at the given distances.
        Args:
            distance: Tensor of distances of shape (n,) or (n, m)
        Returns:
            Kernel values of same shape as distance
        """
        lengthscale = self.get_hyper('lengthscale')
        variance = self.get_hyper('variance')
        nu = self.nu
        scaled_dist = torch.abs(distance) / lengthscale
        if nu is not None and abs(nu - 0.5) < 1e-6:
            return variance * torch.exp(-scaled_dist)
        elif nu is not None and abs(nu - 1.5) < 1e-6:
            return variance * (1.0 + math.sqrt(3.0) * scaled_dist) * torch.exp(-math.sqrt(3.0) * scaled_dist)
        elif nu is not None and abs(nu - 2.5) < 1e-6:
            return variance * (1.0 + math.sqrt(5.0) * scaled_dist + 5.0 * (scaled_dist**2) / 3.0) * torch.exp(-math.sqrt(5.0) * scaled_dist)
        else:
            raise ValueError("Unknown or unsupported Matérn kernel type / ν value.")

    def spectral_density(self, xid: torch.Tensor) -> torch.Tensor:
        """
        Compute the spectral density at the frequency xid.
        The spectral density of the Matérn kernel is:
        S(w) = σ² (2π)^{d/2} (2ν)^ν Γ(ν + d/2) / [Γ(ν) ℓ^{2ν}] * (2ν/ℓ² + |w|²)^(-(ν+d/2)),
        where w = 2πξ.
        Args:
            xid: Frequency tensor of shape (n, d) or (n,)
        Returns:
            Spectral density of shape (n,)
        """
        lengthscale = self.get_hyper('lengthscale')
        variance = self.get_hyper('variance')
        nu = self.nu
        d = self.dimension
        if xid.ndim == 1:
            xid = xid.unsqueeze(-1)
        xid_norm_sq = torch.sum(xid**2, dim=-1)
        two_pi_sq = (2.0 * math.pi) ** 2
        w_norm_sq = two_pi_sq * xid_norm_sq
        prefactor = (
            variance
            * (2.0 * math.pi) ** (d / 2)
            * (2.0 * nu) ** nu
            * math.gamma(nu + d / 2)
            / (math.gamma(nu) * (lengthscale ** (2.0 * nu)))
        )
        denom = (2.0 * nu) / (lengthscale**2) + w_norm_sq
        exponent = nu + (d / 2.0)
        return prefactor * (denom ** (-exponent))

    def spectral_grad(self, xid: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient of the spectral density with respect to hyperparameters.
        Args:
            xid: Frequency tensor of shape (n, d) or (n,)
        Returns:
            Gradient tensor of shape (n, 2) - [∂S/∂l, ∂S/∂σ²]
        """
        lengthscale = self.get_hyper('lengthscale')
        variance = self.get_hyper('variance')
        nu = self.nu
        d = self.dimension
        if xid.ndim == 1:
            xid = xid.unsqueeze(-1)
        xid_norm_sq = torch.sum(xid**2, dim=-1)
        two_pi_sq = (2.0 * math.pi) ** 2
        w_norm_sq = two_pi_sq * xid_norm_sq
        S_val = self.spectral_density(xid)
        dS_dvariance = S_val / variance
        denom = (2.0 * nu) / (lengthscale**2) + w_norm_sq
        exponent = nu + (d / 2.0)
        part2 = exponent * (4.0 * nu) / (lengthscale**3) / denom
        dlogS_dl = - (2.0 * nu) / lengthscale + part2
        dS_dlengthscale = S_val * dlogS_dl
        return torch.stack([dS_dlengthscale, dS_dvariance], dim=-1)

    def log_marginal(self, x: torch.Tensor, y: torch.Tensor, sigmasq: float) -> float:
        """
        Compute the log marginal likelihood for the Matérn kernel.
        Args:
            x: Input tensor of shape (n, d) or (n,)
            y: Target tensor of shape (n,)
            sigmasq: Noise variance
        Returns:
            Log marginal likelihood value
        """
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        n = x.shape[0]
        K = self.kernel_matrix(x, x)
        K_noise = K + sigmasq * torch.eye(n, device=K.device)
        try:
            L = torch.linalg.cholesky(K_noise)
            alpha = torch.cholesky_solve(y.unsqueeze(-1), L).squeeze(-1)
            data_fit = 0.5 * torch.sum(y * alpha)
            complexity = torch.sum(torch.log(torch.diag(L)))
            constant = 0.5 * n * math.log(2 * math.pi)
            return -(data_fit + complexity + constant).item()
        except RuntimeError:
            return float('-inf')

    def estimate_hyperparameters(self, x: torch.Tensor, y: torch.Tensor, K: int = 1000) -> Tuple[float, float, float]:
        """
        Estimate reasonable initial hyperparameters based on data characteristics.
        For the Matérn kernel:
        - lengthscale: Use 0.5 * median distance between randomly selected points
        - variance: Set to variance of target values
        - noise_var: Set to a small fraction of target variance
        """
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        n, d = x.shape
        y_var = torch.var(y).item()
        if n > K:
            idx = torch.randperm(n)[:K]
            x_sample = x[idx]
        else:
            x_sample = x
        dists = torch.cdist(x_sample, x_sample)
        mask = dists > 0
        if mask.sum() > 0:
            median_dist = torch.median(dists[mask]).item()
        else:
            median_dist = 1.0
        lengthscale = 0.5 * median_dist
        variance = y_var
        noise_var = 0.01 * y_var
        return lengthscale, variance, noise_var
