import math
from typing import Any, List, Sequence, Tuple, Union

import numpy as np
import torch
from pydantic import Field

from .kernel import Kernel


class ARDSquaredExponential(Kernel):
    """Squared-Exponential kernel with per-dimension lengthscales (ARD).

    .. math::

        k(x, x') = \\sigma_f^2 \\exp\\Big(-\\tfrac{1}{2}
                       \\sum_{i=1}^{d} (x_i - x'_i)^2 / \\ell_i^2\\Big)

    Hyperparameters: ``lengthscale_0``, ``lengthscale_1``, ..., ``lengthscale_{d-1}``,
    ``variance``. (Plus ``noise variance``, registered separately by ``GPParams``.)

    Constructor accepts ``init_lengthscales`` as either a length-``d`` sequence
    or a single scalar that broadcasts to all dims.

    Notes
    -----
    The kernel cannot be expressed as a function of scalar Euclidean distance
    alone (``k`` depends on the *direction* of the displacement). Two methods
    must therefore approximate when called in radial mode:

    - ``kernel(distance)`` returns ``variance * exp(-0.5 * d^2 / ell_max^2)``,
      i.e. the slowest-decaying isotropic upper bound. Used by ``get_xis`` for
      the spatial truncation bound; conservative.
    - ``spectral_density(r)`` for scalar ``r`` returns the most-conservative
      *isotropic* spectral density using ``ell_min`` (widest spectrum).

    For the actual EFGP machinery, ``spectral_density`` and ``spectral_grad``
    are called with vector inputs (shape ``(N, d)``) and use the proper
    anisotropic ARD formulae.
    """

    # Pydantic fields
    init_lengthscales: Tuple[float, ...] = Field(default=(), description=(
        "Per-dim initial lengthscales. May be a length-d sequence or a single"
        " float (broadcast). Validated in model_post_init."
    ))
    init_variance: float = Field(default=float('nan'), ge=1e-6)

    # Filled in model_post_init based on dimension; not frozen so we can
    # populate after pydantic validates the fixed fields.
    hypers: List[str] = Field(default_factory=list)
    num_hypers: int = Field(default=0)

    # Optional convenience: callers can also pass a single scalar.
    def __init__(self, **data):
        # Accept ``init_lengthscales`` as a scalar by broadcasting to a tuple.
        ils = data.get('init_lengthscales')
        d = data.get('dimension')
        if d is not None and ils is not None and not hasattr(ils, '__len__'):
            data['init_lengthscales'] = tuple(float(ils) for _ in range(int(d)))
        elif ils is not None:
            data['init_lengthscales'] = tuple(float(v) for v in ils)
        super().__init__(**data)

    def model_post_init(self, __context: Any) -> None:
        # Validate / broadcast init_lengthscales now that ``dimension`` exists.
        d = self.dimension
        ils = tuple(self.init_lengthscales)
        if len(ils) == 0:
            raise ValueError("ARDSquaredExponential requires init_lengthscales")
        if len(ils) == 1 and d > 1:
            ils = tuple(ils[0] for _ in range(d))
        if len(ils) != d:
            raise ValueError(
                f"init_lengthscales must have length {d} (or be a scalar), "
                f"got length {len(ils)}"
            )

        # Build dynamic hyper names
        hypers_list = [f"lengthscale_{i}" for i in range(d)] + ["variance"]
        # ``num_hypers`` includes noise (per Kernel base-class convention)
        n_hypers = d + 1 + 1

        # Pydantic v2 with default-factory fields: assign through dict to bypass
        # the default-immutability conventions used elsewhere in the codebase.
        self.__dict__['init_lengthscales'] = ils
        self.__dict__['hypers'] = hypers_list
        self.__dict__['num_hypers'] = n_hypers

        # Provide init_lengthscale_<i> + init_variance entries that the base
        # Kernel.model_post_init will look up to populate the GPParams store.
        for i, l_i in enumerate(ils):
            self.__dict__[f"init_lengthscale_{i}"] = float(l_i)

        # Now hand off to the base for GPParams registration.
        super().model_post_init(__context)

    # ------------------------------------------------------------------
    # Per-dimension lengthscale accessors
    # ------------------------------------------------------------------
    @property
    def lengthscales(self) -> torch.Tensor:
        """Return the d per-dimension lengthscales as a 1D tensor."""
        d = self.dimension
        vals = [self.get_hyper(f"lengthscale_{i}") for i in range(d)]
        return torch.tensor(vals, dtype=torch.get_default_dtype())

    @property
    def variance(self) -> float:
        return self.get_hyper('variance')

    @variance.setter
    def variance(self, value: float) -> None:
        self.set_hyper('variance', float(value))

    def set_lengthscales(self, values: Sequence[float]) -> None:
        d = self.dimension
        if len(values) != d:
            raise ValueError(f"Expected {d} lengthscales, got {len(values)}")
        for i, v in enumerate(values):
            self.set_hyper(f"lengthscale_{i}", float(v))

    # ------------------------------------------------------------------
    # Conservative scalar/radial fallbacks
    # ------------------------------------------------------------------
    def kernel(self, distance: torch.Tensor) -> torch.Tensor:
        """Conservative scalar-distance evaluation.

        Returns the slowest-decaying isotropic upper bound,
        ``variance * exp(-0.5 * r^2 / ell_max^2)``. This is what ``get_xis``
        uses to find the spatial truncation bound; the result is an over-
        estimate of ``Ltime`` for the ARD kernel, which is safe.
        """
        ls = self.lengthscales
        l_max = float(ls.max())
        variance = float(self.get_hyper('variance'))
        return variance * torch.exp(-0.5 * (distance / l_max) ** 2)

    def _spectral_density_radial(self, r: torch.Tensor) -> torch.Tensor:
        """Conservative isotropic spectral density at scalar radii ``r``.

        Uses ``ell_min`` for the widest spectral support (most-conservative
        estimate of the frequency truncation bound).
        """
        ls = self.lengthscales
        l_min = float(ls.min())
        variance = float(self.get_hyper('variance'))
        # Isotropic-radial form with proper d-dim prefactor
        d = self.dimension
        prefactor = ((2 * np.pi) * l_min ** 2) ** (d / 2) * variance
        two_pi_sq = (2 * np.pi) ** 2
        r_sq = r * r
        return prefactor * torch.exp(-two_pi_sq * l_min ** 2 * r_sq / 2)

    # ------------------------------------------------------------------
    # Proper anisotropic spectral density / gradient (vector inputs)
    # ------------------------------------------------------------------
    def spectral_density(self, xid: torch.Tensor) -> torch.Tensor:
        """Spectral density at frequencies ``xid``.

        - Vector input ``(N, d)``: proper ARD form.
        - Scalar/1D input: isotropic-radial fallback using ``ell_min``.
        """
        d = self.dimension
        if xid.ndim >= 2 and xid.shape[-1] == d:
            ls = self.lengthscales.to(device=xid.device, dtype=xid.dtype)
            variance = float(self.get_hyper('variance'))
            two_pi_sq = (2 * np.pi) ** 2
            # Anisotropic exponent: sum_i (ell_i * xi_i)^2
            xi_sq = (xid * ls.unsqueeze(0)) ** 2
            exponent = -two_pi_sq * xi_sq.sum(dim=-1) / 2
            prefactor = (2 * np.pi) ** (d / 2) * float(ls.prod()) * variance
            return prefactor * torch.exp(exponent)
        # Radial / scalar input → conservative isotropic
        return self._spectral_density_radial(xid)

    def spectral_grad(self, xid: torch.Tensor) -> torch.Tensor:
        """Gradient of spectral density w.r.t. (lengthscale_0, ..., variance).

        Returns shape ``(N, d + 1)``. Only vector inputs are supported here
        (this method is only called from the EFGP gradient path).
        """
        d = self.dimension
        if xid.ndim < 2 or xid.shape[-1] != d:
            raise ValueError(
                f"spectral_grad expects vector input of shape (N, {d}); got {tuple(xid.shape)}"
            )
        ls = self.lengthscales.to(device=xid.device, dtype=xid.dtype)
        variance = float(self.get_hyper('variance'))
        two_pi_sq = (2 * np.pi) ** 2
        xi_sq = xid ** 2  # (N, d)
        exponent = -two_pi_sq * (xi_sq * ls.unsqueeze(0) ** 2).sum(dim=-1) / 2
        prefactor = (2 * np.pi) ** (d / 2) * float(ls.prod()) * variance
        s_w = prefactor * torch.exp(exponent)  # (N,)

        # ∂S/∂ell_i = S * (1/ell_i - (2π)^2 * ell_i * xi_i^2)
        inv_ls = 1.0 / ls  # (d,)
        d_ls = s_w.unsqueeze(-1) * (
            inv_ls.unsqueeze(0) - two_pi_sq * ls.unsqueeze(0) * xi_sq
        )  # (N, d)

        # ∂S/∂σ_f^2 = S / variance
        d_var = (s_w / variance).unsqueeze(-1)  # (N, 1)

        return torch.cat([d_ls, d_var], dim=-1)  # (N, d + 1)

    # ------------------------------------------------------------------
    # Proper anisotropic kernel matrix (for any external caller)
    # ------------------------------------------------------------------
    def kernel_matrix(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """K(x, y) for ARD inputs, computed properly (no scalar-distance hack)."""
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        ls = self.lengthscales.to(device=x.device, dtype=x.dtype)
        variance = float(self.get_hyper('variance'))
        # Standardise by per-dim lengthscale, then use Euclidean cdist on the
        # whitened coordinates: equivalent to the anisotropic exponent.
        x_std = x / ls.unsqueeze(0)
        y_std = y / ls.unsqueeze(0)
        d2 = torch.cdist(x_std, y_std) ** 2
        return variance * torch.exp(-0.5 * d2)

    # ------------------------------------------------------------------
    # Hyperparameter estimation: lift SE estimator and broadcast
    # ------------------------------------------------------------------
    def estimate_hyperparameters(
        self, x: torch.Tensor, y: torch.Tensor, K: int = 1000
    ) -> Tuple[List[float], float, float]:
        """Same heuristic as SquaredExponential; returns the same lengthscale
        for every dim. User can override via ``init_lengthscales``.
        """
        from .squared_exponential import SquaredExponential
        proxy = SquaredExponential(
            dimension=self.dimension,
            init_lengthscale=1.0,
            init_variance=1.0,
        )
        l, v, n = proxy.estimate_hyperparameters(x, y, K=K)
        return [float(l)] * self.dimension, float(v), float(n)
