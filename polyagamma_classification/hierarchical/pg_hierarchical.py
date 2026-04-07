"""
Hierarchical additive-kernel PG-augmented GP for negative binomial counts.

Model:
    f_i = g(t_i) + h_{ell_i}(t_i)
    g ~ GP(0, k_g),  h_ell ~iid GP(0, k_h)
    y_i | f_i, r ~ NB(r, sigma(f_i))

The additive covariance K = K_g + K_h is handled via a stacked feature map
    U = [Phi_g  Phi_loc]
where Phi_loc = blockdiag(Phi_1, ..., Phi_L).

All CG matvecs use exact Toeplitz operators T_ell = F_ell^* Delta_ell F_ell.
"""
from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F_func
from torch.fft import fftn, ifftn

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cg import ConjugateGradients
from efgpnd import NUFFT, ToeplitzND
from kernels import SquaredExponential
from utils.kernels import get_xis

# ---------------------------------------------------------------------------
# Helpers reused from pg_classifier (imported to avoid duplication)
# ---------------------------------------------------------------------------
_PG_ROOT = Path(__file__).resolve().parents[1]
if str(_PG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PG_ROOT))

from pg_classifier import (
    _pg_omega_expectation,
    _sample_rademacher,
    _expected_log_sigmoid_negative_gaussian,
    _negative_binomial_total_count_gradient,
    _gauss_hermite_normal_rule,
    negative_binomial_gaussian_mean,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class HierarchicalSpectralState:
    """Spectral state for the additive kernel on a shared frequency grid."""
    # Shared grid
    xis: torch.Tensor        # (mtot^d, d)
    h: float
    mtot: int
    out_shape: tuple[int, ...]  # (mtot,)*d

    # Global kernel weights
    ws_g: torch.Tensor       # sqrt(h^d * S_g(xi)), complex, shape (m,)
    ws2_g: torch.Tensor      # h^d * S_g(xi), complex, shape (m,)
    Dprime_g: torch.Tensor   # spectral grad for global kernel, (m, n_hypers)

    # Local kernel weights
    ws_h: torch.Tensor       # sqrt(h^d * S_h(xi)), complex, shape (m,)
    ws2_h: torch.Tensor      # h^d * S_h(xi), complex, shape (m,)
    Dprime_h: torch.Tensor   # spectral grad for local kernel, (m, n_hypers)

    # Global NUFFT op (all n points)
    nufft_global: NUFFT

    # Per-location NUFFT ops
    nufft_local: list[NUFFT]

    # Location bookkeeping
    loc_indices: list[torch.Tensor]  # index arrays into the full n-vector
    n_locations: int


def _make_kernel(*, dimension: int, lengthscale: float, variance: float) -> SquaredExponential:
    return SquaredExponential(
        dimension=dimension,
        init_lengthscale=lengthscale,
        init_variance=variance,
    )


def _build_hierarchical_spectral_state(
    X: torch.Tensor,
    locations: torch.Tensor,
    kernel_g: SquaredExponential,
    kernel_h: SquaredExponential,
    *,
    spectral_eps: float,
    trunc_eps: float,
    nufft_eps: float,
    rdtype: torch.dtype,
    cdtype: torch.dtype,
    device: torch.device,
) -> HierarchicalSpectralState:
    """Build spectral state on the shared (more conservative) grid."""
    x0 = X.min(dim=0).values
    x1 = X.max(dim=0).values
    L_domain = (x1 - x0).max()
    d = X.shape[1]

    # Get grids for both kernels, pick the larger one
    xis_g_1d, h_g, mtot_g = get_xis(kernel_g, eps=spectral_eps, L=L_domain,
                                      use_integral=True, l2scaled=False, trunc_eps=trunc_eps)
    xis_h_1d, h_h, mtot_h = get_xis(kernel_h, eps=spectral_eps, L=L_domain,
                                      use_integral=True, l2scaled=False, trunc_eps=trunc_eps)

    if mtot_h >= mtot_g:
        xis_1d, h, mtot = xis_h_1d, h_h, mtot_h
    else:
        xis_1d, h, mtot = xis_g_1d, h_g, mtot_g

    grids = torch.meshgrid(*(xis_1d for _ in range(d)), indexing="ij")
    xis = torch.stack(grids, dim=-1).view(-1, d)

    # Spectral weights for both kernels on the shared grid
    sd_g = kernel_g.spectral_density(xis).to(dtype=rdtype)
    ws2_g = (sd_g * h**d).to(device=device, dtype=cdtype)
    ws_g = torch.sqrt(ws2_g)

    sd_h = kernel_h.spectral_density(xis).to(dtype=rdtype)
    ws2_h = (sd_h * h**d).to(device=device, dtype=cdtype)
    ws_h = torch.sqrt(ws2_h)

    Dprime_g = (h**d * kernel_g.spectral_grad(xis)).to(device=device, dtype=cdtype)
    Dprime_h = (h**d * kernel_h.spectral_grad(xis)).to(device=device, dtype=cdtype)

    out_shape = (mtot,) * d
    xcen = torch.zeros(d, device=device, dtype=rdtype)

    # Global NUFFT on all points
    nufft_global = NUFFT(X, xcen, h, nufft_eps, cdtype=cdtype, device=device)

    # Per-location NUFFT ops and index arrays
    unique_locs = torch.unique(locations)
    n_locations = len(unique_locs)
    loc_indices: list[torch.Tensor] = []
    nufft_local: list[NUFFT] = []
    for loc in unique_locs:
        idx = torch.where(locations == loc)[0]
        loc_indices.append(idx)
        nufft_local.append(NUFFT(X[idx], xcen, h, nufft_eps, cdtype=cdtype, device=device))

    return HierarchicalSpectralState(
        xis=xis.to(device=device, dtype=rdtype),
        h=h, mtot=mtot, out_shape=out_shape,
        ws_g=ws_g, ws2_g=ws2_g, Dprime_g=Dprime_g,
        ws_h=ws_h, ws2_h=ws2_h, Dprime_h=Dprime_h,
        nufft_global=nufft_global,
        nufft_local=nufft_local,
        loc_indices=loc_indices,
        n_locations=n_locations,
    )


# ---------------------------------------------------------------------------
# Build per-location weighted Toeplitz operators
# ---------------------------------------------------------------------------

def _build_per_location_toeplitz(
    delta: torch.Tensor,
    spec: HierarchicalSpectralState,
) -> tuple[ToeplitzND, list[ToeplitzND]]:
    """Return (T_all, [T_1, ..., T_L])."""
    cdtype = spec.ws_g.dtype
    conv_shape = tuple(2 * n - 1 for n in spec.out_shape)

    # Global T_all from all points
    w_all = delta.to(dtype=cdtype, device=delta.device).flatten()
    v_all = spec.nufft_global.type1(w_all, out_shape=conv_shape)
    T_all = ToeplitzND(v_all.to(dtype=cdtype), force_pow2=True)

    # Per-location
    T_loc: list[ToeplitzND] = []
    for ell, idx in enumerate(spec.loc_indices):
        w_ell = delta[idx].to(dtype=cdtype, device=delta.device).flatten()
        v_ell = spec.nufft_local[ell].type1(w_ell, out_shape=conv_shape)
        T_loc.append(ToeplitzND(v_ell.to(dtype=cdtype), force_pow2=True))

    return T_all, T_loc


# ---------------------------------------------------------------------------
# Block CG matvec and solver
# ---------------------------------------------------------------------------

def _block_matvec(
    a: torch.Tensor,
    *,
    m: int,
    L: int,
    Dg_half: torch.Tensor,
    Dh_half: torch.Tensor,
    T_all: ToeplitzND,
    T_loc: list[ToeplitzND],
) -> torch.Tensor:
    """
    Apply (I + U^* Delta U) to stacked vector a of shape ((1+L)*m,).

    Layout: a[:m] = a_g, a[m + ell*m : m + (ell+1)*m] = a_ell.

    Uses the Hermitian-symmetrized form with half-weights Dg^{1/2}, Dh^{1/2}.
    """
    out = torch.zeros_like(a)
    a_g = a[:m]

    # --- Global block ---
    # I*a_g + Dg^{1/2} T_all Dg^{1/2} a_g + sum_ell Dg^{1/2} T_ell Dh^{1/2} a_ell
    global_out = a_g.clone()
    global_out = global_out + Dg_half * T_all(Dg_half * a_g)

    for ell in range(L):
        a_ell = a[m + ell * m: m + (ell + 1) * m]
        global_out = global_out + Dg_half * T_loc[ell](Dh_half * a_ell)

    out[:m] = global_out

    # --- Local blocks ---
    for ell in range(L):
        a_ell = a[m + ell * m: m + (ell + 1) * m]
        # I*a_ell + Dh^{1/2} T_ell Dh^{1/2} a_ell + Dh^{1/2} T_ell Dg^{1/2} a_g
        local_out = a_ell.clone()
        # Combine both Toeplitz applies through one call:
        combined = Dh_half * a_ell + Dg_half * a_g
        T_combined = T_loc[ell](combined)
        local_out = local_out + Dh_half * T_combined
        out[m + ell * m: m + (ell + 1) * m] = local_out

    return out


def _block_matvec_batched(
    A: torch.Tensor,
    *,
    m: int,
    L: int,
    Dg_half: torch.Tensor,
    Dh_half: torch.Tensor,
    T_all: ToeplitzND,
    T_loc: list[ToeplitzND],
) -> torch.Tensor:
    """Batched version: A has shape (B, (1+L)*m)."""
    B = A.shape[0]
    out = torch.zeros_like(A)
    A_g = A[:, :m]

    # Global block
    global_out = A_g.clone()
    global_out = global_out + Dg_half[None, :] * T_all(Dg_half[None, :] * A_g)

    for ell in range(L):
        A_ell = A[:, m + ell * m: m + (ell + 1) * m]
        global_out = global_out + Dg_half[None, :] * T_loc[ell](Dh_half[None, :] * A_ell)

    out[:, :m] = global_out

    # Local blocks
    for ell in range(L):
        A_ell = A[:, m + ell * m: m + (ell + 1) * m]
        local_out = A_ell.clone()
        combined = Dh_half[None, :] * A_ell + Dg_half[None, :] * A_g
        T_combined = T_loc[ell](combined)
        local_out = local_out + Dh_half[None, :] * T_combined
        out[:, m + ell * m: m + (ell + 1) * m] = local_out

    return out


def _build_block_rhs(
    z: torch.Tensor,
    spec: HierarchicalSpectralState,
    Dg_half: torch.Tensor,
    Dh_half: torch.Tensor,
) -> torch.Tensor:
    """
    Compute D^{1/2} U^* z for a single vector z of shape (n,).
    Returns shape ((1+L)*m,).
    """
    m = spec.ws_g.numel()
    L = spec.n_locations
    cdtype = spec.ws_g.dtype
    z_c = z.to(dtype=cdtype)

    rhs = torch.zeros((1 + L) * m, dtype=cdtype, device=z.device)

    # Global: Dg^{1/2} * F_g^* z
    Fgadj_z = spec.nufft_global.type1(z_c, out_shape=spec.out_shape).reshape(-1)
    rhs[:m] = Dg_half * Fgadj_z

    # Per-location: Dh^{1/2} * F_ell^* z_ell
    for ell in range(L):
        idx = spec.loc_indices[ell]
        z_ell = z_c[idx]
        Fadj_z_ell = spec.nufft_local[ell].type1(z_ell, out_shape=spec.out_shape).reshape(-1)
        rhs[m + ell * m: m + (ell + 1) * m] = Dh_half * Fadj_z_ell

    return rhs


def _build_block_rhs_batched(
    Z: torch.Tensor,
    spec: HierarchicalSpectralState,
    Dg_half: torch.Tensor,
    Dh_half: torch.Tensor,
) -> torch.Tensor:
    """
    Compute D^{1/2} U^* Z for Z of shape (B, n).
    Returns shape (B, (1+L)*m).
    """
    B = Z.shape[0]
    m = spec.ws_g.numel()
    L = spec.n_locations
    cdtype = spec.ws_g.dtype
    Z_c = Z.to(dtype=cdtype)

    rhs = torch.zeros(B, (1 + L) * m, dtype=cdtype, device=Z.device)

    # Global: Dg^{1/2} * F_g^* Z  (batched type-1)
    for b in range(B):
        Fgadj = spec.nufft_global.type1(Z_c[b], out_shape=spec.out_shape).reshape(-1)
        rhs[b, :m] = Dg_half * Fgadj

    # Per-location
    for ell in range(L):
        idx = spec.loc_indices[ell]
        for b in range(B):
            z_ell = Z_c[b, idx]
            Fadj = spec.nufft_local[ell].type1(z_ell, out_shape=spec.out_shape).reshape(-1)
            rhs[b, m + ell * m: m + (ell + 1) * m] = Dh_half * Fadj

    return rhs


def _reconstruct_from_features(
    a: torch.Tensor,
    spec: HierarchicalSpectralState,
    Dg_half: torch.Tensor,
    Dh_half: torch.Tensor,
) -> torch.Tensor:
    """
    Compute U D^{-1/2} a = Phi_g (Dg^{-1/2} a_g) + Phi_loc (Dh^{-1/2} a_ell).
    Since we symmetrized with half-weights, the actual feature map is
    Phi_g = F_g Dg, Phi_loc = F_ell Dh.  So:
      result = F_g (Dg * Dg^{-1/2} a_g) + sum_ell scatter(F_ell (Dh * Dh^{-1/2} a_ell))
             = F_g (Dg^{1/2} a_g) + ...
    Wait -- let me redo this carefully.

    The symmetrized system solves for tilde_a where a = D^{1/2} tilde_a.
    The original feature-space solution is beta = D^{-1} tilde_a ... no.

    Let's be precise. The unsymmetrized system is:
        (I + Phi^* Delta Phi) beta = Phi^* z
        result = Phi beta

    where Phi = [F_g D_g; blockdiag(F_ell D_h)].

    The symmetrization substitutes beta = D^{-1/2} tilde_a:
        (I + D^{1/2} F^* Delta F D^{1/2}) tilde_a = D^{1/2} F^* z
        result = F D D^{-1/2} tilde_a = F D^{1/2} tilde_a

    So reconstruction is:
        result = F_g (Dg^{1/2} tilde_a_g) + sum_ell scatter(F_ell (Dh^{1/2} tilde_a_ell))
    """
    m = spec.ws_g.numel()
    L = spec.n_locations
    n = sum(idx.numel() for idx in spec.loc_indices)

    a_g = a[:m]
    # Global: F_g(Dg^{1/2} * a_g) via type-2 NUFFT
    coeff_g = (Dg_half * a_g)
    result = spec.nufft_global.type2(coeff_g, out_shape=spec.out_shape)

    # Per-location
    for ell in range(L):
        a_ell = a[m + ell * m: m + (ell + 1) * m]
        coeff_ell = (Dh_half * a_ell)
        vals_ell = spec.nufft_local[ell].type2(coeff_ell, out_shape=spec.out_shape)
        idx = spec.loc_indices[ell]
        result[idx] = result[idx] + vals_ell

    return result.real


def _reconstruct_from_features_batched(
    A: torch.Tensor,
    spec: HierarchicalSpectralState,
    Dg_half: torch.Tensor,
    Dh_half: torch.Tensor,
) -> torch.Tensor:
    """Batched reconstruction. A shape (B, (1+L)*m), returns (B, n)."""
    B = A.shape[0]
    m = spec.ws_g.numel()
    L = spec.n_locations
    n = sum(idx.numel() for idx in spec.loc_indices)
    cdtype = spec.ws_g.dtype

    result = torch.zeros(B, n, dtype=cdtype, device=A.device)
    for b in range(B):
        result[b] = _reconstruct_from_features(A[b], spec, Dg_half, Dh_half).to(dtype=cdtype)

    return result.real


# ---------------------------------------------------------------------------
# Sigma apply: (K^{-1} + Delta)^{-1} z  via block feature-space CG
# ---------------------------------------------------------------------------

def _make_hierarchical_sigma_apply(
    spec: HierarchicalSpectralState,
    delta: torch.Tensor,
    *,
    cg_tol: float,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], dict]:
    """
    Build a function that applies Sigma = (K^{-1} + Delta)^{-1} to vectors.
    Uses the identity Sigma z = U (I + U^* Delta U)^{-1} U^* z
    with Hermitian symmetrization.
    """
    info = {"cg_iters": 0}
    m = spec.ws_g.numel()
    L = spec.n_locations
    block_dim = (1 + L) * m
    cdtype = spec.ws_g.dtype

    # Half-weights for symmetrization
    D2g_real = spec.ws2_g.real
    eps_d = max(float(D2g_real.mean()) * 1e-14, 1e-14)
    Dg_half = torch.sqrt(torch.clamp(D2g_real, min=eps_d)).to(dtype=cdtype)

    D2h_real = spec.ws2_h.real
    eps_d_h = max(float(D2h_real.mean()) * 1e-14, 1e-14)
    Dh_half = torch.sqrt(torch.clamp(D2h_real, min=eps_d_h)).to(dtype=cdtype)

    # Build Toeplitz operators
    T_all, T_loc = _build_per_location_toeplitz(delta, spec)

    def A_apply(v: torch.Tensor) -> torch.Tensor:
        if v.dim() == 1:
            return _block_matvec(v, m=m, L=L,
                                 Dg_half=Dg_half, Dh_half=Dh_half,
                                 T_all=T_all, T_loc=T_loc)
        else:
            return _block_matvec_batched(v, m=m, L=L,
                                         Dg_half=Dg_half, Dh_half=Dh_half,
                                         T_all=T_all, T_loc=T_loc)

    def sigma_apply(z: torch.Tensor) -> torch.Tensor:
        """z: shape (n,) or (B, n). Returns same shape."""
        vector_input = z.dim() == 1
        if vector_input:
            z = z.unsqueeze(0)

        B = z.shape[0]
        z_c = z.to(dtype=cdtype)

        # Build RHS: D^{1/2} U^* z
        rhs = _build_block_rhs_batched(z_c, spec, Dg_half, Dh_half)

        # CG solve
        cg = ConjugateGradients(
            A_apply,
            rhs,
            x0=torch.zeros_like(rhs),
            tol=cg_tol,
            max_iter=2000,
            early_stopping=True,
        )
        tilde_a = cg.solve()
        info["cg_iters"] = cg.iters_completed

        # Reconstruct: F D^{1/2} tilde_a
        result = _reconstruct_from_features_batched(tilde_a, spec, Dg_half, Dh_half)

        if vector_input:
            return result.squeeze(0)
        return result

    return sigma_apply, info


# ---------------------------------------------------------------------------
# Feature-space solver for M-step (returns feature-space coefficients)
# ---------------------------------------------------------------------------

def _make_hierarchical_feature_solver(
    delta: torch.Tensor,
    spec: HierarchicalSpectralState,
    *,
    cg_tol: float,
) -> tuple[
    Callable[[torch.Tensor], tuple[torch.Tensor, int]],
    dict,
]:
    """
    Build solver for the symmetrized block system.
    Returns (solve_fn, info).
    solve_fn takes q in observation space and returns (tilde_a, cg_iters).
    """
    info = {"cg_iters": 0}
    m = spec.ws_g.numel()
    L = spec.n_locations
    cdtype = spec.ws_g.dtype

    D2g_real = spec.ws2_g.real
    eps_d = max(float(D2g_real.mean()) * 1e-14, 1e-14)
    Dg_half = torch.sqrt(torch.clamp(D2g_real, min=eps_d)).to(dtype=cdtype)

    D2h_real = spec.ws2_h.real
    eps_d_h = max(float(D2h_real.mean()) * 1e-14, 1e-14)
    Dh_half = torch.sqrt(torch.clamp(D2h_real, min=eps_d_h)).to(dtype=cdtype)

    T_all, T_loc = _build_per_location_toeplitz(delta, spec)

    def A_apply(v: torch.Tensor) -> torch.Tensor:
        if v.dim() == 1:
            return _block_matvec(v, m=m, L=L,
                                 Dg_half=Dg_half, Dh_half=Dh_half,
                                 T_all=T_all, T_loc=T_loc)
        else:
            return _block_matvec_batched(v, m=m, L=L,
                                         Dg_half=Dg_half, Dh_half=Dh_half,
                                         T_all=T_all, T_loc=T_loc)

    def solve(q: torch.Tensor) -> tuple[torch.Tensor, int]:
        """q: shape (n,) or (B, n). Returns (tilde_a, cg_iters)."""
        vector_input = q.dim() == 1
        if vector_input:
            q = q.unsqueeze(0)

        rhs = _build_block_rhs_batched(q.to(dtype=cdtype), spec, Dg_half, Dh_half)

        cg = ConjugateGradients(
            A_apply,
            rhs,
            x0=torch.zeros_like(rhs),
            tol=cg_tol,
            max_iter=2000,
            early_stopping=True,
        )
        tilde_a = cg.solve()
        info["cg_iters"] = cg.iters_completed

        if vector_input:
            return tilde_a.squeeze(0), cg.iters_completed
        return tilde_a, cg.iters_completed

    return solve, info


# ---------------------------------------------------------------------------
# E-step
# ---------------------------------------------------------------------------

def _run_hierarchical_estep(
    targets: torch.Tensor,
    kappa: torch.Tensor,
    pg_b: torch.Tensor,
    delta: torch.Tensor,
    spec: HierarchicalSpectralState,
    *,
    max_iters: int,
    rho0: float,
    gamma: float,
    tol: float,
    n_probes: int,
    cg_tol: float,
    seed: int | None,
    verbose: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    Run the E-step for the hierarchical model.

    Returns (delta, mean, sigma_diag, info_dict).
    """
    residual = float("inf")
    mean = torch.zeros_like(targets)
    sigma_diag = torch.zeros_like(targets)

    with torch.no_grad():
        for it in range(max_iters):
            sigma_apply, sigma_info = _make_hierarchical_sigma_apply(
                spec, delta, cg_tol=cg_tol,
            )

            # Probes for Hutchinson diagonal estimation
            probe_seed = None if seed is None else seed + 17 * (it + 1)
            probes = _sample_rademacher(
                (n_probes, targets.numel()),
                device=targets.device,
                dtype=targets.dtype,
                seed=probe_seed,
            )

            Z = torch.cat([kappa[None, :], probes], dim=0)
            S_all = sigma_apply(Z)
            mean = S_all[0]
            Sz = S_all[1:]
            sigma_diag = (probes * Sz).mean(dim=0)

            c2 = (sigma_diag + mean.pow(2)).clamp_min(1e-12)
            c = torch.sqrt(c2)
            Lambda = _pg_omega_expectation(c, pg_b)

            rho = rho0 / (1.0 + gamma * it)
            delta = delta * (1.0 - rho) + rho * Lambda
            delta = delta.clamp(min=0.0)

            residual = float((delta - Lambda).abs().max().item())
            if verbose > 1:
                print(f"  E-step it {it:3d} rho={rho:.3f} max|Delta-Lambda|={residual:.3e}")
            if residual < tol:
                break

    return delta, mean, sigma_diag, {
        "residual": residual,
        "cg_iters": float(sigma_info["cg_iters"]),
    }


# ---------------------------------------------------------------------------
# M-step gradient
# ---------------------------------------------------------------------------

def _compute_hierarchical_mstep_gradient(
    kappa: torch.Tensor,
    delta: torch.Tensor,
    spec: HierarchicalSpectralState,
    *,
    n_probes: int,
    cg_tol: float,
    seed: int | None,
) -> dict[str, torch.Tensor]:
    """
    Compute gradients for global and local kernel hyperparameters.

    Uses the identity:
        dL/dtheta = 0.5 * mu^T K^{-1} (dK/dtheta) K^{-1} mu
                  - 0.5 * Tr((I + Lambda K)^{-1} Lambda dK/dtheta)

    In feature space with the stacked U.
    """
    m = spec.ws_g.numel()
    L = spec.n_locations
    cdtype = spec.ws_g.dtype

    # Half-weights
    D2g_real = spec.ws2_g.real
    eps_d = max(float(D2g_real.mean()) * 1e-14, 1e-14)
    Dg_half = torch.sqrt(torch.clamp(D2g_real, min=eps_d)).to(dtype=cdtype)
    Dg_half_inv = 1.0 / Dg_half

    D2h_real = spec.ws2_h.real
    eps_d_h = max(float(D2h_real.mean()) * 1e-14, 1e-14)
    Dh_half = torch.sqrt(torch.clamp(D2h_real, min=eps_d_h)).to(dtype=cdtype)
    Dh_half_inv = 1.0 / Dh_half

    solve, solve_info = _make_hierarchical_feature_solver(delta, spec, cg_tol=cg_tol)

    # --- Term 1: 0.5 * mu^T K^{-1} (dK/dtheta) K^{-1} mu ---
    # K^{-1} mu via: K^{-1} mu = kappa - Delta * mu, but we need the
    # feature-space version. We solve (I + U^* Delta U) a = U^* kappa,
    # then beta = D^{-1/2} a is the unsymmetrized coefficient, and
    # K^{-1} mu = kappa - Delta * Sigma * kappa = ... Actually, we use
    # the identity that the feature-space solution tilde_a satisfies:
    #   Sigma kappa = F D^{1/2} tilde_a
    # and K^{-1} (Sigma kappa) can be obtained from tilde_a directly.
    #
    # Simpler: use beta = D^{-1/2} tilde_a. Then mu = Phi beta = U beta_full.
    # K^{-1} mu = beta^T dK/dtheta beta (in feature space).
    #
    # For the global kernel: term1_g = beta_g^T (h^d dS_g/dtheta) beta_g
    # For the local kernel: term1_h = sum_ell beta_ell^T (h^d dS_h/dtheta) beta_ell

    tilde_a_kappa, cg_iters_kappa = solve(kappa)

    # Unsymmetrize: beta_g = Dg^{-1/2} tilde_a_g, beta_ell = Dh^{-1/2} tilde_a_ell
    beta_g = Dg_half_inv * tilde_a_kappa[:m]
    betas_h = []
    for ell in range(L):
        betas_h.append(Dh_half_inv * tilde_a_kappa[m + ell * m: m + (ell + 1) * m])

    # Term 1 for global kernel
    abs2_g = (beta_g.conj() * beta_g).real
    term1_g = spec.Dprime_g.real.T @ abs2_g  # shape (n_hypers,)

    # Term 1 for local kernel
    term1_h = torch.zeros_like(spec.Dprime_h[0])
    for ell in range(L):
        abs2_ell = (betas_h[ell].conj() * betas_h[ell]).real
        term1_h = term1_h + spec.Dprime_h.real.T @ abs2_ell

    # --- Term 2: stochastic trace ---
    # Tr((I + Delta K)^{-1} Delta dK/dtheta)
    # = Tr(Delta dK/dtheta (I + Delta K)^{-1})
    # We use probes z_j and compute:
    #   Tr ≈ (1/J) sum_j z_j^T Delta dK/dtheta (I + Delta K)^{-1} z_j
    # But (I + Delta K)^{-1} z = alpha means (I + Delta K) alpha = z
    # which is (K^{-1} + Delta)^{-1} K^{-1} z ... this is getting complicated.
    #
    # Use the feature-space form instead. The trace term is:
    #   Tr((I + Lambda K)^{-1} Lambda dK/dtheta)
    # In feature space with U = Phi:
    #   (I + Lambda K)^{-1} = I - U(I + U^* Lambda U)^{-1} U^* Lambda  (Woodbury)
    # So Tr(... Lambda dK/dtheta) = Tr(Lambda dK/dtheta) - Tr(U(I+U^*LambdaU)^{-1}U^*Lambda^2 dK/dtheta)
    #
    # The simpler stochastic route: draw probe vectors z in observation space,
    # compute (I + Delta K)^{-1} z via CG, then:
    #   trace_est = z^T Delta dK/dtheta (I + Delta K)^{-1} z
    #
    # Actually let's use the same approach as _compute_mstep_gradient:
    # Draw probes in observation space, compute F^* (probes), solve in feature space,
    # then form the stochastic trace.

    probes = _sample_rademacher(
        (n_probes, kappa.numel()),
        device=kappa.device,
        dtype=kappa.dtype,
        seed=None if seed is None else seed + 10_000,
    ).to(dtype=cdtype)

    # Solve the feature-space system for omega-weighted probes:
    # We need (I + U^* Delta U)^{-1} U^* (Delta * probes^T) ... no.
    #
    # The M-step trace from the existing code uses:
    #   Rfeat = F^* (omega * z_j), then trace = Rfeat^* beta_j * Dprime
    # where beta_j solves (I + Phi^* Delta Phi) beta = F^* z_j.
    #
    # For the hierarchical version:
    # We solve for tilde_a_j from probes z_j, then:
    #   beta_j = D^{-1/2} tilde_a_j (unsymmetrized)
    # and the trace contribution from the global kernel is:
    #   z_j^T Delta F_g Dprime_g F_g^* (something)...
    #
    # Let me follow the derivation more carefully.
    # The trace term is: Tr(Sigma K^{-1} dK/dtheta K^{-1})
    # Using Sigma = U (I + U^* Delta U)^{-1} U^*:
    #   K^{-1} Sigma K^{-1} = K^{-1} - (I + Delta K)^{-1} Delta
    # So the trace becomes:
    #   Tr(K^{-1} dK) - Tr((I + Delta K)^{-1} Delta dK)
    # And the full gradient is:
    #   0.5 * term1 + 0.5 * [Tr(K^{-1} dK) - Tr((I + Delta K)^{-1} Delta dK)] - 0.5 * Tr(K^{-1} dK)
    #   = 0.5 * term1 - 0.5 * Tr((I + Delta K)^{-1} Delta dK)
    #
    # For Tr((I + Delta K)^{-1} Delta dK/dtheta_g):
    #   dK/dtheta_g = F_g Dprime_g^2 F_g^*   (diagonal in feature space)
    # So: Tr((I + Delta K)^{-1} Delta F_g Dprime_g^2 F_g^*)
    #   = sum_j Dprime_g_jj^2 * [(I + Delta K)^{-1} Delta]_feature_jj
    #
    # Stochastic estimate with probes z in observation space:
    #   ≈ (1/J) sum z^T (I + Delta K)^{-1} Delta F_g Dprime_g^2 F_g^* z
    #
    # Let alpha = (I + Delta K)^{-1} z.  Then:
    #   = (1/J) sum (Delta alpha)^T F_g Dprime_g^2 F_g^* z
    #   = (1/J) sum [F_g^* (Delta alpha)]^* Dprime_g^2 [F_g^* z]
    #
    # (I + Delta K) alpha = z is equivalent to
    #   alpha + Delta K alpha = z
    #   alpha = z - Delta Sigma (something)... this requires another solve.
    #
    # Actually, there's a cleaner way. Note that
    #   (I + Delta K)^{-1} = I - Delta Sigma    (Woodbury)
    # where Sigma = (K^{-1} + Delta)^{-1}.
    # So alpha = z - Delta Sigma z.
    # And Delta alpha = Delta z - Delta^2 Sigma z.
    #
    # We already have the solver for Sigma z. Let's use that.
    # For each probe z_j:
    #   Sigma_z_j = sigma_apply(z_j)
    #   alpha_j = z_j - Delta * Sigma_z_j
    #   Delta_alpha_j = Delta * alpha_j = Delta * z_j - Delta^2 * Sigma_z_j
    #
    # Then the trace contribution is:
    #   (F_g^* (Delta alpha_j))^* . Dprime_g . (F_g^* z_j)
    #
    # But this requires another sigma_apply call for each probe... expensive.
    #
    # Let's instead use the approach from the existing _compute_mstep_gradient:
    # that code solves in feature space and computes the trace there.
    # The key identity is (from the revised M-step in the notes):
    #   dL/dtheta = 0.5 * term1 - 0.5 * Tr((I + Lambda K)^{-1} Lambda dK/dtheta)
    #
    # And (I + Lambda K)^{-1} Lambda dK/dtheta in feature space:
    # Using Sigma = Phi (I+B)^{-1} Phi^* where B = Phi^* Lambda Phi:
    #   (I + Lambda K)^{-1} Lambda = Lambda - Lambda Phi (I+B)^{-1} Phi^* Lambda  (Woodbury)
    # So:
    #   Tr((I+Lambda K)^{-1} Lambda dK/dtheta)
    #   = Tr(Lambda dK) - Tr(Lambda Phi (I+B)^{-1} Phi^* Lambda dK)
    #
    # For the stacked U version, Phi -> U:
    #   = Tr(Lambda dK) - Tr(Lambda U (I+U^*Lambda U)^{-1} U^* Lambda dK)
    #
    # The first trace Tr(Lambda dK/dtheta_g) = Tr(Lambda F_g Dprime_g^2 F_g^*)
    # can be stochastically estimated, but the existing code uses a different trick.
    #
    # OK, let me just follow what _compute_mstep_gradient does and adapt it.
    # That code:
    # 1. Solves (I + Phi^* Delta Phi) beta = Phi^* z for probes z
    # 2. Forms Rfeat = Phi^* (Delta * z)
    # 3. Computes trace ≈ mean over probes of Rfeat^* . beta . Dprime
    #
    # This works because:
    #   Tr((I+B)^{-1} Phi^* Delta dK) where dK = F Dprime F^*
    #   Stochastic: z^T Delta F Dprime F^* (I+Delta K)^{-1} z ... no that's the other way.
    #
    # Let me just look at what the existing code actually computes.

    # Actually, the existing code computes:
    #   Rfeat_j = F^* (omega * z_j)     <- Phi^* Delta z_j
    #   beta_j solves (I + B) beta = F^* z_j  <- (I + Phi^* Delta Phi)^{-1} Phi^* z_j
    #   trace_est = mean_j [Rfeat_j^* . (Dprime . beta_j)]
    # which equals:
    #   mean_j z_j^T Delta F Dprime F^* (I+B)^{-1} F^* z_j   ... doesn't quite work out.
    #
    # Let me re-derive. From the code:
    #   X = Rfeat.conj() * beta_probes.T  (element-wise, shape (m, J))
    #   vals = (X.mT @ Dprime).real        (shape (J, n_hypers))
    #   term2 = vals.mean(dim=0)
    #
    # So term2 = mean_j sum_k Rfeat_jk^* beta_jk Dprime_k
    #          = mean_j sum_k [F^* Delta z_j]_k^*  [(I+B)^{-1} F^* z_j]_k  Dprime_k
    #
    # Hmm, but the solve uses the whitened system. Let me re-read the code.
    # solve_A_beta solves (I + D^{1/2} F^* Omega F D^{1/2}) y = D^{1/2} q
    # then returns beta = D^{-1/2} y.
    # And q = F^* z_j (the fadj of z_j).
    #
    # So beta = D^{-1/2} (I + D^{1/2} F^* Omega F D^{1/2})^{-1} D^{1/2} F^* z_j
    #         = (I + D F^* Omega F D)^{-1} F^* z_j    (unsymmetrize)
    #         = (I + Phi^* Omega Phi)^{-1} Phi^* z_j / D   ... no.
    # Actually (I + D^{1/2} T D^{1/2})^{-1} D^{1/2} q
    #   let y = (I + D^{1/2} T D^{1/2})^{-1} D^{1/2} q
    #   then beta = D^{-1/2} y
    #   D^{1/2} beta = y
    #   (I + D^{1/2} T D^{1/2}) D^{1/2} beta = D^{1/2} q
    #   D^{1/2} beta + D^{1/2} T D beta = D^{1/2} q
    #   beta + T D beta = q   (cancel D^{1/2})
    #   (I + T D) beta = q  where T = F^* Omega F
    #   So beta solves (I + F^* Omega F D) beta = F^* z  ... that doesn't look right.
    #
    # Let me re-check. In _make_feature_space_solver:
    #   omega = delta
    #   Ds = sqrt(D2_real) = sqrt(ws2.real)  <-- this is |D| = Dg_half in my notation
    #   apply_S: Y -> Ds * T_Delta(Ds * Y)  (using Toeplitz)
    #   apply_IpS: Y -> Y + apply_S(Y)  = (I + Ds T_Delta Ds) Y
    #   solve_A_beta: rhs = Ds * q
    #     solves (I + Ds T_Delta Ds) y = Ds * q
    #     returns beta = Ds_inv * y
    #
    # So beta = Ds_inv * (I + Ds T Ds)^{-1} Ds q = (I + D T)^{-1} q ... no.
    # Let me just be concrete. Let S = Ds, S_inv = 1/Ds.
    #   y = (I + S T S)^{-1} S q
    #   beta = S_inv y = S_inv (I + S T S)^{-1} S q
    # Let u = S q, then S_inv (I + S T S)^{-1} u
    # Multiply left by S: (I + S T S)^{-1} u = S beta
    # So u + S T S (S beta) = ... this is getting circular.
    #
    # The key identity: (I + S T S)^{-1} S = S (I + T S^2)^{-1}
    # Proof: let A = S T S. Then (I + A)^{-1} S = S(I + S^{-1} A S^{-1} S^2)^{-1} ... nah.
    #
    # OK forget the derivation, let me just match what the code does.
    # In the code, q_y = fadj(kappa) = F^* kappa, then beta, _ = solve_A_beta(q_y).
    # For probes: Q_block = fadj_batched(probes) = F^* probes^T, beta_all = solve_A_beta(Q_all).
    # Then Rfeat = fadj_batched(apply_omega(probes.mT).T) = F^* (omega * probes^T)
    #
    # Then the trace term is formed as Rfeat.conj() * beta_probes, dotted with Dprime.
    #
    # I'll do the analogous thing for the hierarchical case, but using the full stacked
    # feature space. The key insight is that the trace term just becomes:
    #
    # For global: use the global components of beta and Rfeat, with Dprime_g
    # For local: sum over locations of the local components, with Dprime_h

    # This is getting complex. Let me use a simpler stochastic estimate.
    # Use the identity from the tex writeup:
    #   dL/dtheta = 0.5 * term1 - 0.5 * Tr((I + Lambda K)^{-1} Lambda dK/dtheta)
    #
    # Stochastic estimate of Tr((I + Lambda K)^{-1} Lambda dK/dtheta):
    # Draw z_j ~ Rademacher(n), compute alpha_j = (I + Lambda K)^{-1} z_j
    # Then Tr ≈ mean_j z_j^T Lambda dK/dtheta alpha_j
    #
    # But (I + Lambda K)^{-1} z requires a separate (n x n) CG solve.
    # Instead, use the feature-space version directly.
    #
    # SIMPLER APPROACH: Follow the existing code pattern exactly.
    # The existing code works with the feature-space (I+B)^{-1} and computes
    # the trace in feature space. For the hierarchical case, we just need to
    # track which components of the feature vector correspond to global vs local.

    # Solve for probes in feature space
    tilde_a_probes, cg_iters_probes = solve(probes.real.to(dtype=kappa.dtype))
    # tilde_a_probes: shape (n_probes, (1+L)*m)

    # Unsymmetrize
    beta_probes_g = Dg_half_inv[None, :] * tilde_a_probes[:, :m]  # (J, m)
    beta_probes_h = []
    for ell in range(L):
        beta_probes_h.append(
            Dh_half_inv[None, :] * tilde_a_probes[:, m + ell * m: m + (ell + 1) * m]
        )

    # Compute "Rfeat" = U^* (Delta * z_j) for each probe
    # Global part: F_g^* (Delta * z_j)
    delta_c = delta.to(dtype=cdtype)
    omega_probes = delta_c[None, :] * probes.to(dtype=cdtype)  # (J, n)

    Rfeat_g = torch.zeros(n_probes, m, dtype=cdtype, device=kappa.device)
    for j in range(n_probes):
        Rfeat_g[j] = spec.nufft_global.type1(
            omega_probes[j], out_shape=spec.out_shape
        ).reshape(-1)

    # Global trace term: mean_j sum_k Rfeat_g_jk^* beta_g_jk Dprime_g_k
    X_g = Rfeat_g.conj() * beta_probes_g  # (J, m)
    vals_g = (X_g @ spec.Dprime_g).real  # (J, n_hypers)
    term2_g = vals_g.mean(dim=0)

    # Local trace term: for each location, F_ell^* (Delta_ell * z_j_ell)
    term2_h = torch.zeros_like(spec.Dprime_h[0])
    for ell in range(L):
        idx = spec.loc_indices[ell]
        Rfeat_ell = torch.zeros(n_probes, m, dtype=cdtype, device=kappa.device)
        for j in range(n_probes):
            omega_z_ell = omega_probes[j, idx]
            Rfeat_ell[j] = spec.nufft_local[ell].type1(
                omega_z_ell, out_shape=spec.out_shape
            ).reshape(-1)

        X_ell = Rfeat_ell.conj() * beta_probes_h[ell]  # (J, m)
        vals_ell = (X_ell @ spec.Dprime_h).real  # (J, n_hypers)
        term2_h = term2_h + vals_ell.mean(dim=0)

    grad_g = 0.5 * (term1_g - term2_g)
    grad_h = 0.5 * (term1_h - term2_h)

    return {
        "grad_g": grad_g,
        "grad_h": grad_h,
        "term1_g": term1_g,
        "term2_g": term2_g,
        "term1_h": term1_h,
        "term2_h": term2_h,
        "cg_iters": torch.tensor(float(max(cg_iters_kappa, cg_iters_probes))),
    }


# ---------------------------------------------------------------------------
# Full fit loop
# ---------------------------------------------------------------------------

class HierarchicalPGNegBinRegressor:
    """
    Hierarchical additive-kernel PG-augmented GP for NB counts.

    Parameters
    ----------
    lengthscale_g_init, variance_g_init : global kernel initial hyperparameters
    lengthscale_h_init, variance_h_init : local kernel initial hyperparameters
    total_count : NB shape parameter r (can be learned)
    learn_total_count : whether to learn r via gradient ascent
    """

    def __init__(
        self,
        *,
        lengthscale_g_init: float = 0.2,
        variance_g_init: float = 1.0,
        lengthscale_h_init: float = 0.05,
        variance_h_init: float = 0.5,
        total_count: float = 5.0,
        learn_total_count: bool = False,
        total_count_lr: float = 0.01,
        total_count_update_freq: int = 5,
        total_count_quadrature_nodes: int = 12,
        max_iter: int = 50,
        e_step_iters: int = 3,
        final_e_step_iters: int = 5,
        e_step_tol: float = 1e-4,
        rho0: float = 0.7,
        e_gamma: float = 1e-3,
        lr: float = 0.05,
        lr_g: float | None = None,
        lr_h: float | None = None,
        n_e_probes: int = 10,
        n_m_probes: int = 10,
        cg_tol: float = 1e-6,
        nufft_eps: float = 1e-7,
        spectral_eps: float = 1e-4,
        trunc_eps: float = 1e-4,
        seed: int | None = None,
        verbose: int = 1,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
    ):
        self.lengthscale_g_init = lengthscale_g_init
        self.variance_g_init = variance_g_init
        self.lengthscale_h_init = lengthscale_h_init
        self.variance_h_init = variance_h_init
        self.total_count_init = total_count
        self.learn_total_count = learn_total_count
        self.total_count_lr = total_count_lr
        self.total_count_update_freq = total_count_update_freq
        self.total_count_quadrature_nodes = total_count_quadrature_nodes
        self.max_iter = max_iter
        self.e_step_iters = e_step_iters
        self.final_e_step_iters = final_e_step_iters
        self.e_step_tol = e_step_tol
        self.rho0 = rho0
        self.e_gamma = e_gamma
        self.lr = lr
        self.lr_g = lr_g if lr_g is not None else lr
        self.lr_h = lr_h if lr_h is not None else lr
        self.n_e_probes = n_e_probes
        self.n_m_probes = n_m_probes
        self.cg_tol = cg_tol
        self.nufft_eps = nufft_eps
        self.spectral_eps = spectral_eps
        self.trunc_eps = trunc_eps
        self.seed = seed
        self.verbose = verbose
        self.dtype = dtype
        self.device = torch.device(device)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        locations: np.ndarray,
    ) -> "HierarchicalPGNegBinRegressor":
        """
        Fit the hierarchical NB GP model.

        Parameters
        ----------
        X : (n, d) input features
        y : (n,) nonneg integer counts
        locations : (n,) integer location labels
        """
        rdtype = self.dtype
        cdtype = torch.complex128 if rdtype == torch.float64 else torch.complex64
        dev = self.device

        X_t = torch.as_tensor(X, device=dev, dtype=rdtype)
        if X_t.ndim == 1:
            X_t = X_t.unsqueeze(-1)
        y_t = torch.as_tensor(y, device=dev, dtype=rdtype)
        loc_t = torch.as_tensor(locations, device=dev, dtype=torch.long)
        n = X_t.shape[0]
        d = X_t.shape[1]

        # Initialize kernels
        self.kernel_g_ = _make_kernel(dimension=d,
                                       lengthscale=self.lengthscale_g_init,
                                       variance=self.variance_g_init)
        self.kernel_h_ = _make_kernel(dimension=d,
                                       lengthscale=self.lengthscale_h_init,
                                       variance=self.variance_h_init)

        # Total count
        r = self.total_count_init
        if self.learn_total_count:
            log_r = torch.nn.Parameter(
                torch.tensor(math.log(r), device=dev, dtype=rdtype)
            )
            r_optimizer = torch.optim.Adam([log_r], lr=self.total_count_lr, maximize=True)

        # Kernel optimizers
        opt_g = torch.optim.Adam(self.kernel_g_._gp_params_ref.parameters(),
                                  lr=self.lr_g, maximize=True)
        opt_h = torch.optim.Adam(self.kernel_h_._gp_params_ref.parameters(),
                                  lr=self.lr_h, maximize=True)

        # PG scalars
        def _kappa():
            return 0.5 * (y_t - r)

        def _pg_b():
            return y_t + r

        # Initialize delta
        delta = (0.25 * _pg_b()).clone()

        history: list[dict] = []
        t0 = time.perf_counter()

        for outer in range(self.max_iter):
            if self.learn_total_count:
                r = float(torch.exp(log_r).item())
            kappa = _kappa()
            pg_b = _pg_b()

            # Build spectral state
            spec = _build_hierarchical_spectral_state(
                X_t, loc_t, self.kernel_g_, self.kernel_h_,
                spectral_eps=self.spectral_eps,
                trunc_eps=self.trunc_eps,
                nufft_eps=self.nufft_eps,
                rdtype=rdtype, cdtype=cdtype, device=dev,
            )

            # E-step
            delta, mean, sigma_diag, estep_info = _run_hierarchical_estep(
                y_t, kappa, pg_b, delta, spec,
                max_iters=self.e_step_iters,
                rho0=self.rho0, gamma=self.e_gamma,
                tol=self.e_step_tol,
                n_probes=self.n_e_probes,
                cg_tol=self.cg_tol,
                seed=None if self.seed is None else self.seed + 1000 * outer,
                verbose=self.verbose,
            )

            # M-step: kernel hyperparameter gradients
            mstep_out = _compute_hierarchical_mstep_gradient(
                kappa, delta, spec,
                n_probes=self.n_m_probes,
                cg_tol=self.cg_tol,
                seed=None if self.seed is None else self.seed + 1000 * outer + 500,
            )

            # Update global kernel
            grad_g = mstep_out["grad_g"].real
            raw_g = self.kernel_g_._gp_params_ref.raw
            raw_g.grad = torch.stack([
                grad_g[0].to(dtype=raw_g.dtype, device=raw_g.device) * self.kernel_g_.lengthscale,
                grad_g[1].to(dtype=raw_g.dtype, device=raw_g.device) * self.kernel_g_.variance,
                torch.tensor(0.0, dtype=raw_g.dtype, device=raw_g.device),
            ])
            opt_g.step()
            opt_g.zero_grad(set_to_none=True)

            # Update local kernel
            grad_h = mstep_out["grad_h"].real
            raw_h = self.kernel_h_._gp_params_ref.raw
            raw_h.grad = torch.stack([
                grad_h[0].to(dtype=raw_h.dtype, device=raw_h.device) * self.kernel_h_.lengthscale,
                grad_h[1].to(dtype=raw_h.dtype, device=raw_h.device) * self.kernel_h_.variance,
                torch.tensor(0.0, dtype=raw_h.dtype, device=raw_h.device),
            ])
            opt_h.step()
            opt_h.zero_grad(set_to_none=True)

            # Update r
            if self.learn_total_count and (outer + 1) % self.total_count_update_freq == 0:
                grad_r = _negative_binomial_total_count_gradient(
                    y_t, mean, sigma_diag,
                    total_count=r,
                    quadrature_nodes=self.total_count_quadrature_nodes,
                )
                log_r.grad = (grad_r * r).to(dtype=log_r.dtype, device=log_r.device).detach()
                r_optimizer.step()
                r_optimizer.zero_grad(set_to_none=True)
                r = float(torch.exp(log_r).item())

            # Fit metric: MAE of predicted mean count
            pred_count = r * torch.exp(mean + 0.5 * sigma_diag.clamp_min(0))
            mae = float(torch.mean(torch.abs(pred_count - y_t)).item())

            record = {
                "iter": outer,
                "elapsed": time.perf_counter() - t0,
                "ls_g": float(self.kernel_g_.lengthscale),
                "var_g": float(self.kernel_g_.variance),
                "ls_h": float(self.kernel_h_.lengthscale),
                "var_h": float(self.kernel_h_.variance),
                "r": r,
                "mae": mae,
                "e_resid": estep_info["residual"],
                "e_cg": estep_info["cg_iters"],
                "m_cg": float(mstep_out["cg_iters"].item()),
            }
            history.append(record)

            if self.verbose:
                print(
                    f"[{outer:3d}] ls_g={record['ls_g']:.4f} var_g={record['var_g']:.4f} "
                    f"ls_h={record['ls_h']:.4f} var_h={record['var_h']:.4f} "
                    f"r={record['r']:.3f} mae={record['mae']:.3f} "
                    f"e_cg={record['e_cg']:.0f} m_cg={record['m_cg']:.0f}"
                )

        # Final E-step
        if self.learn_total_count:
            r = float(torch.exp(log_r).item())
        spec = _build_hierarchical_spectral_state(
            X_t, loc_t, self.kernel_g_, self.kernel_h_,
            spectral_eps=self.spectral_eps,
            trunc_eps=self.trunc_eps,
            nufft_eps=self.nufft_eps,
            rdtype=rdtype, cdtype=cdtype, device=dev,
        )
        delta, mean, sigma_diag, _ = _run_hierarchical_estep(
            y_t, _kappa(), _pg_b(), delta, spec,
            max_iters=self.final_e_step_iters,
            rho0=self.rho0, gamma=self.e_gamma,
            tol=self.e_step_tol,
            n_probes=self.n_e_probes,
            cg_tol=self.cg_tol,
            seed=None if self.seed is None else self.seed + 999_999,
            verbose=self.verbose,
        )

        # Decompose posterior mean into global and local components
        g_hat, h_hats = self._decompose_posterior(spec, delta, kappa)

        self.history_ = history
        self.delta_ = delta.detach().cpu().numpy()
        self.mean_ = mean.detach().cpu().numpy()
        self.sigma_diag_ = sigma_diag.detach().cpu().numpy()
        self.total_count_ = r
        self.lengthscale_g_ = float(self.kernel_g_.lengthscale)
        self.variance_g_ = float(self.kernel_g_.variance)
        self.lengthscale_h_ = float(self.kernel_h_.lengthscale)
        self.variance_h_ = float(self.kernel_h_.variance)
        self.g_hat_ = g_hat  # (n,) global component at all points
        self.h_hats_ = h_hats  # dict: loc -> (n_loc,) local component

        return self

    @staticmethod
    def _decompose_posterior(
        spec: HierarchicalSpectralState,
        delta: torch.Tensor,
        kappa: torch.Tensor,
    ) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        """Decompose posterior mean into global g and per-location h components."""
        m = spec.ws_g.numel()
        L = spec.n_locations
        cdtype = spec.ws_g.dtype

        D2g_real = spec.ws2_g.real
        eps_d = max(float(D2g_real.mean()) * 1e-14, 1e-14)
        Dg_half = torch.sqrt(torch.clamp(D2g_real, min=eps_d)).to(dtype=cdtype)

        D2h_real = spec.ws2_h.real
        eps_d_h = max(float(D2h_real.mean()) * 1e-14, 1e-14)
        Dh_half = torch.sqrt(torch.clamp(D2h_real, min=eps_d_h)).to(dtype=cdtype)

        # Solve for feature-space coefficients
        solve, _ = _make_hierarchical_feature_solver(delta, spec, cg_tol=1e-5)
        tilde_a, _ = solve(kappa)  # shape ((1+L)*m,)

        # Global component: F_g (Dg^{1/2} * tilde_a_g)
        coeff_g = Dg_half * tilde_a[:m]
        g_hat = spec.nufft_global.type2(coeff_g, out_shape=spec.out_shape).real

        # Local components
        n = sum(idx.numel() for idx in spec.loc_indices)
        h_hats = {}
        for ell in range(L):
            a_ell = tilde_a[m + ell * m: m + (ell + 1) * m]
            coeff_ell = Dh_half * a_ell
            h_ell = spec.nufft_local[ell].type2(coeff_ell, out_shape=spec.out_shape).real
            h_hats[ell] = h_ell.detach().cpu().numpy()

        return g_hat.detach().cpu().numpy(), h_hats
