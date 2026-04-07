"""
Hierarchical additive-kernel PG-augmented GP for negative binomial counts.
Block coordinate descent variant.

E-step: Instead of solving a single (1+L)*m CG system, we alternate:
  1. Fix all h_ell, solve for g (m-dim CG on all n points)
  2. Fix g, solve each h_ell independently (m-dim CG on n_ell points each)

M-step: Uses the original joint (1+L)*m solver for correct gradients.
  (Only 2 solves per outer iter vs ~33 in the E-step, so this is cheap.)

Model:
    f_i = g(t_i) + h_{ell_i}(t_i)
    g ~ GP(0, k_g),  h_ell ~iid GP(0, k_h)
    y_i | f_i, r ~ NB(r, sigma(f_i))
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
from torch.fft import fftn, ifftn

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cg import ConjugateGradients
from efgpnd import NUFFT, ToeplitzND
from kernels import SquaredExponential
from utils.kernels import get_xis

_PG_ROOT = Path(__file__).resolve().parents[1]
if str(_PG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PG_ROOT))

from pg_classifier import (
    _pg_omega_expectation,
    _sample_rademacher,
    _negative_binomial_total_count_gradient,
    _gauss_hermite_normal_rule,
    negative_binomial_gaussian_mean,
    _SpectralState,
    _build_spectral_state,
    _build_weighted_toeplitz,
)

from hierarchical.pg_hierarchical import (
    _build_hierarchical_spectral_state,
    _compute_hierarchical_mstep_gradient,
)

# Re-exports kept for backward compatibility; the fully-block M-step
# (_compute_blockcd_mstep_gradient_coupled below) removes the need for
# the joint solver, but we keep the import so test scripts can choose.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_kernel(*, dimension: int, lengthscale: float, variance: float) -> SquaredExponential:
    return SquaredExponential(
        dimension=dimension,
        init_lengthscale=lengthscale,
        init_variance=variance,
    )


def _build_local_spectral_state(
    X_ell: torch.Tensor,
    kernel: SquaredExponential,
    *,
    xis_1d: torch.Tensor,
    h: float,
    mtot: int,
    spectral_eps: float,
    nufft_eps: float,
    rdtype: torch.dtype,
    cdtype: torch.dtype,
    device: torch.device,
) -> _SpectralState:
    """Build a _SpectralState for a subset of points, reusing the shared grid."""
    d = X_ell.shape[1]
    grids = torch.meshgrid(*(xis_1d for _ in range(d)), indexing="ij")
    xis = torch.stack(grids, dim=-1).view(-1, d)
    out_shape = (mtot,) * d

    sd = kernel.spectral_density(xis).to(dtype=rdtype)
    ws2 = (sd * h**d).to(device=device, dtype=cdtype)
    ws = torch.sqrt(ws2)
    Dprime = (h**d * kernel.spectral_grad(xis)).to(device=device, dtype=cdtype)

    xcen = torch.zeros(d, device=device, dtype=rdtype)
    nufft_op = NUFFT(X_ell, xcen, h, nufft_eps, cdtype=cdtype, device=device)

    conv_shape = tuple(2 * n - 1 for n in out_shape)
    ones = torch.ones(X_ell.shape[0], device=device, dtype=cdtype)
    v = nufft_op.type1(ones, out_shape=conv_shape)
    toeplitz = ToeplitzND(v.to(dtype=cdtype), force_pow2=True)

    def fadj(z):
        return nufft_op.type1(z, out_shape=out_shape).reshape(-1)

    def fwd(c):
        return nufft_op.type2(c, out_shape=out_shape)

    def fadj_batched(z):
        if z.dim() == 1:
            return fadj(z).unsqueeze(0)
        return torch.stack([fadj(z[i]) for i in range(z.shape[0])])

    def fwd_batched(c):
        if c.dim() == 1:
            return fwd(c).unsqueeze(0)
        return torch.stack([fwd(c[i]) for i in range(c.shape[0])])

    return _SpectralState(
        xis=xis.to(device=device, dtype=rdtype),
        h=h, mtot=mtot, ws=ws, ws2=ws2, Dprime=Dprime,
        toeplitz=toeplitz, nufft_op=nufft_op,
        fadj=fadj, fwd=fwd,
        fadj_batched=fadj_batched, fwd_batched=fwd_batched,
        out_shape=out_shape,
    )


def _solve_single_gp(
    z: torch.Tensor,
    delta: torch.Tensor,
    spec: _SpectralState,
    *,
    cg_tol: float,
    x0: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    """
    Solve for posterior mean of a single GP component.

    Given pseudo-observations z and PG precision delta, compute:
        mean = (K^{-1} + Delta)^{-1} z

    Returns (mean, cg_iters, x_feat) where x_feat is the feature-space
    coefficient (tilde_a in the symmetrized system).
    """
    cdtype = spec.ws.dtype

    # Build weighted Toeplitz for this component
    delta_c = delta.to(dtype=cdtype)
    conv_shape = tuple(2 * n - 1 for n in spec.out_shape)
    v = spec.nufft_op.type1(delta_c, out_shape=conv_shape)
    T = ToeplitzND(v.to(dtype=cdtype), force_pow2=True)

    # RHS: ws * F^* z
    rhs = spec.ws * spec.fadj(z.to(dtype=cdtype))

    def A_feat(u):
        return u + spec.ws * T(spec.ws * u)

    init = x0 if x0 is not None else torch.zeros_like(rhs)
    cg = ConjugateGradients(A_feat, rhs, x0=init,
                            tol=cg_tol, max_iter=2000, early_stopping=True)
    x = cg.solve()

    mean = spec.fwd(spec.ws * x).real
    return mean, cg.iters_completed, x


def _solve_single_gp_with_probes(
    z: torch.Tensor,
    probes: torch.Tensor,
    delta: torch.Tensor,
    spec: _SpectralState,
    *,
    cg_tol: float,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Solve for posterior mean and Hutchinson variance estimate simultaneously.

    z: (n,) pseudo-observations (kappa for this component)
    probes: (n_probes, n) Rademacher vectors
    delta: (n,) PG precision

    Returns (mean, sigma_diag_estimate, cg_iters).
    """
    cdtype = spec.ws.dtype

    delta_c = delta.to(dtype=cdtype)
    conv_shape = tuple(2 * n - 1 for n in spec.out_shape)
    v = spec.nufft_op.type1(delta_c, out_shape=conv_shape)
    T = ToeplitzND(v.to(dtype=cdtype), force_pow2=True)

    # Stack z and probes: (1 + n_probes, n)
    Z = torch.cat([z.unsqueeze(0), probes], dim=0).to(dtype=cdtype)

    # RHS: ws * F^* Z  — batched
    rhs = spec.ws[None, :] * spec.fadj_batched(Z)

    def A_feat(u):
        return u + spec.ws[None, :] * T(spec.ws[None, :] * u)

    cg = ConjugateGradients(A_feat, rhs, x0=torch.zeros_like(rhs),
                            tol=cg_tol, max_iter=2000, early_stopping=True)
    x = cg.solve()

    result = spec.fwd_batched(spec.ws[None, :] * x).real
    mean = result[0]
    Sz = result[1:]
    sigma_diag = (probes * Sz).mean(dim=0)

    return mean, sigma_diag, cg.iters_completed


# ---------------------------------------------------------------------------
# Block CD feature-space solve (returns coefficients, not just means)
# ---------------------------------------------------------------------------

def _blockcd_feature_solve(
    Z: torch.Tensor,
    delta: torch.Tensor,
    spec_g: _SpectralState,
    specs_h: list[_SpectralState],
    loc_indices: list[torch.Tensor],
    *,
    cd_sweeps: int,
    cg_tol: float,
) -> dict:
    """
    Solve (I + U^* Δ U) β = U^* Z via block CD sweeps.

    Z: (B, n) or (n,) — batched RHS in observation space.
    Returns dict with:
      - 'x_g': (B, m) feature-space coefficients for global GP (tilde_a)
      - 'x_h': list of (B, m_ell) for each location
      - 'g_mean': (B, n) function-space global means
      - 'h_means': list of (B, n_ell) function-space local means
      - 'cg_iters': total CG iterations
    """
    vector_input = Z.dim() == 1
    if vector_input:
        Z = Z.unsqueeze(0)
    B = Z.shape[0]
    n = Z.shape[1]
    L = len(specs_h)
    rdtype = delta.dtype
    cdtype = spec_g.ws.dtype
    device = Z.device

    # Initialize component means in function space
    g_mean = torch.zeros(B, n, dtype=rdtype, device=device)
    h_means = [torch.zeros(B, idx.numel(), dtype=rdtype, device=device) for idx in loc_indices]

    # Feature-space coefficients (tilde_a = ws * beta in the symmetrized sense)
    m = spec_g.ws.numel()
    x_g = torch.zeros(B, m, dtype=cdtype, device=device)
    x_h = [torch.zeros(B, specs_h[ell].ws.numel(), dtype=cdtype, device=device) for ell in range(L)]

    total_cg = 0

    for sweep in range(cd_sweeps):
        # --- Solve for g, fixing all h_ell ---
        h_combined = torch.zeros(B, n, dtype=rdtype, device=device)
        for ell, idx in enumerate(loc_indices):
            h_combined[:, idx] = h_means[ell]

        z_g = Z - delta[None, :] * h_combined  # (B, n) effective pseudo-obs for g

        # Build weighted Toeplitz for global component
        delta_c = delta.to(dtype=cdtype)
        conv_shape_g = tuple(2 * s - 1 for s in spec_g.out_shape)
        v_g = spec_g.nufft_op.type1(delta_c, out_shape=conv_shape_g)
        T_g = ToeplitzND(v_g.to(dtype=cdtype), force_pow2=True)

        # Batched RHS: ws * F^* z_g for each batch
        rhs_g = torch.zeros(B, m, dtype=cdtype, device=device)
        for b in range(B):
            rhs_g[b] = spec_g.ws * spec_g.fadj(z_g[b].to(dtype=cdtype))

        def A_feat_g(u, _T=T_g, _s=spec_g):
            return u + _s.ws[None, :] * _T(_s.ws[None, :] * u)

        cg = ConjugateGradients(A_feat_g, rhs_g, x0=x_g.clone(),
                                tol=cg_tol, max_iter=2000, early_stopping=True)
        x_g = cg.solve()
        total_cg += cg.iters_completed

        # Recover g_mean in function space
        for b in range(B):
            g_mean[b] = spec_g.fwd(spec_g.ws * x_g[b]).real

        # --- Solve each h_ell, fixing g ---
        for ell, idx in enumerate(loc_indices):
            z_ell = Z[:, idx] - delta[None, idx] * g_mean[:, idx]  # (B, n_ell)

            spec_ell = specs_h[ell]
            delta_ell = delta[idx].to(dtype=cdtype)
            conv_shape_ell = tuple(2 * s - 1 for s in spec_ell.out_shape)
            v_ell = spec_ell.nufft_op.type1(delta_ell, out_shape=conv_shape_ell)
            T_ell = ToeplitzND(v_ell.to(dtype=cdtype), force_pow2=True)

            m_ell = spec_ell.ws.numel()
            rhs_ell = torch.zeros(B, m_ell, dtype=cdtype, device=device)
            for b in range(B):
                rhs_ell[b] = spec_ell.ws * spec_ell.fadj(z_ell[b].to(dtype=cdtype))

            def A_feat_ell(u, _T=T_ell, _s=spec_ell):
                return u + _s.ws[None, :] * _T(_s.ws[None, :] * u)

            cg_ell = ConjugateGradients(A_feat_ell, rhs_ell, x0=x_h[ell].clone(),
                                         tol=cg_tol, max_iter=2000, early_stopping=True)
            x_h[ell] = cg_ell.solve()
            total_cg += cg_ell.iters_completed

            for b in range(B):
                h_means[ell][b] = spec_ell.fwd(spec_ell.ws * x_h[ell][b]).real

    if vector_input:
        g_mean = g_mean.squeeze(0)
        h_means = [h.squeeze(0) for h in h_means]
        x_g = x_g.squeeze(0)
        x_h = [x.squeeze(0) for x in x_h]

    return {
        'x_g': x_g,
        'x_h': x_h,
        'g_mean': g_mean,
        'h_means': h_means,
        'cg_iters': total_cg,
    }


# ---------------------------------------------------------------------------
# Schur complement M-step gradient
# ---------------------------------------------------------------------------

def _schur_solve(
    rhs_g: torch.Tensor,
    rhs_h: list[torch.Tensor],
    *,
    Dg_half: torch.Tensor,
    Dh_half: torch.Tensor,
    T_all: ToeplitzND,
    T_locs: list[ToeplitzND],
    cg_tol: float,
    inner_cg_tol: float,
) -> tuple[torch.Tensor, list[torch.Tensor], int]:
    """
    Solve M [a_g; a_h] = [rhs_g; rhs_h] via the Schur complement.

    M = [A B; B^* D] where:
      A = I + Dg^{1/2} T_all Dg^{1/2}
      D_ℓ = I + Dh^{1/2} T_ℓ Dh^{1/2}
      B_ℓ = Dg^{1/2} T_ℓ Dh^{1/2}

    Steps:
      1. For each ℓ: w_ℓ = D_ℓ^{-1} rhs_h_ℓ
      2. Form: rhs_g_eff = rhs_g - Σ_ℓ B_ℓ w_ℓ
      3. Solve S a_g = rhs_g_eff via CG (S = A - Σ B D^{-1} B^*)
      4. Recover a_h_ℓ = D_ℓ^{-1}(rhs_h_ℓ - B_ℓ^* a_g)

    rhs_g: (B, m) or (m,) — batched or single
    rhs_h: list of (B, m) or (m,) for each location

    Returns (a_g, a_h_list, total_cg_iters).
    """
    L = len(rhs_h)
    vector_input = rhs_g.dim() == 1
    if vector_input:
        rhs_g = rhs_g.unsqueeze(0)
        rhs_h = [r.unsqueeze(0) for r in rhs_h]

    B = rhs_g.shape[0]
    m = rhs_g.shape[1]
    total_cg = 0

    # --- Step 1: D_ℓ^{-1} rhs_h_ℓ for each ℓ ---
    w_h = []
    for ell in range(L):
        T_ell = T_locs[ell]

        def D_ell_apply(u, _T=T_ell):
            return u + Dh_half[None, :] * _T(Dh_half[None, :] * u)

        cg = ConjugateGradients(D_ell_apply, rhs_h[ell],
                                x0=torch.zeros_like(rhs_h[ell]),
                                tol=inner_cg_tol, max_iter=500, early_stopping=True)
        w_ell = cg.solve()
        total_cg += cg.iters_completed
        w_h.append(w_ell)

    # --- Step 2: rhs_g_eff = rhs_g - Σ_ℓ B_ℓ w_ℓ ---
    rhs_g_eff = rhs_g.clone()
    for ell in range(L):
        # B_ℓ w = Dg^{1/2} T_ℓ(Dh^{1/2} w)
        B_w = Dg_half[None, :] * T_locs[ell](Dh_half[None, :] * w_h[ell])
        rhs_g_eff = rhs_g_eff - B_w

    # --- Step 3: Solve S a_g = rhs_g_eff ---
    def schur_matvec(v):
        # A v
        Av = v + Dg_half[None, :] * T_all(Dg_half[None, :] * v)

        # Σ_ℓ B_ℓ D_ℓ^{-1} B_ℓ^* v
        correction = torch.zeros_like(v)
        for ell in range(L):
            T_ell = T_locs[ell]
            # B_ℓ^* v = Dh^{1/2} T_ℓ(Dg^{1/2} v)
            Bstar_v = Dh_half[None, :] * T_ell(Dg_half[None, :] * v)

            # D_ℓ^{-1} Bstar_v
            def D_ell_apply(u, _T=T_ell):
                return u + Dh_half[None, :] * _T(Dh_half[None, :] * u)

            inner_cg = ConjugateGradients(D_ell_apply, Bstar_v,
                                          x0=torch.zeros_like(Bstar_v),
                                          tol=inner_cg_tol, max_iter=500,
                                          early_stopping=True)
            w = inner_cg.solve()

            # B_ℓ w = Dg^{1/2} T_ℓ(Dh^{1/2} w)
            correction = correction + Dg_half[None, :] * T_ell(Dh_half[None, :] * w)

        return Av - correction

    cg_schur = ConjugateGradients(schur_matvec, rhs_g_eff,
                                  x0=torch.zeros_like(rhs_g_eff),
                                  tol=cg_tol, max_iter=2000, early_stopping=True)
    a_g = cg_schur.solve()
    total_cg += cg_schur.iters_completed

    # --- Step 4: a_h_ℓ = D_ℓ^{-1}(rhs_h_ℓ - B_ℓ^* a_g) ---
    a_h = []
    for ell in range(L):
        T_ell = T_locs[ell]
        # B_ℓ^* a_g = Dh^{1/2} T_ℓ(Dg^{1/2} a_g)
        Bstar_ag = Dh_half[None, :] * T_ell(Dg_half[None, :] * a_g)
        rhs_ell_corrected = rhs_h[ell] - Bstar_ag

        def D_ell_apply(u, _T=T_ell):
            return u + Dh_half[None, :] * _T(Dh_half[None, :] * u)

        cg_ell = ConjugateGradients(D_ell_apply, rhs_ell_corrected,
                                    x0=w_h[ell],  # warm-start from step 1
                                    tol=inner_cg_tol, max_iter=500,
                                    early_stopping=True)
        a_h.append(cg_ell.solve())
        total_cg += cg_ell.iters_completed

    if vector_input:
        a_g = a_g.squeeze(0)
        a_h = [a.squeeze(0) for a in a_h]

    return a_g, a_h, total_cg


def _compute_schur_mstep_gradient(
    kappa: torch.Tensor,
    delta: torch.Tensor,
    spec_g: _SpectralState,
    specs_h: list[_SpectralState],
    loc_indices: list[torch.Tensor],
    *,
    n_probes: int,
    cg_tol: float,
    inner_cg_tol: float | None = None,
    seed: int | None,
    x_g_warm: torch.Tensor | None = None,
    x_h_warm: list[torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """
    Compute M-step gradients via the Schur complement.

    Uses E-step coefficients for term1 (data-fit) if provided,
    and Schur complement solves for term2 (trace via Hutchinson probes).

    The Schur complement analytically eliminates h, giving an m-dim CG
    for the g-block instead of an (1+L)*m coupled system.
    """
    n = kappa.numel()
    L = len(specs_h)
    m = spec_g.ws.numel()
    cdtype = spec_g.ws.dtype
    rdtype = kappa.dtype
    device = kappa.device
    if inner_cg_tol is None:
        inner_cg_tol = cg_tol

    # Build all Toeplitz operators
    Dg_half = spec_g.ws
    Dg_half_inv = 1.0 / Dg_half

    delta_c = delta.to(dtype=cdtype)
    conv_shape = tuple(2 * s - 1 for s in spec_g.out_shape)
    v_all = spec_g.nufft_op.type1(delta_c, out_shape=conv_shape)
    T_all = ToeplitzND(v_all.to(dtype=cdtype), force_pow2=True)

    T_locs = []
    Dh_halfs = []
    for ell, idx in enumerate(loc_indices):
        spec_ell = specs_h[ell]
        Dh_halfs.append(spec_ell.ws)
        delta_ell = delta[idx].to(dtype=cdtype)
        conv_shape_ell = tuple(2 * s - 1 for s in spec_ell.out_shape)
        v_ell = spec_ell.nufft_op.type1(delta_ell, out_shape=conv_shape_ell)
        T_locs.append(ToeplitzND(v_ell.to(dtype=cdtype), force_pow2=True))

    # All locations share the same kernel, so Dh_half is the same
    Dh_half = Dh_halfs[0]

    total_cg = 0

    # --- Term 1: data-fit from E-step coefficients ---
    if x_g_warm is not None and x_h_warm is not None:
        beta_g = x_g_warm / Dg_half
        betas_h = [x_h_warm[ell] / Dh_halfs[ell] for ell in range(L)]
    else:
        # Fall back: solve via Schur for kappa
        rhs_g_kappa = Dg_half * spec_g.fadj(kappa.to(dtype=cdtype))
        rhs_h_kappa = []
        for ell, idx in enumerate(loc_indices):
            rhs_h_kappa.append(
                Dh_halfs[ell] * specs_h[ell].fadj(kappa[idx].to(dtype=cdtype))
            )
        a_g, a_h, cg_k = _schur_solve(
            rhs_g_kappa, rhs_h_kappa,
            Dg_half=Dg_half, Dh_half=Dh_half,
            T_all=T_all, T_locs=T_locs,
            cg_tol=cg_tol, inner_cg_tol=inner_cg_tol,
        )
        total_cg += cg_k
        beta_g = a_g / Dg_half
        betas_h = [a_h[ell] / Dh_halfs[ell] for ell in range(L)]

    abs2_g = (beta_g.conj() * beta_g).real
    n_hypers = spec_g.Dprime.shape[1]
    term1_g = spec_g.Dprime.real.T @ abs2_g

    term1_h = torch.zeros(n_hypers, dtype=rdtype, device=device)
    for ell in range(L):
        abs2_ell = (betas_h[ell].conj() * betas_h[ell]).real
        term1_h = term1_h + specs_h[ell].Dprime.real.T @ abs2_ell

    # --- Term 2: stochastic trace via Schur probe solves ---
    probes = _sample_rademacher(
        (n_probes, n), device=device, dtype=rdtype,
        seed=None if seed is None else seed + 10_000,
    )

    # Build RHS for probes: D^{1/2} U^* z
    probes_c = probes.to(dtype=cdtype)
    rhs_g_probes = torch.zeros(n_probes, m, dtype=cdtype, device=device)
    for j in range(n_probes):
        rhs_g_probes[j] = Dg_half * spec_g.fadj(probes_c[j])

    rhs_h_probes = []
    for ell, idx in enumerate(loc_indices):
        rhs_ell = torch.zeros(n_probes, specs_h[ell].ws.numel(), dtype=cdtype, device=device)
        for j in range(n_probes):
            rhs_ell[j] = Dh_halfs[ell] * specs_h[ell].fadj(probes_c[j, idx])
        rhs_h_probes.append(rhs_ell)

    # Solve via Schur complement
    a_g_probes, a_h_probes, cg_p = _schur_solve(
        rhs_g_probes, rhs_h_probes,
        Dg_half=Dg_half, Dh_half=Dh_half,
        T_all=T_all, T_locs=T_locs,
        cg_tol=cg_tol, inner_cg_tol=inner_cg_tol,
    )
    total_cg += cg_p

    # Un-symmetrize probe solutions
    beta_probes_g = a_g_probes / Dg_half[None, :]
    beta_probes_h = [a_h_probes[ell] / Dh_halfs[ell][None, :] for ell in range(L)]

    # Compute Rfeat = F^*(Delta * z_j) for each probe
    delta_c = delta.to(dtype=cdtype)
    omega_probes = delta_c[None, :] * probes_c

    # Global Rfeat
    Rfeat_g = torch.zeros(n_probes, m, dtype=cdtype, device=device)
    for j in range(n_probes):
        Rfeat_g[j] = spec_g.nufft_op.type1(
            omega_probes[j], out_shape=spec_g.out_shape
        ).reshape(-1)

    X_g = Rfeat_g.conj() * beta_probes_g.to(dtype=cdtype)
    vals_g = (X_g @ spec_g.Dprime).real
    term2_g = vals_g.mean(dim=0)

    # Local Rfeat and trace
    term2_h = torch.zeros(n_hypers, dtype=rdtype, device=device)
    for ell, idx in enumerate(loc_indices):
        spec_ell = specs_h[ell]
        m_ell = spec_ell.ws.numel()
        Rfeat_ell = torch.zeros(n_probes, m_ell, dtype=cdtype, device=device)
        for j in range(n_probes):
            Rfeat_ell[j] = spec_ell.nufft_op.type1(
                omega_probes[j, idx], out_shape=spec_ell.out_shape
            ).reshape(-1)

        X_ell = Rfeat_ell.conj() * beta_probes_h[ell].to(dtype=cdtype)
        vals_ell = (X_ell @ spec_ell.Dprime).real
        term2_h = term2_h + vals_ell.mean(dim=0)

    grad_g = 0.5 * (term1_g - term2_g)
    grad_h = 0.5 * (term1_h - term2_h)

    return {
        "grad_g": grad_g,
        "grad_h": grad_h,
        "cg_iters": torch.tensor(float(total_cg)),
    }


# ---------------------------------------------------------------------------
# Coupled M-step gradient via block CD
# ---------------------------------------------------------------------------

def _compute_blockcd_mstep_gradient_coupled(
    kappa: torch.Tensor,
    delta: torch.Tensor,
    spec_g: _SpectralState,
    specs_h: list[_SpectralState],
    loc_indices: list[torch.Tensor],
    *,
    cd_sweeps: int,
    n_probes: int,
    cg_tol: float,
    seed: int | None,
    x_g_warm: torch.Tensor | None = None,
    x_h_warm: list[torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """
    Compute kernel hyperparameter gradients using block CD for the coupled system.

    Term 1 (data-fit) uses pre-computed feature-space coefficients from E-step
    if x_g_warm/x_h_warm are provided, otherwise re-solves via block CD.

    Term 2 (trace) uses block CD probe solves.
    """
    n = kappa.numel()
    L = len(specs_h)
    m = spec_g.ws.numel()
    cdtype = spec_g.ws.dtype
    rdtype = kappa.dtype
    device = kappa.device
    total_cg = 0

    # --- Term 1: data-fit from E-step coefficients ---
    if x_g_warm is not None and x_h_warm is not None:
        # Use the E-step's converged feature-space coefficients directly
        beta_g = x_g_warm / spec_g.ws
        betas_h = [x_h_warm[ell] / specs_h[ell].ws for ell in range(L)]
    else:
        # Re-solve via block CD
        sol_kappa = _blockcd_feature_solve(
            kappa, delta, spec_g, specs_h, loc_indices,
            cd_sweeps=cd_sweeps, cg_tol=cg_tol,
        )
        beta_g = sol_kappa['x_g'] / spec_g.ws
        betas_h = [sol_kappa['x_h'][ell] / specs_h[ell].ws for ell in range(L)]
        total_cg += sol_kappa['cg_iters']

    abs2_g = (beta_g.conj() * beta_g).real
    n_hypers = spec_g.Dprime.shape[1]
    term1_g = (spec_g.Dprime.real.T @ abs2_g)

    term1_h = torch.zeros(n_hypers, dtype=rdtype, device=device)
    for ell in range(L):
        abs2_ell = (betas_h[ell].conj() * betas_h[ell]).real
        term1_h = term1_h + (specs_h[ell].Dprime.real.T @ abs2_ell)

    # --- Term 2: stochastic trace via Hutchinson probes ---
    probes = _sample_rademacher(
        (n_probes, n), device=device, dtype=rdtype,
        seed=None if seed is None else seed + 10_000,
    )

    # Solve the coupled system for all probes via block CD
    sol_probes = _blockcd_feature_solve(
        probes, delta, spec_g, specs_h, loc_indices,
        cd_sweeps=cd_sweeps, cg_tol=cg_tol,
    )
    x_g_probes = sol_probes['x_g']  # (J, m)
    x_h_probes = sol_probes['x_h']  # list of (J, m)
    total_cg += sol_probes['cg_iters']

    # Un-symmetrize probe solutions
    beta_probes_g = x_g_probes / spec_g.ws[None, :]
    beta_probes_h = [x_h_probes[ell] / specs_h[ell].ws[None, :] for ell in range(L)]

    # Compute Rfeat = F^*(Delta * z_j) for each probe
    probes_c = probes.to(dtype=cdtype)
    delta_c = delta.to(dtype=cdtype)
    omega_probes = delta_c[None, :] * probes_c

    # Global Rfeat
    Rfeat_g = torch.zeros(n_probes, m, dtype=cdtype, device=device)
    for j in range(n_probes):
        Rfeat_g[j] = spec_g.nufft_op.type1(
            omega_probes[j], out_shape=spec_g.out_shape
        ).reshape(-1)

    X_g = Rfeat_g.conj() * beta_probes_g.to(dtype=cdtype)
    vals_g = (X_g @ spec_g.Dprime).real
    term2_g = vals_g.mean(dim=0)

    # Local Rfeat and trace
    term2_h = torch.zeros(n_hypers, dtype=rdtype, device=device)
    for ell, idx in enumerate(loc_indices):
        spec_ell = specs_h[ell]
        m_ell = spec_ell.ws.numel()
        Rfeat_ell = torch.zeros(n_probes, m_ell, dtype=cdtype, device=device)
        for j in range(n_probes):
            Rfeat_ell[j] = spec_ell.nufft_op.type1(
                omega_probes[j, idx], out_shape=spec_ell.out_shape
            ).reshape(-1)

        X_ell = Rfeat_ell.conj() * beta_probes_h[ell].to(dtype=cdtype)
        vals_ell = (X_ell @ spec_ell.Dprime).real
        term2_h = term2_h + vals_ell.mean(dim=0)

    grad_g = 0.5 * (term1_g - term2_g)
    grad_h = 0.5 * (term1_h - term2_h)

    return {
        "grad_g": grad_g,
        "grad_h": grad_h,
        "cg_iters": torch.tensor(float(total_cg)),
    }


# ---------------------------------------------------------------------------
# Block coordinate descent E-step
# ---------------------------------------------------------------------------

def _run_blockcd_estep(
    targets: torch.Tensor,
    kappa: torch.Tensor,
    pg_b: torch.Tensor,
    delta: torch.Tensor,
    spec_g: _SpectralState,
    specs_h: list[_SpectralState],
    loc_indices: list[torch.Tensor],
    *,
    max_iters: int,
    cd_sweeps: int,
    rho0: float,
    gamma: float,
    tol: float,
    n_probes: int,
    cg_tol: float,
    seed: int | None,
    verbose: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    E-step via block coordinate descent.

    Each outer iteration:
      1. Update delta from current mean/var
      2. Run cd_sweeps of: solve g | {h_ell}, then solve each h_ell | g
      3. Estimate sigma_diag via Hutchinson on the final sweep
    """
    n = targets.numel()
    L = len(specs_h)
    rdtype = targets.dtype
    device = targets.device

    # Initialize component estimates
    g_mean = torch.zeros(n, dtype=rdtype, device=device)
    h_means = [torch.zeros(idx.numel(), dtype=rdtype, device=device) for idx in loc_indices]
    cdtype = spec_g.ws.dtype
    x_g = torch.zeros(spec_g.ws.numel(), dtype=cdtype, device=device)
    x_h_list = [torch.zeros(specs_h[ell].ws.numel(), dtype=cdtype, device=device) for ell in range(L)]

    residual = float("inf")
    total_cg = 0

    with torch.no_grad():
        for it in range(max_iters):
            # Block CD sweeps
            for sweep in range(cd_sweeps):
                # --- Step 1: Solve for g, fixing all h_ell ---
                # Pseudo-observations for g: kappa - delta * h_combined
                h_combined = torch.zeros(n, dtype=rdtype, device=device)
                for ell, idx in enumerate(loc_indices):
                    h_combined[idx] = h_means[ell]

                z_g = kappa - delta * h_combined
                g_mean, cg_g, x_g = _solve_single_gp(z_g, delta, spec_g, cg_tol=cg_tol)
                total_cg += cg_g

                # --- Step 2: Solve each h_ell, fixing g ---
                for ell, idx in enumerate(loc_indices):
                    z_ell = kappa[idx] - delta[idx] * g_mean[idx]
                    h_means[ell], cg_ell, x_h_ell = _solve_single_gp(
                        z_ell, delta[idx], specs_h[ell], cg_tol=cg_tol,
                    )
                    x_h_list[ell] = x_h_ell
                    total_cg += cg_ell

            # Compute full posterior mean
            mean = g_mean.clone()
            for ell, idx in enumerate(loc_indices):
                mean[idx] = mean[idx] + h_means[ell]

            # Estimate sigma_diag via Hutchinson with batched independent solves
            # (ignores cross-covariance, slightly overestimates variance, but cheap)
            probe_seed = None if seed is None else seed + 17 * (it + 1)
            probes = _sample_rademacher(
                (n_probes, n), device=device, dtype=rdtype, seed=probe_seed,
            )
            # g-component: batched solve for all probes at once
            _, sig_g, cg_g = _solve_single_gp_with_probes(
                torch.zeros(n, dtype=rdtype, device=device), probes,
                delta, spec_g, cg_tol=cg_tol,
            )
            total_cg += cg_g
            sigma_diag = sig_g
            # h-components: batched solve per location
            for ell, idx in enumerate(loc_indices):
                _, sig_h, cg_h = _solve_single_gp_with_probes(
                    torch.zeros(idx.numel(), dtype=rdtype, device=device),
                    probes[:, idx], delta[idx], specs_h[ell], cg_tol=cg_tol,
                )
                total_cg += cg_h
                sigma_diag[idx] = sigma_diag[idx] + sig_h

            # Update delta
            c2 = (sigma_diag + mean.pow(2)).clamp_min(1e-12)
            c = torch.sqrt(c2)
            Lambda = _pg_omega_expectation(c, pg_b)

            rho = rho0 / (1.0 + gamma * it)
            delta = delta * (1.0 - rho) + rho * Lambda
            delta = delta.clamp(min=0.0)

            residual = float((delta - Lambda).abs().max().item())
            if verbose > 1:
                print(f"  E-step it {it:3d} rho={rho:.3f} max|Delta-Lambda|={residual:.3e} cg={total_cg}")
            if residual < tol:
                break

    return delta, mean, sigma_diag, {
        "residual": residual,
        "cg_iters": total_cg,
        "x_g": x_g,
        "x_h": x_h_list,
    }


# ---------------------------------------------------------------------------
# M-step gradient (recomputed via block CD structure)
# ---------------------------------------------------------------------------

def _compute_blockcd_mstep_gradient(
    kappa: torch.Tensor,
    delta: torch.Tensor,
    spec_g: _SpectralState,
    specs_h: list[_SpectralState],
    loc_indices: list[torch.Tensor],
    *,
    n_probes: int,
    cg_tol: float,
    seed: int | None,
) -> dict[str, torch.Tensor]:
    """
    Compute kernel hyperparameter gradients.

    Uses the feature-space identity:
        dL/dtheta = 0.5 * beta^T (dD/dtheta) beta
                  - 0.5 * Tr(S_feat * (dD/dtheta))
    for each kernel independently.
    """
    n = kappa.numel()
    L = len(specs_h)
    cdtype = spec_g.ws.dtype
    rdtype = kappa.dtype
    device = kappa.device

    # Build weighted Toeplitz for global
    delta_c_g = delta.to(dtype=cdtype)
    conv_shape = tuple(2 * n - 1 for n in spec_g.out_shape)
    v_g = spec_g.nufft_op.type1(delta_c_g, out_shape=conv_shape)
    T_g = ToeplitzND(v_g.to(dtype=cdtype), force_pow2=True)

    def A_feat_g(u):
        return u + spec_g.ws * T_g(spec_g.ws * u)

    # Solve for global coefficients
    rhs_g = spec_g.ws * spec_g.fadj(kappa.to(dtype=cdtype))
    cg_g = ConjugateGradients(A_feat_g, rhs_g, x0=torch.zeros_like(rhs_g),
                               tol=cg_tol, max_iter=2000, early_stopping=True)
    beta_g = cg_g.solve()
    total_cg = cg_g.iters_completed

    # Term 1 for global: beta_g^T (Dprime_g / ws2_g) beta_g
    # Dprime has shape (m, n_hypers)
    m = spec_g.ws.numel()
    n_hypers_g = spec_g.Dprime.shape[1]
    grad1_g = torch.zeros(n_hypers_g, dtype=cdtype, device=device)
    for j in range(n_hypers_g):
        dprime_j = spec_g.Dprime[:, j]
        ws2_safe = spec_g.ws2.real.clamp_min(1e-30).to(dtype=cdtype)
        grad1_g[j] = 0.5 * (beta_g.conj() * dprime_j / ws2_safe * beta_g).sum()

    # Term 2 for global: Hutchinson trace
    rng = None if seed is None else torch.Generator(device=device).manual_seed(seed)
    probes_feat = _sample_rademacher((n_probes, m), device=device, dtype=rdtype, seed=seed)
    probes_feat = probes_feat.to(dtype=cdtype)

    cg_probes = ConjugateGradients(A_feat_g, probes_feat, x0=torch.zeros_like(probes_feat),
                                    tol=cg_tol, max_iter=2000, early_stopping=True)
    S_feat_probes = cg_probes.solve()
    total_cg += cg_probes.iters_completed

    grad2_g = torch.zeros(n_hypers_g, dtype=cdtype, device=device)
    for j in range(n_hypers_g):
        dprime_j = spec_g.Dprime[:, j]
        dD_j = dprime_j / spec_g.ws2.real.clamp_min(1e-30).to(dtype=cdtype)
        grad2_g[j] = -0.5 * (probes_feat * (dD_j[None, :] * S_feat_probes)).mean(dim=0).sum()

    grad_g = (grad1_g + grad2_g)

    # Local kernel gradient: sum over all locations
    n_hypers_h = specs_h[0].Dprime.shape[1]
    grad1_h = torch.zeros(n_hypers_h, dtype=cdtype, device=device)
    grad2_h = torch.zeros(n_hypers_h, dtype=cdtype, device=device)

    for ell, idx in enumerate(loc_indices):
        spec_ell = specs_h[ell]
        delta_ell = delta[idx].to(dtype=cdtype)

        conv_shape_ell = tuple(2 * n - 1 for n in spec_ell.out_shape)
        v_ell = spec_ell.nufft_op.type1(delta_ell, out_shape=conv_shape_ell)
        T_ell = ToeplitzND(v_ell.to(dtype=cdtype), force_pow2=True)

        def A_feat_ell(u, T=T_ell, s=spec_ell):
            return u + s.ws * T(s.ws * u)

        rhs_ell = spec_ell.ws * spec_ell.fadj(kappa[idx].to(dtype=cdtype))
        cg_ell = ConjugateGradients(A_feat_ell, rhs_ell, x0=torch.zeros_like(rhs_ell),
                                     tol=cg_tol, max_iter=2000, early_stopping=True)
        beta_ell = cg_ell.solve()
        total_cg += cg_ell.iters_completed

        for j in range(n_hypers_h):
            dprime_j = spec_ell.Dprime[:, j]
            grad1_h[j] += 0.5 * (beta_ell.conj() * dprime_j / spec_ell.ws2.real.clamp_min(1e-30).to(dtype=cdtype) * beta_ell).sum()

        m_ell = spec_ell.ws.numel()
        probes_ell = _sample_rademacher((n_probes, m_ell), device=device, dtype=rdtype,
                                         seed=None if seed is None else seed + 100 + ell)
        probes_ell = probes_ell.to(dtype=cdtype)
        cg_p = ConjugateGradients(A_feat_ell, probes_ell, x0=torch.zeros_like(probes_ell),
                                   tol=cg_tol, max_iter=2000, early_stopping=True)
        S_p = cg_p.solve()
        total_cg += cg_p.iters_completed

        for j in range(n_hypers_h):
            dprime_j = spec_ell.Dprime[:, j]
            dD_j = dprime_j / spec_ell.ws2.real.clamp_min(1e-30).to(dtype=cdtype)
            grad2_h[j] += -0.5 * (probes_ell * (dD_j[None, :] * S_p)).mean(dim=0).sum()

    grad_h = (grad1_h + grad2_h)

    return {
        "grad_g": grad_g,
        "grad_h": grad_h,
        "cg_iters": torch.tensor(total_cg),
    }


# ---------------------------------------------------------------------------
# Main regressor class
# ---------------------------------------------------------------------------

class HierarchicalPGNegBinRegressorBlockCD:
    """
    Hierarchical additive-kernel PG-augmented GP for NB counts.
    Uses block coordinate descent instead of a coupled (1+L)*m CG system.
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
        cd_sweeps: int = 3,
        m_cd_sweeps: int | None = None,
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
        self.cd_sweeps = cd_sweeps
        self.m_cd_sweeps = m_cd_sweeps if m_cd_sweeps is not None else cd_sweeps
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

    def _build_spectral_states(self, X_t, loc_t, kernel_g, kernel_h):
        """Build spectral states for global and per-location GPs."""
        rdtype = self.dtype
        cdtype = torch.complex128 if rdtype == torch.float64 else torch.complex64
        dev = self.device
        d = X_t.shape[1]

        x0 = X_t.min(dim=0).values
        x1 = X_t.max(dim=0).values
        L_domain = (x1 - x0).max()

        # Get shared grid (use the finer of the two kernels)
        xis_g_1d, h_g, mtot_g = get_xis(kernel_g, eps=self.spectral_eps, L=L_domain,
                                          use_integral=True, l2scaled=False, trunc_eps=self.trunc_eps)
        xis_h_1d, h_h, mtot_h = get_xis(kernel_h, eps=self.spectral_eps, L=L_domain,
                                          use_integral=True, l2scaled=False, trunc_eps=self.trunc_eps)
        if mtot_h >= mtot_g:
            xis_1d, h, mtot = xis_h_1d, h_h, mtot_h
        else:
            xis_1d, h, mtot = xis_g_1d, h_g, mtot_g

        # Global spectral state (all n points)
        spec_g = _build_local_spectral_state(
            X_t, kernel_g, xis_1d=xis_1d, h=h, mtot=mtot,
            spectral_eps=self.spectral_eps, nufft_eps=self.nufft_eps,
            rdtype=rdtype, cdtype=cdtype, device=dev,
        )

        # Per-location spectral states
        unique_locs = torch.unique(loc_t)
        loc_indices = []
        specs_h = []
        for loc in unique_locs:
            idx = torch.where(loc_t == loc)[0]
            loc_indices.append(idx)
            spec_ell = _build_local_spectral_state(
                X_t[idx], kernel_h, xis_1d=xis_1d, h=h, mtot=mtot,
                spectral_eps=self.spectral_eps, nufft_eps=self.nufft_eps,
                rdtype=rdtype, cdtype=cdtype, device=dev,
            )
            specs_h.append(spec_ell)

        return spec_g, specs_h, loc_indices

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        locations: np.ndarray,
    ) -> "HierarchicalPGNegBinRegressorBlockCD":
        rdtype = self.dtype
        dev = self.device

        X_t = torch.as_tensor(X, device=dev, dtype=rdtype)
        if X_t.ndim == 1:
            X_t = X_t.unsqueeze(-1)
        y_t = torch.as_tensor(y, device=dev, dtype=rdtype)
        loc_t = torch.as_tensor(locations, device=dev, dtype=torch.long)
        n = X_t.shape[0]
        d = X_t.shape[1]

        self.kernel_g_ = _make_kernel(dimension=d,
                                       lengthscale=self.lengthscale_g_init,
                                       variance=self.variance_g_init)
        self.kernel_h_ = _make_kernel(dimension=d,
                                       lengthscale=self.lengthscale_h_init,
                                       variance=self.variance_h_init)

        r = self.total_count_init
        if self.learn_total_count:
            log_r = torch.nn.Parameter(
                torch.tensor(math.log(r), device=dev, dtype=rdtype)
            )
            r_optimizer = torch.optim.Adam([log_r], lr=self.total_count_lr, maximize=True)

        opt_g = torch.optim.Adam(self.kernel_g_._gp_params_ref.parameters(),
                                  lr=self.lr_g, maximize=True)
        opt_h = torch.optim.Adam(self.kernel_h_._gp_params_ref.parameters(),
                                  lr=self.lr_h, maximize=True)

        def _kappa():
            return 0.5 * (y_t - r)

        def _pg_b():
            return y_t + r

        delta = (0.25 * _pg_b()).clone()

        history: list[dict] = []
        t0 = time.perf_counter()

        for outer in range(self.max_iter):
            if self.learn_total_count:
                r = float(torch.exp(log_r).item())
            kappa = _kappa()
            pg_b = _pg_b()

            # Build spectral states
            spec_g, specs_h, loc_indices = self._build_spectral_states(
                X_t, loc_t, self.kernel_g_, self.kernel_h_,
            )

            # E-step via block CD
            delta, mean, sigma_diag, estep_info = _run_blockcd_estep(
                y_t, kappa, pg_b, delta, spec_g, specs_h, loc_indices,
                max_iters=self.e_step_iters,
                cd_sweeps=self.cd_sweeps,
                rho0=self.rho0, gamma=self.e_gamma,
                tol=self.e_step_tol,
                n_probes=self.n_e_probes,
                cg_tol=self.cg_tol,
                seed=None if self.seed is None else self.seed + 1000 * outer,
                verbose=self.verbose,
            )

            # M-step: Schur complement for correct gradients
            # Uses E-step coefficients for term1, Schur solve for term2
            mstep_out = _compute_schur_mstep_gradient(
                kappa, delta, spec_g, specs_h, loc_indices,
                n_probes=self.n_m_probes,
                cg_tol=self.cg_tol,
                seed=None if self.seed is None else self.seed + 1000 * outer + 500,
                x_g_warm=estep_info['x_g'],
                x_h_warm=estep_info['x_h'],
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
        spec_g, specs_h, loc_indices = self._build_spectral_states(
            X_t, loc_t, self.kernel_g_, self.kernel_h_,
        )
        delta, mean, sigma_diag, _ = _run_blockcd_estep(
            y_t, _kappa(), _pg_b(), delta, spec_g, specs_h, loc_indices,
            max_iters=self.final_e_step_iters,
            cd_sweeps=self.cd_sweeps + 2,
            rho0=self.rho0, gamma=self.e_gamma,
            tol=self.e_step_tol,
            n_probes=self.n_e_probes,
            cg_tol=self.cg_tol,
            seed=None if self.seed is None else self.seed + 999_999,
            verbose=self.verbose,
        )

        # Decompose into g and h components via one more CD sweep
        h_combined = torch.zeros_like(mean)
        # First pass: solve each h given current g estimate
        g_hat = torch.zeros_like(mean)  # start from zero, will re-solve
        for ell, idx in enumerate(loc_indices):
            z_ell = kappa[idx] - delta[idx] * g_hat[idx]
            h_ell, _, _ = _solve_single_gp(z_ell, delta[idx], specs_h[ell], cg_tol=self.cg_tol)
            h_combined[idx] = h_ell

        # Re-solve g given h
        z_g = kappa - delta * h_combined
        g_hat, _, _ = _solve_single_gp(z_g, delta, spec_g, cg_tol=self.cg_tol)

        # One more sweep to refine
        h_combined = torch.zeros_like(mean)
        for ell, idx in enumerate(loc_indices):
            z_ell = kappa[idx] - delta[idx] * g_hat[idx]
            h_ell, _, _ = _solve_single_gp(z_ell, delta[idx], specs_h[ell], cg_tol=self.cg_tol)
            h_combined[idx] = h_ell
        z_g = kappa - delta * h_combined
        g_hat, _, _ = _solve_single_gp(z_g, delta, spec_g, cg_tol=self.cg_tol)

        self.history_ = history
        self.delta_ = delta.detach().cpu().numpy()
        self.mean_ = mean.detach().cpu().numpy()
        self.sigma_diag_ = sigma_diag.detach().cpu().numpy()
        self.total_count_ = r
        self.lengthscale_g_ = float(self.kernel_g_.lengthscale)
        self.variance_g_ = float(self.kernel_g_.variance)
        self.lengthscale_h_ = float(self.kernel_h_.lengthscale)
        self.variance_h_ = float(self.kernel_h_.variance)
        self.g_hat_ = g_hat.detach().cpu().numpy()
        self.h_hats_ = {}
        for ell, idx in enumerate(loc_indices):
            self.h_hats_[ell] = h_combined[idx].detach().cpu().numpy()

        return self

    def _extract_g(self, mean, loc_indices, specs_h, delta, kappa):
        """Helper to extract g component from current mean."""
        # Approximate: g ≈ mean - h_combined, but we need to solve for it
        # This is a placeholder; the actual decomposition is done in the final step
        return torch.zeros_like(mean)
