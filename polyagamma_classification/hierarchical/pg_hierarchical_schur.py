"""
Hierarchical additive-kernel PG-augmented GP for negative binomial counts.
Two Schur-complement-based approaches that minimize NUFFT usage.

Approach C: HierarchicalPGNegBinRegressorSchurCG
    - E-step: Single Schur CG solve per inner iteration (feature-space only)
    - M-step: Feature-space Hutchinson probes (ZERO NUFFTs)
    - Generalizes to any d

Approach D: HierarchicalPGNegBinRegressorDirect
    - E-step: Dense Cholesky factorization + direct solve
    - M-step: Exact trace via block-inverse formulas (ZERO NUFFTs)
    - d=1 specific (m small)

Model:
    f_i = g(t_i) + h_{ell_i}(t_i)
    g ~ GP(0, k_g),  h_ell ~iid GP(0, k_h)
    y_i | f_i, r ~ NB(r, sigma(f_i))
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.fft import fftn, ifftn

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_PG_ROOT = Path(__file__).resolve().parents[1]
if str(_PG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PG_ROOT))

from cg import ConjugateGradients
from efgpnd import NUFFT, ToeplitzND
from kernels import SquaredExponential

from pg_classifier import (
    _pg_omega_expectation,
    _sample_rademacher,
    _negative_binomial_total_count_gradient,
    _SpectralState,
)

from hierarchical.pg_hierarchical_blockcd import (
    _build_local_spectral_state,
    _solve_single_gp,
    _solve_single_gp_with_probes,
    _schur_solve,
    _make_kernel,
)


# ---------------------------------------------------------------------------
# Approach C: Schur CG (no matrix factorization)
# ---------------------------------------------------------------------------

def _build_toeplitz_operators(
    delta: torch.Tensor,
    spec_g: _SpectralState,
    specs_h: list[_SpectralState],
    loc_indices: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, ToeplitzND, list[ToeplitzND]]:
    """
    Build all Toeplitz operators and diagonal half-weights.
    Costs (1+L) NUFFTs.

    Returns (Dg_half, Dh_half, T_all, T_locs).
    """
    cdtype = spec_g.ws.dtype
    L = len(specs_h)

    Dg_half = spec_g.ws
    Dh_half = specs_h[0].ws  # all locations share same kernel

    delta_c = delta.to(dtype=cdtype)
    conv_shape = tuple(2 * s - 1 for s in spec_g.out_shape)
    v_all = spec_g.nufft_op.type1(delta_c, out_shape=conv_shape)
    T_all = ToeplitzND(v_all.to(dtype=cdtype), force_pow2=False)

    T_locs = []
    for ell, idx in enumerate(loc_indices):
        spec_ell = specs_h[ell]
        delta_ell = delta[idx].to(dtype=cdtype)
        conv_shape_ell = tuple(2 * s - 1 for s in spec_ell.out_shape)
        v_ell = spec_ell.nufft_op.type1(delta_ell, out_shape=conv_shape_ell)
        T_locs.append(ToeplitzND(v_ell.to(dtype=cdtype), force_pow2=False))

    return Dg_half, Dh_half, T_all, T_locs


def _transform_kappa_to_feature(
    kappa: torch.Tensor,
    spec_g: _SpectralState,
    specs_h: list[_SpectralState],
    loc_indices: list[torch.Tensor],
    Dg_half: torch.Tensor,
    Dh_half: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    Transform kappa to feature-space RHS: Dg^{1/2} F^* kappa, Dh^{1/2} F_ell^* kappa_ell.
    Costs (1+L) NUFFTs.
    """
    cdtype = spec_g.ws.dtype

    rhs_g = Dg_half * spec_g.fadj(kappa.to(dtype=cdtype))
    rhs_h = []
    for ell, idx in enumerate(loc_indices):
        rhs_h.append(Dh_half * specs_h[ell].fadj(kappa[idx].to(dtype=cdtype)))

    return rhs_g, rhs_h


def _transform_mean_to_obs(
    a_g: torch.Tensor,
    a_h: list[torch.Tensor],
    spec_g: _SpectralState,
    specs_h: list[_SpectralState],
    loc_indices: list[torch.Tensor],
    Dg_half: torch.Tensor,
    Dh_half: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """
    Transform feature-space solution back to observation space.
    Costs (1+L) NUFFTs.

    Returns (g_mean, h_means, full_mean).
    """
    device = a_g.device

    g_mean = spec_g.fwd(Dg_half * a_g).real
    h_means = []
    full_mean = g_mean.clone()
    for ell, idx in enumerate(loc_indices):
        h_ell = specs_h[ell].fwd(Dh_half * a_h[ell]).real
        h_means.append(h_ell)
        full_mean[idx] = full_mean[idx] + h_ell

    return g_mean, h_means, full_mean


def _run_schur_cg_estep(
    targets: torch.Tensor,
    kappa: torch.Tensor,
    pg_b: torch.Tensor,
    delta: torch.Tensor,
    spec_g: _SpectralState,
    specs_h: list[_SpectralState],
    loc_indices: list[torch.Tensor],
    *,
    max_iters: int,
    rho0: float,
    gamma: float,
    tol: float,
    n_probes: int,
    cg_tol: float,
    inner_cg_tol: float,
    seed: int | None,
    verbose: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    E-step via Schur CG solve.

    Each inner iteration:
      1. Build Toeplitz operators: (1+L) NUFFTs
      2. Transform kappa to feature space: (1+L) NUFFTs
      3. Schur CG solve in feature space: 0 NUFFTs
      4. Transform mean back: (1+L) NUFFTs
      5. sigma_diag via independent batched CG: 2(1+L) NUFFTs
      Total: 5(1+L) NUFFTs per inner iteration
    """
    n = targets.numel()
    L = len(specs_h)
    rdtype = targets.dtype
    device = targets.device
    cdtype = spec_g.ws.dtype

    residual = float("inf")
    total_cg = 0

    a_g = None
    a_h = None

    with torch.no_grad():
        for it in range(max_iters):
            # 1. Build Toeplitz operators
            Dg_half, Dh_half, T_all, T_locs = _build_toeplitz_operators(
                delta, spec_g, specs_h, loc_indices,
            )

            # 2. Transform kappa to feature space
            rhs_g, rhs_h = _transform_kappa_to_feature(
                kappa, spec_g, specs_h, loc_indices, Dg_half, Dh_half,
            )

            # 3. Schur CG solve in feature space (0 NUFFTs)
            a_g, a_h, cg_iters = _schur_solve(
                rhs_g, rhs_h,
                Dg_half=Dg_half, Dh_half=Dh_half,
                T_all=T_all, T_locs=T_locs,
                cg_tol=cg_tol, inner_cg_tol=inner_cg_tol,
            )
            total_cg += cg_iters

            # 4. Transform mean back to observation space
            g_mean, h_means, mean = _transform_mean_to_obs(
                a_g, a_h, spec_g, specs_h, loc_indices, Dg_half, Dh_half,
            )

            # 5. sigma_diag via independent batched CG (ignoring cross-covariance)
            probe_seed = None if seed is None else seed + 17 * (it + 1)
            probes = _sample_rademacher(
                (n_probes, n), device=device, dtype=rdtype, seed=probe_seed,
            )
            _, sig_g, cg_g = _solve_single_gp_with_probes(
                torch.zeros(n, dtype=rdtype, device=device), probes,
                delta, spec_g, cg_tol=cg_tol,
            )
            total_cg += cg_g
            sigma_diag = sig_g
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
        "a_g": a_g,
        "a_h": a_h,
    }


def _compute_feature_space_mstep_gradient(
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
    a_g_warm: torch.Tensor | None = None,
    a_h_warm: list[torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """
    Compute M-step gradients entirely in feature space (ZERO NUFFTs).

    Term 1 (data-fit): from E-step coefficients (pre-computed).
    Term 2 (trace): feature-space Hutchinson probes.
      - Sample J probes z_j in R^{(1+L)m}
      - Solve y_j = M^{-1} z_j via _schur_solve
      - For each hyperparameter theta, compute v_j = (dM/dtheta) z_j
      - Accumulate: tr(M^{-1} dM/dtheta) ~ (1/J) sum y_j^T v_j
    """
    L = len(specs_h)
    m = spec_g.ws.numel()
    cdtype = spec_g.ws.dtype
    rdtype = kappa.dtype
    device = kappa.device
    if inner_cg_tol is None:
        inner_cg_tol = cg_tol

    # Build Toeplitz operators (reuse from E-step in practice, but we rebuild here)
    Dg_half, Dh_half, T_all, T_locs = _build_toeplitz_operators(
        delta, spec_g, specs_h, loc_indices,
    )

    total_cg = 0

    # --- Term 1: data-fit from E-step coefficients ---
    if a_g_warm is not None and a_h_warm is not None:
        beta_g = a_g_warm / Dg_half
        betas_h = [a_h_warm[ell] / Dh_half for ell in range(L)]
    else:
        # Fall back: solve via Schur for kappa
        rhs_g_kappa = Dg_half * spec_g.fadj(kappa.to(dtype=cdtype))
        rhs_h_kappa = []
        for ell, idx in enumerate(loc_indices):
            rhs_h_kappa.append(Dh_half * specs_h[ell].fadj(kappa[idx].to(dtype=cdtype)))
        a_g, a_h, cg_k = _schur_solve(
            rhs_g_kappa, rhs_h_kappa,
            Dg_half=Dg_half, Dh_half=Dh_half,
            T_all=T_all, T_locs=T_locs,
            cg_tol=cg_tol, inner_cg_tol=inner_cg_tol,
        )
        total_cg += cg_k
        beta_g = a_g / Dg_half
        betas_h = [a_h[ell] / Dh_half for ell in range(L)]

    abs2_g = (beta_g.conj() * beta_g).real
    n_hypers = spec_g.Dprime.shape[1]
    term1_g = spec_g.Dprime.real.T @ abs2_g

    term1_h = torch.zeros(n_hypers, dtype=rdtype, device=device)
    for ell in range(L):
        abs2_ell = (betas_h[ell].conj() * betas_h[ell]).real
        term1_h = term1_h + specs_h[ell].Dprime.real.T @ abs2_ell

    # --- Term 2: feature-space Hutchinson probes (ZERO NUFFTs) ---
    # Sample probes in R^{(1+L)*m} feature space
    probe_seed_g = None if seed is None else seed + 20_000
    probes_g = _sample_rademacher(
        (n_probes, m), device=device, dtype=rdtype, seed=probe_seed_g,
    ).to(dtype=cdtype)

    probes_h = []
    for ell in range(L):
        ps = None if seed is None else seed + 20_000 + ell + 1
        p_ell = _sample_rademacher(
            (n_probes, m), device=device, dtype=rdtype, seed=ps,
        ).to(dtype=cdtype)
        probes_h.append(p_ell)

    # Solve M^{-1} z for all probes via Schur complement
    probes_h_list = [probes_h[ell] for ell in range(L)]
    y_g, y_h, cg_p = _schur_solve(
        probes_g, probes_h_list,
        Dg_half=Dg_half, Dh_half=Dh_half,
        T_all=T_all, T_locs=T_locs,
        cg_tol=cg_tol, inner_cg_tol=inner_cg_tol,
    )
    total_cg += cg_p

    # Compute tr(M^{-1} dM/dtheta_g) for each global hyperparameter
    # dM/dtheta_g only has a nonzero g-block: dA/dtheta_g
    # dA/dtheta_g = P_theta T_all Dg^{1/2} + Dg^{1/2} T_all P_theta
    # where P_theta = diag(Dprime[:,j] / (2 * ws_g))
    # (dM/dtheta_g) z has g-component: (dA/dtheta_g) z_g, h-components: 0

    ws_g = spec_g.ws
    ws2_g = spec_g.ws2

    term2_g = torch.zeros(n_hypers, dtype=rdtype, device=device)
    for j in range(n_hypers):
        # P_theta = diag(Dprime[:,j] / (2 * ws_g))
        P_theta = spec_g.Dprime[:, j] / (2.0 * ws_g)

        # (dA/dtheta_g) z_g = P_theta * T_all(Dg^{1/2} * z_g) + Dg^{1/2} * T_all(P_theta * z_g)
        for probe_idx in range(n_probes):
            z_g = probes_g[probe_idx]
            dA_z = P_theta * T_all(Dg_half * z_g) + Dg_half * T_all(P_theta * z_g)
            # y_j^T (dM/dtheta) z_j = y_g_j^T (dA/dtheta) z_g_j (h components are zero)
            val = (y_g[probe_idx].conj() * dA_z).sum()
            term2_g[j] = term2_g[j] + val.real / n_probes

    # Compute tr(M^{-1} dM/dtheta_h) for each local hyperparameter
    # dM/dtheta_h has:
    #   g-block: sum_ell dB_ell/dtheta_h * z_h_ell  (off-diagonal)
    #   h_ell-block: dD_ell/dtheta_h * z_h_ell + dB_ell^*/dtheta_h * z_g
    # dB_ell/dtheta_h = Dg^{1/2} T_ell Q_theta
    # dD_ell/dtheta_h = Q_theta T_ell Dh^{1/2} + Dh^{1/2} T_ell Q_theta
    # dB_ell^*/dtheta_h = Q_theta T_ell Dg^{1/2}

    ws_h = specs_h[0].ws

    term2_h = torch.zeros(n_hypers, dtype=rdtype, device=device)
    for j in range(n_hypers):
        Q_theta = specs_h[0].Dprime[:, j] / (2.0 * ws_h)

        for probe_idx in range(n_probes):
            z_g = probes_g[probe_idx]
            y_g_j = y_g[probe_idx]
            val = torch.zeros(1, dtype=cdtype, device=device)

            for ell in range(L):
                z_h_ell = probes_h[ell][probe_idx]
                y_h_ell = y_h[ell][probe_idx]
                T_ell = T_locs[ell]

                # dB_ell/dtheta_h * z_h_ell = Dg^{1/2} * T_ell(Q_theta * z_h_ell)
                dB_z_h = Dg_half * T_ell(Q_theta * z_h_ell)

                # dD_ell/dtheta_h * z_h_ell = Q_theta * T_ell(Dh^{1/2} * z_h_ell)
                #                             + Dh^{1/2} * T_ell(Q_theta * z_h_ell)
                dD_z_h = Q_theta * T_ell(Dh_half * z_h_ell) + Dh_half * T_ell(Q_theta * z_h_ell)

                # dB_ell^*/dtheta_h * z_g = Q_theta * T_ell(Dg^{1/2} * z_g)
                dBstar_z_g = Q_theta * T_ell(Dg_half * z_g)

                # Contribution: y_g^T (dB z_h) + y_h^T (dD z_h + dB^* z_g)
                val = val + (y_g_j.conj() * dB_z_h).sum()
                val = val + (y_h_ell.conj() * dD_z_h).sum()
                val = val + (y_h_ell.conj() * dBstar_z_g).sum()

            term2_h[j] = term2_h[j] + val.real / n_probes

    grad_g = 0.5 * (term1_g - term2_g)
    grad_h = 0.5 * (term1_h - term2_h)

    return {
        "grad_g": grad_g,
        "grad_h": grad_h,
        "cg_iters": torch.tensor(float(total_cg)),
    }


class HierarchicalPGNegBinRegressorSchurCG:
    """
    Approach C: Schur CG solver.
    E-step uses a single Schur CG solve per inner iteration.
    M-step uses feature-space Hutchinson probes (ZERO NUFFTs).
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
        inner_cg_tol: float | None = None,
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
        self.inner_cg_tol = inner_cg_tol if inner_cg_tol is not None else cg_tol
        self.nufft_eps = nufft_eps
        self.spectral_eps = spectral_eps
        self.trunc_eps = trunc_eps
        self.seed = seed
        self.verbose = verbose
        self.dtype = dtype
        self.device = torch.device(device)

    def _build_spectral_states(self, X_t, loc_t, kernel_g, kernel_h):
        """Build spectral states for global and per-location GPs."""
        from utils.kernels import get_xis

        rdtype = self.dtype
        cdtype = torch.complex128 if rdtype == torch.float64 else torch.complex64
        dev = self.device
        d = X_t.shape[1]

        x0 = X_t.min(dim=0).values
        x1 = X_t.max(dim=0).values
        L_domain = (x1 - x0).max()

        xis_g_1d, h_g, mtot_g = get_xis(kernel_g, eps=self.spectral_eps, L=L_domain,
                                          use_integral=True, l2scaled=False, trunc_eps=self.trunc_eps)
        xis_h_1d, h_h, mtot_h = get_xis(kernel_h, eps=self.spectral_eps, L=L_domain,
                                          use_integral=True, l2scaled=False, trunc_eps=self.trunc_eps)
        if mtot_h >= mtot_g:
            xis_1d, h, mtot = xis_h_1d, h_h, mtot_h
        else:
            xis_1d, h, mtot = xis_g_1d, h_g, mtot_g

        spec_g = _build_local_spectral_state(
            X_t, kernel_g, xis_1d=xis_1d, h=h, mtot=mtot,
            spectral_eps=self.spectral_eps, nufft_eps=self.nufft_eps,
            rdtype=rdtype, cdtype=cdtype, device=dev,
        )

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
    ) -> "HierarchicalPGNegBinRegressorSchurCG":
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

            spec_g, specs_h, loc_indices = self._build_spectral_states(
                X_t, loc_t, self.kernel_g_, self.kernel_h_,
            )

            # E-step via Schur CG
            delta, mean, sigma_diag, estep_info = _run_schur_cg_estep(
                y_t, kappa, pg_b, delta, spec_g, specs_h, loc_indices,
                max_iters=self.e_step_iters,
                rho0=self.rho0, gamma=self.e_gamma,
                tol=self.e_step_tol,
                n_probes=self.n_e_probes,
                cg_tol=self.cg_tol,
                inner_cg_tol=self.inner_cg_tol,
                seed=None if self.seed is None else self.seed + 1000 * outer,
                verbose=self.verbose,
            )

            # M-step: feature-space gradients (ZERO NUFFTs)
            mstep_out = _compute_feature_space_mstep_gradient(
                kappa, delta, spec_g, specs_h, loc_indices,
                n_probes=self.n_m_probes,
                cg_tol=self.cg_tol,
                inner_cg_tol=self.inner_cg_tol,
                seed=None if self.seed is None else self.seed + 1000 * outer + 500,
                a_g_warm=estep_info['a_g'],
                a_h_warm=estep_info['a_h'],
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
                    f"e_cg={record['e_cg']:.0f} m_cg={record['m_cg']:.0f} "
                    f"t={record['elapsed']:.1f}s"
                )

        # Final E-step
        if self.learn_total_count:
            r = float(torch.exp(log_r).item())
        kappa = _kappa()
        pg_b = _pg_b()
        spec_g, specs_h, loc_indices = self._build_spectral_states(
            X_t, loc_t, self.kernel_g_, self.kernel_h_,
        )
        delta, mean, sigma_diag, final_info = _run_schur_cg_estep(
            y_t, kappa, pg_b, delta, spec_g, specs_h, loc_indices,
            max_iters=self.final_e_step_iters,
            rho0=self.rho0, gamma=self.e_gamma,
            tol=self.e_step_tol,
            n_probes=self.n_e_probes,
            cg_tol=self.cg_tol,
            inner_cg_tol=self.inner_cg_tol,
            seed=None if self.seed is None else self.seed + 999_999,
            verbose=self.verbose,
        )

        # Decompose into g and h via the final Schur solution
        Dg_half, Dh_half, T_all, T_locs = _build_toeplitz_operators(
            delta, spec_g, specs_h, loc_indices,
        )
        a_g = final_info['a_g']
        a_h = final_info['a_h']
        g_mean, h_means, _ = _transform_mean_to_obs(
            a_g, a_h, spec_g, specs_h, loc_indices, Dg_half, Dh_half,
        )

        self.history_ = history
        self.delta_ = delta.detach().cpu().numpy()
        self.mean_ = mean.detach().cpu().numpy()
        self.sigma_diag_ = sigma_diag.detach().cpu().numpy()
        self.total_count_ = r
        self.lengthscale_g_ = float(self.kernel_g_.lengthscale)
        self.variance_g_ = float(self.kernel_g_.variance)
        self.lengthscale_h_ = float(self.kernel_h_.lengthscale)
        self.variance_h_ = float(self.kernel_h_.variance)
        self.g_hat_ = g_mean.detach().cpu().numpy()
        self.h_hats_ = {}
        for ell, idx in enumerate(loc_indices):
            self.h_hats_[ell] = h_means[ell].detach().cpu().numpy()

        return self


# ---------------------------------------------------------------------------
# Approach D: Dense Cholesky factorization (d=1, m small)
# ---------------------------------------------------------------------------

def _form_dense_toeplitz(T_op: ToeplitzND, m: int, cdtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Form a dense m x m matrix from a Toeplitz operator by applying it to m basis vectors."""
    I = torch.eye(m, dtype=cdtype, device=device)
    # Apply T to each basis vector
    return torch.stack([T_op(I[i]) for i in range(m)], dim=1)  # (m, m)


def _build_schur_dense(
    Dg_half: torch.Tensor,
    Dh_half: torch.Tensor,
    T_all: ToeplitzND,
    T_locs: list[ToeplitzND],
    m: int,
    cdtype: torch.dtype,
    device: torch.device,
) -> dict:
    """
    Form all dense m x m blocks and Cholesky factorize.

    Returns dict with:
      A, D_ells, B_ells, S (dense matrices)
      L_S, L_D_ells (Cholesky factors)
      D_ell_invs (precomputed D_ell^{-1})
    """
    L = len(T_locs)

    # Form T matrices as dense
    T_all_dense = _form_dense_toeplitz(T_all, m, cdtype, device)
    T_loc_denses = [_form_dense_toeplitz(T_locs[ell], m, cdtype, device) for ell in range(L)]

    # A = I + Dg^{1/2} T_all Dg^{1/2}
    Dg_diag = torch.diag(Dg_half)
    A = torch.eye(m, dtype=cdtype, device=device) + Dg_diag @ T_all_dense @ Dg_diag

    # D_ell = I + Dh^{1/2} T_ell Dh^{1/2}
    Dh_diag = torch.diag(Dh_half)
    D_ells = []
    D_ell_invs = []
    for ell in range(L):
        D_ell = torch.eye(m, dtype=cdtype, device=device) + Dh_diag @ T_loc_denses[ell] @ Dh_diag
        # Symmetrize (should already be Hermitian up to numerics)
        D_ell = 0.5 * (D_ell + D_ell.conj().T)
        D_ells.append(D_ell)
        D_inv = torch.linalg.inv(D_ell)
        D_ell_invs.append(D_inv)

    # B_ell = Dg^{1/2} T_ell Dh^{1/2}
    B_ells = []
    for ell in range(L):
        B_ell = Dg_diag @ T_loc_denses[ell] @ Dh_diag
        B_ells.append(B_ell)

    # Schur complement S = A - sum_ell B_ell D_ell^{-1} B_ell^*
    S = A.clone()
    for ell in range(L):
        S = S - B_ells[ell] @ D_ell_invs[ell] @ B_ells[ell].conj().T

    # Invert S (small m x m, direct inversion is fine)
    S = 0.5 * (S + S.conj().T)
    S_inv = torch.linalg.inv(S)

    return {
        "A": A, "D_ells": D_ells, "B_ells": B_ells, "S": S,
        "D_ell_invs": D_ell_invs, "S_inv": S_inv,
        "T_all_dense": T_all_dense, "T_loc_denses": T_loc_denses,
    }


def _direct_schur_solve(
    rhs_g: torch.Tensor,
    rhs_h: list[torch.Tensor],
    blocks: dict,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    Direct solve M [a_g; a_h] = [rhs_g; rhs_h] via Cholesky.

    rhs_g: (m,) or (B, m)
    rhs_h: list of (m,) or (B, m)
    """
    vector_input = rhs_g.dim() == 1
    if vector_input:
        rhs_g = rhs_g.unsqueeze(0)
        rhs_h = [r.unsqueeze(0) for r in rhs_h]

    L = len(rhs_h)
    S_inv = blocks["S_inv"]
    D_ell_invs = blocks["D_ell_invs"]
    B_ells = blocks["B_ells"]

    # Step 1: w_ell = D_ell^{-1} rhs_h_ell
    w_h = []
    for ell in range(L):
        w_h.append((D_ell_invs[ell] @ rhs_h[ell].unsqueeze(-1)).squeeze(-1))

    # Step 2: rhs_g_eff = rhs_g - sum B_ell w_ell
    rhs_g_eff = rhs_g.clone()
    for ell in range(L):
        rhs_g_eff = rhs_g_eff - (B_ells[ell] @ w_h[ell].unsqueeze(-1)).squeeze(-1)

    # Step 3: a_g = S^{-1} rhs_g_eff
    a_g = (S_inv @ rhs_g_eff.unsqueeze(-1)).squeeze(-1)

    # Step 4: a_h_ell = D_ell^{-1} (rhs_h_ell - B_ell^* a_g)
    a_h = []
    for ell in range(L):
        rhs_corr = rhs_h[ell] - (B_ells[ell].conj().T @ a_g.unsqueeze(-1)).squeeze(-1)
        a_h.append((D_ell_invs[ell] @ rhs_corr.unsqueeze(-1)).squeeze(-1))

    if vector_input:
        a_g = a_g.squeeze(0)
        a_h = [a.squeeze(0) for a in a_h]

    return a_g, a_h


def _run_direct_estep(
    targets: torch.Tensor,
    kappa: torch.Tensor,
    pg_b: torch.Tensor,
    delta: torch.Tensor,
    spec_g: _SpectralState,
    specs_h: list[_SpectralState],
    loc_indices: list[torch.Tensor],
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
    E-step via direct dense Schur solve.

    Each inner iteration:
      1. Build Toeplitz + dense factorization: (1+L) NUFFTs
      2. Transform kappa to feature space: (1+L) NUFFTs
      3. Direct Schur solve: 0 NUFFTs
      4. Transform mean back: (1+L) NUFFTs
      5. sigma_diag via independent batched CG: 2(1+L) NUFFTs
    """
    n = targets.numel()
    L = len(specs_h)
    rdtype = targets.dtype
    device = targets.device
    cdtype = spec_g.ws.dtype
    m = spec_g.ws.numel()

    residual = float("inf")
    total_cg = 0

    a_g = None
    a_h = None

    with torch.no_grad():
        for it in range(max_iters):
            # 1. Build Toeplitz + dense factorization
            Dg_half, Dh_half, T_all, T_locs = _build_toeplitz_operators(
                delta, spec_g, specs_h, loc_indices,
            )
            blocks = _build_schur_dense(Dg_half, Dh_half, T_all, T_locs, m, cdtype, device)

            # 2. Transform kappa to feature space
            rhs_g, rhs_h = _transform_kappa_to_feature(
                kappa, spec_g, specs_h, loc_indices, Dg_half, Dh_half,
            )

            # 3. Direct Schur solve (0 NUFFTs)
            a_g, a_h = _direct_schur_solve(rhs_g, rhs_h, blocks)

            # 4. Transform mean back
            g_mean, h_means, mean = _transform_mean_to_obs(
                a_g, a_h, spec_g, specs_h, loc_indices, Dg_half, Dh_half,
            )

            # 5. sigma_diag via independent batched CG
            probe_seed = None if seed is None else seed + 17 * (it + 1)
            probes = _sample_rademacher(
                (n_probes, n), device=device, dtype=rdtype, seed=probe_seed,
            )
            _, sig_g, cg_g = _solve_single_gp_with_probes(
                torch.zeros(n, dtype=rdtype, device=device), probes,
                delta, spec_g, cg_tol=cg_tol,
            )
            total_cg += cg_g
            sigma_diag = sig_g
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
        "a_g": a_g,
        "a_h": a_h,
        "blocks": blocks,
    }


def _compute_direct_mstep_gradient(
    kappa: torch.Tensor,
    delta: torch.Tensor,
    spec_g: _SpectralState,
    specs_h: list[_SpectralState],
    loc_indices: list[torch.Tensor],
    *,
    seed: int | None,
    a_g_warm: torch.Tensor | None = None,
    a_h_warm: list[torch.Tensor] | None = None,
    blocks: dict | None = None,
) -> dict[str, torch.Tensor]:
    """
    Compute M-step gradients via exact dense trace (ZERO NUFFTs, no probes).

    Term 1 (data-fit): from E-step coefficients.
    Term 2 (trace): exact via block-inverse formulas:
      [M^{-1}]_{gg} = S^{-1}
      [M^{-1}]_{g,hℓ} = -S^{-1} B_ell D_ell^{-1}
      [M^{-1}]_{hℓ,hℓ} = D_ell^{-1} + D_ell^{-1} B_ell^* S^{-1} B_ell D_ell^{-1}

    tr(M^{-1} dM/dtheta) uses dense m x m matrix products.
    """
    L = len(specs_h)
    m = spec_g.ws.numel()
    cdtype = spec_g.ws.dtype
    rdtype = kappa.dtype
    device = kappa.device

    # Build blocks if not provided
    if blocks is None:
        Dg_half, Dh_half, T_all, T_locs = _build_toeplitz_operators(
            delta, spec_g, specs_h, loc_indices,
        )
        blocks = _build_schur_dense(Dg_half, Dh_half, T_all, T_locs, m, cdtype, device)

    Dg_half = spec_g.ws
    Dh_half = specs_h[0].ws

    S_inv = blocks["S_inv"]
    D_ell_invs = blocks["D_ell_invs"]
    B_ells = blocks["B_ells"]
    A = blocks["A"]
    D_ells = blocks["D_ells"]
    T_all_dense = blocks["T_all_dense"]
    T_loc_denses = blocks["T_loc_denses"]

    # --- Term 1: data-fit from E-step coefficients ---
    if a_g_warm is not None and a_h_warm is not None:
        beta_g = a_g_warm / Dg_half
        betas_h = [a_h_warm[ell] / Dh_half for ell in range(L)]
    else:
        # Fall back: direct solve
        rhs_g = Dg_half * spec_g.fadj(kappa.to(dtype=cdtype))
        rhs_h = []
        for ell, idx in enumerate(loc_indices):
            rhs_h.append(Dh_half * specs_h[ell].fadj(kappa[idx].to(dtype=cdtype)))
        a_g, a_h = _direct_schur_solve(rhs_g, rhs_h, blocks)
        beta_g = a_g / Dg_half
        betas_h = [a_h[ell] / Dh_half for ell in range(L)]

    abs2_g = (beta_g.conj() * beta_g).real
    n_hypers = spec_g.Dprime.shape[1]
    term1_g = spec_g.Dprime.real.T @ abs2_g

    term1_h = torch.zeros(n_hypers, dtype=rdtype, device=device)
    for ell in range(L):
        abs2_ell = (betas_h[ell].conj() * betas_h[ell]).real
        term1_h = term1_h + specs_h[ell].Dprime.real.T @ abs2_ell

    # --- Term 2: exact trace via block-inverse formulas ---
    # Precompute block-inverse components:
    # M_inv_gg = S^{-1}
    # M_inv_g_hell = -S^{-1} B_ell D_ell^{-1}
    # M_inv_hell_hell = D_ell^{-1} + D_ell^{-1} B_ell^* S^{-1} B_ell D_ell^{-1}

    ws_g = spec_g.ws
    ws_h = specs_h[0].ws

    Dg_diag = torch.diag(Dg_half)
    Dh_diag = torch.diag(Dh_half)

    term2_g = torch.zeros(n_hypers, dtype=rdtype, device=device)
    term2_h = torch.zeros(n_hypers, dtype=rdtype, device=device)

    # For global hyperparameters theta_g:
    # dA/dtheta_g = P_theta T_all Dg_diag + Dg_diag T_all P_theta
    # where P_theta = diag(Dprime[:,j] / (2 * ws_g))
    # tr(M^{-1} dM/dtheta_g) = tr(S^{-1} dA/dtheta_g)
    for j in range(n_hypers):
        P_theta = torch.diag(spec_g.Dprime[:, j] / (2.0 * ws_g))
        dA = P_theta @ T_all_dense @ Dg_diag + Dg_diag @ T_all_dense @ P_theta
        # tr(S^{-1} dA) = sum of element-wise product of S^{-1}.T and dA
        term2_g[j] = (S_inv * dA.T).sum().real

    # For local hyperparameters theta_h:
    # Need to sum over locations.
    # tr(M^{-1} dM/dtheta_h) = sum_ell [
    #   tr(M_inv_gg * dA_ell_contribution) +  -- this is zero since dA/dtheta_h = sum dB dB
    #   ... actually let's use the full block formula
    # ]
    #
    # dM/dtheta_h has blocks:
    #   (g,g): sum_ell (dB_ell * D_ell^{-1} * ... no, dM has no g,g block for theta_h
    #   Actually: dM/dtheta_h blocks are:
    #   (g, h_ell): dB_ell/dtheta_h = Dg_diag T_ell Q_theta
    #   (h_ell, g): dB_ell^*/dtheta_h = Q_theta T_ell Dg_diag
    #   (h_ell, h_ell): dD_ell/dtheta_h = Q_theta T_ell Dh_diag + Dh_diag T_ell Q_theta
    #
    # tr(M^{-1} dM/dtheta_h) = sum_ell [
    #   tr(M_inv_{hell,g} dB_ell/dtheta_h)           -- from (g, h_ell) block
    #   + tr(M_inv_{g,hell} dB_ell^*/dtheta_h)       -- from (h_ell, g) block
    #   + tr(M_inv_{hell,hell} dD_ell/dtheta_h)      -- from (h_ell, h_ell) block
    # ]
    #
    # M_inv_{hell,g} = (M_inv_{g,hell})^* = (-S^{-1} B_ell D_ell^{-1})^* = -D_ell^{-1} B_ell^* S^{-1*}
    #   But S^{-1} is Hermitian (since S is), so S^{-1*} = S^{-1}.T.conj() = S^{-1}
    #   M_inv_{hell,g} = -D_ell^{-1} B_ell^* S^{-1}
    # Actually for real systems, M_inv_{g,hell} = -S^{-1} B_ell D_ell^{-1}
    # and M_inv_{hell,g} = M_inv_{g,hell}^T = -D_ell^{-1} B_ell^T S^{-1}
    # (in general complex: M_inv_{hell,g} = -D_ell^{-*} B_ell^* S^{-*})

    for j in range(n_hypers):
        Q_theta = torch.diag(specs_h[0].Dprime[:, j] / (2.0 * ws_h))

        for ell in range(L):
            T_ell_dense = T_loc_denses[ell]
            B_ell = B_ells[ell]
            D_ell_inv = D_ell_invs[ell]

            # dB_ell/dtheta_h = Dg_diag T_ell Q_theta
            dB_ell = Dg_diag @ T_ell_dense @ Q_theta

            # dD_ell/dtheta_h = Q_theta T_ell Dh_diag + Dh_diag T_ell Q_theta
            dD_ell = Q_theta @ T_ell_dense @ Dh_diag + Dh_diag @ T_ell_dense @ Q_theta

            # Block-inverse pieces
            # M_inv_{g,hell} = -S^{-1} B_ell D_ell^{-1}
            M_inv_g_hell = -S_inv @ B_ell @ D_ell_inv

            # M_inv_{hell,g} = -D_ell^{-1} B_ell^* S^{-1}  (using S^{-1} Hermitian)
            # For complex Hermitian: M_inv_{hell,g} = M_inv_{g,hell}^{conj().T}
            M_inv_hell_g = M_inv_g_hell.conj().T

            # M_inv_{hell,hell} = D_ell^{-1} + D_ell^{-1} B_ell^* S^{-1} B_ell D_ell^{-1}
            M_inv_hell_hell = D_ell_inv + D_ell_inv @ B_ell.conj().T @ S_inv @ B_ell @ D_ell_inv

            # Contributions:
            # From (g, h_ell) block: tr(M_inv_{hell,g} dB_ell)
            c1 = (M_inv_hell_g * dB_ell.T).sum()
            # From (h_ell, g) block: tr(M_inv_{g,hell} dB_ell^*)
            # dB_ell^*/dtheta_h = Q_theta T_ell Dg_diag
            dBstar_ell = Q_theta @ T_ell_dense @ Dg_diag
            c2 = (M_inv_g_hell * dBstar_ell.T).sum()
            # From (h_ell, h_ell) block: tr(M_inv_{hell,hell} dD_ell)
            c3 = (M_inv_hell_hell * dD_ell.T).sum()

            term2_h[j] = term2_h[j] + (c1 + c2 + c3).real

    grad_g = 0.5 * (term1_g - term2_g)
    grad_h = 0.5 * (term1_h - term2_h)

    return {
        "grad_g": grad_g,
        "grad_h": grad_h,
        "cg_iters": torch.tensor(0.0),  # no CG in direct approach
    }


class HierarchicalPGNegBinRegressorDirect:
    """
    Approach D: Dense Cholesky factorization.
    E-step uses direct Schur solve via dense Cholesky.
    M-step uses exact trace via block-inverse formulas (ZERO NUFFTs, no probes).
    Specialized for d=1 with small m.
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
        from utils.kernels import get_xis

        rdtype = self.dtype
        cdtype = torch.complex128 if rdtype == torch.float64 else torch.complex64
        dev = self.device
        d = X_t.shape[1]

        x0 = X_t.min(dim=0).values
        x1 = X_t.max(dim=0).values
        L_domain = (x1 - x0).max()

        xis_g_1d, h_g, mtot_g = get_xis(kernel_g, eps=self.spectral_eps, L=L_domain,
                                          use_integral=True, l2scaled=False, trunc_eps=self.trunc_eps)
        xis_h_1d, h_h, mtot_h = get_xis(kernel_h, eps=self.spectral_eps, L=L_domain,
                                          use_integral=True, l2scaled=False, trunc_eps=self.trunc_eps)
        if mtot_h >= mtot_g:
            xis_1d, h, mtot = xis_h_1d, h_h, mtot_h
        else:
            xis_1d, h, mtot = xis_g_1d, h_g, mtot_g

        spec_g = _build_local_spectral_state(
            X_t, kernel_g, xis_1d=xis_1d, h=h, mtot=mtot,
            spectral_eps=self.spectral_eps, nufft_eps=self.nufft_eps,
            rdtype=rdtype, cdtype=cdtype, device=dev,
        )

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
    ) -> "HierarchicalPGNegBinRegressorDirect":
        rdtype = self.dtype
        dev = self.device

        X_t = torch.as_tensor(X, device=dev, dtype=rdtype)
        if X_t.ndim == 1:
            X_t = X_t.unsqueeze(-1)
        y_t = torch.as_tensor(y, device=dev, dtype=rdtype)
        loc_t = torch.as_tensor(locations, device=dev, dtype=torch.long)
        n = X_t.shape[0]
        d = X_t.shape[1]

        assert d == 1, "Direct approach is specialized for d=1"

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

            spec_g, specs_h, loc_indices = self._build_spectral_states(
                X_t, loc_t, self.kernel_g_, self.kernel_h_,
            )

            # E-step via direct Schur solve
            delta, mean, sigma_diag, estep_info = _run_direct_estep(
                y_t, kappa, pg_b, delta, spec_g, specs_h, loc_indices,
                max_iters=self.e_step_iters,
                rho0=self.rho0, gamma=self.e_gamma,
                tol=self.e_step_tol,
                n_probes=self.n_e_probes,
                cg_tol=self.cg_tol,
                seed=None if self.seed is None else self.seed + 1000 * outer,
                verbose=self.verbose,
            )

            # M-step: exact dense trace (ZERO NUFFTs, no probes)
            mstep_out = _compute_direct_mstep_gradient(
                kappa, delta, spec_g, specs_h, loc_indices,
                seed=None if self.seed is None else self.seed + 1000 * outer + 500,
                a_g_warm=estep_info['a_g'],
                a_h_warm=estep_info['a_h'],
                blocks=estep_info['blocks'],
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
                "m_cg": 0,
            }
            history.append(record)

            if self.verbose:
                print(
                    f"[{outer:3d}] ls_g={record['ls_g']:.4f} var_g={record['var_g']:.4f} "
                    f"ls_h={record['ls_h']:.4f} var_h={record['var_h']:.4f} "
                    f"r={record['r']:.3f} mae={record['mae']:.3f} "
                    f"e_cg={record['e_cg']:.0f} m_cg=0 "
                    f"t={record['elapsed']:.1f}s"
                )

        # Final E-step
        if self.learn_total_count:
            r = float(torch.exp(log_r).item())
        kappa = _kappa()
        pg_b = _pg_b()
        spec_g, specs_h, loc_indices = self._build_spectral_states(
            X_t, loc_t, self.kernel_g_, self.kernel_h_,
        )
        delta, mean, sigma_diag, final_info = _run_direct_estep(
            y_t, kappa, pg_b, delta, spec_g, specs_h, loc_indices,
            max_iters=self.final_e_step_iters,
            rho0=self.rho0, gamma=self.e_gamma,
            tol=self.e_step_tol,
            n_probes=self.n_e_probes,
            cg_tol=self.cg_tol,
            seed=None if self.seed is None else self.seed + 999_999,
            verbose=self.verbose,
        )

        # Decompose into g and h
        Dg_half, Dh_half, T_all, T_locs = _build_toeplitz_operators(
            delta, spec_g, specs_h, loc_indices,
        )
        a_g = final_info['a_g']
        a_h = final_info['a_h']
        g_mean, h_means, _ = _transform_mean_to_obs(
            a_g, a_h, spec_g, specs_h, loc_indices, Dg_half, Dh_half,
        )

        self.history_ = history
        self.delta_ = delta.detach().cpu().numpy()
        self.mean_ = mean.detach().cpu().numpy()
        self.sigma_diag_ = sigma_diag.detach().cpu().numpy()
        self.total_count_ = r
        self.lengthscale_g_ = float(self.kernel_g_.lengthscale)
        self.variance_g_ = float(self.kernel_g_.variance)
        self.lengthscale_h_ = float(self.kernel_h_.lengthscale)
        self.variance_h_ = float(self.kernel_h_.variance)
        self.g_hat_ = g_mean.detach().cpu().numpy()
        self.h_hats_ = {}
        for ell, idx in enumerate(loc_indices):
            self.h_hats_[ell] = h_means[ell].detach().cpu().numpy()

        return self
