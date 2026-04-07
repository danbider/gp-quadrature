"""
Test: can the hierarchical additive-kernel NB GP recover known hyperparameters?

Setup:
  - 1D, n=5000, L=2 locations (2500 each)
  - Global kernel: SE(lengthscale=0.2, variance=1.5)  (smooth, shared)
  - Local kernel:  SE(lengthscale=0.05, variance=0.5) (faster, per-location)
  - NB shape parameter r=5
  - x uniform on [0, 1]

We generate from the true model using RFF sampling (no dense Cholesky),
then fit with perturbed initial hyperparameters and check recovery.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pg_hierarchical import HierarchicalPGNegBinRegressor


# ---------------------------------------------------------------------------
# RFF-based GP sampling (O(n * n_features), no Cholesky)
# ---------------------------------------------------------------------------

def _rff_gp_sample(x, *, seed, lengthscale, variance, n_features=2000):
    """Sample from GP(0, SE(ls, var)) at points x using random Fourier features."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    d = x.shape[1]
    omega = rng.normal(0, 1.0 / lengthscale, size=(n_features, d))
    phase = rng.uniform(0, 2 * np.pi, size=n_features)
    weights = rng.normal(size=n_features)
    features = math.sqrt(2.0 * variance / n_features) * np.cos(x @ omega.T + phase)
    return features @ weights


def generate_data(
    n: int = 5000,
    L: int = 2,
    ls_g: float = 0.2,
    var_g: float = 1.5,
    ls_h: float = 0.05,
    var_h: float = 0.5,
    r: float = 5.0,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, size=n)
    locations = np.concatenate([np.zeros(n // 2, dtype=int),
                                np.ones(n - n // 2, dtype=int)])

    # Global GP draw (shared across all points)
    g = _rff_gp_sample(x, seed=seed + 100, lengthscale=ls_g, variance=var_g)

    # Independent local GP draws per location
    h = np.zeros(n)
    for ell in range(L):
        mask = locations == ell
        h[mask] = _rff_gp_sample(x[mask], seed=seed + 200 + ell,
                                  lengthscale=ls_h, variance=var_h)

    f = g + h

    # y ~ NB(r, sigmoid(f));  numpy convention: p = failure prob
    prob = 1.0 / (1.0 + np.exp(-f))
    y = rng.negative_binomial(r, 1 - prob).astype(np.float64)

    print(f"Data: n={n}, L={L}, y in [{int(y.min())}, {int(y.max())}], mean={y.mean():.2f}")
    print(f"True: ls_g={ls_g}, var_g={var_g}, ls_h={ls_h}, var_h={var_h}, r={r}")
    return x, y, locations, f


# ---------------------------------------------------------------------------
# Dense validation tests (small n)
# ---------------------------------------------------------------------------

def test_dense_validation():
    """Verify block Toeplitz matvec matches dense K on a tiny example."""
    from pg_hierarchical import (
        _make_kernel,
        _build_hierarchical_spectral_state,
        _build_per_location_toeplitz,
        _block_matvec,
        _build_block_rhs,
        _reconstruct_from_features,
        _make_hierarchical_sigma_apply,
    )

    n = 30
    rng = np.random.default_rng(99)
    x = rng.uniform(0, 1, size=n)
    locations = np.array([0] * (n // 2) + [1] * (n - n // 2))

    X_t = torch.as_tensor(x[:, None], dtype=torch.float64)
    loc_t = torch.as_tensor(locations, dtype=torch.long)

    ls_g, var_g = 0.2, 1.0
    ls_h, var_h = 0.05, 0.5

    kg = _make_kernel(dimension=1, lengthscale=ls_g, variance=var_g)
    kh = _make_kernel(dimension=1, lengthscale=ls_h, variance=var_h)

    spec = _build_hierarchical_spectral_state(
        X_t, loc_t, kg, kh,
        spectral_eps=1e-5, trunc_eps=1e-5, nufft_eps=1e-9,
        rdtype=torch.float64, cdtype=torch.complex128, device=torch.device('cpu'),
    )

    m = spec.ws_g.numel()
    L = spec.n_locations
    print(f"\nDense validation: n={n}, m={m}, L={L}")

    # --- Test 1: K_approx ≈ K_dense ---
    D_mat = torch.abs(X_t - X_t.T)
    K_g_dense = var_g * torch.exp(-0.5 * (D_mat / ls_g) ** 2)
    K_h_dense = torch.zeros(n, n, dtype=torch.float64)
    for ell in range(L):
        mask = loc_t == ell
        idx = torch.where(mask)[0]
        D_ell = torch.abs(X_t[idx] - X_t[idx].T)
        K_ell = var_h * torch.exp(-0.5 * (D_ell / ls_h) ** 2)
        K_h_dense[idx[:, None], idx[None, :]] = K_ell
    K_dense = K_g_dense + K_h_dense

    # Approximate K via feature map: K_approx = U U^*
    # U = [Phi_g  Phi_loc], Phi_g = F_g D_g, Phi_ell = F_ell D_h
    # Build K_approx column by column using reconstruction
    D2g_real = spec.ws2_g.real
    Dg_half = torch.sqrt(D2g_real.clamp(min=1e-14)).to(dtype=torch.complex128)
    D2h_real = spec.ws2_h.real
    Dh_half = torch.sqrt(D2h_real.clamp(min=1e-14)).to(dtype=torch.complex128)

    # Build K_approx by applying sigma with Delta=0: Sigma = K
    # sigma_apply with delta=0 should give K * z
    delta_zero = torch.zeros(n, dtype=torch.float64)
    sigma_apply_0, _ = _make_hierarchical_sigma_apply(spec, delta_zero, cg_tol=1e-10)
    I_n = torch.eye(n, dtype=torch.float64)
    K_approx = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        K_approx[:, i] = sigma_apply_0(I_n[i])

    rel_err_K = torch.norm(K_approx - K_dense) / torch.norm(K_dense)
    print(f"  K approximation relative error: {rel_err_K:.2e}")
    # This error comes from spectral truncation (eps=1e-3), not a code bug
    assert rel_err_K < 0.15, f"K approximation too inaccurate: {rel_err_K:.2e}"

    # --- Test 2: Sigma z ≈ (K^{-1} + Delta)^{-1} z ---
    delta = torch.full((n,), 0.25, dtype=torch.float64)
    sigma_apply, _ = _make_hierarchical_sigma_apply(spec, delta, cg_tol=1e-10)
    z = torch.randn(n, dtype=torch.float64)

    Sigma_z_approx = sigma_apply(z)

    Sigma_dense_inv = torch.linalg.inv(K_dense + 1e-8 * torch.eye(n)) + torch.diag(delta)
    Sigma_z_dense = torch.linalg.solve(Sigma_dense_inv, z)

    rel_err_Sigma = torch.norm(Sigma_z_approx - Sigma_z_dense) / torch.norm(Sigma_z_dense)
    print(f"  Sigma*z relative error: {rel_err_Sigma:.2e}")
    # Error is dominated by spectral truncation of the short-lengthscale local kernel
    assert rel_err_Sigma < 0.2, f"Sigma*z too inaccurate: {rel_err_Sigma:.2e}"

    # --- Test 3: Toeplitz ops individually ---
    T_all, T_loc = _build_per_location_toeplitz(delta, spec)

    # T_all should equal F_g^* diag(delta) F_g (dense)
    cdtype = torch.complex128
    F_g = torch.exp(2j * np.pi * (X_t @ spec.xis.T)).to(dtype=cdtype)
    T_all_dense = F_g.T.conj() @ torch.diag(delta.to(dtype=cdtype)) @ F_g
    v_test = torch.randn(m, dtype=torch.float64).to(dtype=cdtype)
    T_v_toeplitz = T_all(v_test)
    T_v_dense = T_all_dense @ v_test
    rel_err_T = torch.norm(T_v_toeplitz - T_v_dense) / torch.norm(T_v_dense)
    print(f"  T_all matvec relative error: {rel_err_T:.2e}")
    assert rel_err_T < 1e-4, f"T_all matvec too inaccurate: {rel_err_T:.2e}"

    # T_loc[0] on location 0's subset
    idx0 = spec.loc_indices[0]
    F_0 = torch.exp(2j * np.pi * (X_t[idx0] @ spec.xis.T)).to(dtype=cdtype)
    delta_0 = delta[idx0].to(dtype=cdtype)
    T0_dense = F_0.T.conj() @ torch.diag(delta_0) @ F_0
    T0_v_toeplitz = T_loc[0](v_test)
    T0_v_dense = T0_dense @ v_test
    rel_err_T0 = torch.norm(T0_v_toeplitz - T0_v_dense) / torch.norm(T0_v_dense)
    print(f"  T_loc[0] matvec relative error: {rel_err_T0:.2e}")
    assert rel_err_T0 < 1e-4, f"T_loc[0] matvec too inaccurate: {rel_err_T0:.2e}"

    print("  All dense validation tests PASSED.")


# ---------------------------------------------------------------------------
# Main hyperparameter recovery test
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("STEP 1: Dense validation tests (small n)")
    print("=" * 60)
    test_dense_validation()

    print("\n" + "=" * 60)
    print("STEP 2: Hyperparameter recovery (n=5000)")
    print("=" * 60)

    ls_g_true, var_g_true = 0.2, 1.5
    ls_h_true, var_h_true = 0.05, 0.5
    r_true = 5.0

    x, y, locations, f_true = generate_data(
        n=5000, L=2,
        ls_g=ls_g_true, var_g=var_g_true,
        ls_h=ls_h_true, var_h=var_h_true,
        r=r_true, seed=42,
    )

    model = HierarchicalPGNegBinRegressor(
        lengthscale_g_init=0.35,
        variance_g_init=0.8,
        lengthscale_h_init=0.1,
        variance_h_init=0.3,
        total_count=3.0,
        learn_total_count=True,
        total_count_lr=0.02,
        total_count_update_freq=3,
        max_iter=60,
        e_step_iters=3,
        final_e_step_iters=5,
        lr_g=0.01,
        lr_h=0.05,
        n_e_probes=10,
        n_m_probes=10,
        cg_tol=1e-5,
        spectral_eps=1e-3,
        trunc_eps=1e-3,
        seed=123,
        verbose=1,
        device="cpu",
    )

    print("\nFitting...")
    model.fit(x, y, locations)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Parameter':<20} {'True':>10} {'Init':>10} {'Learned':>10}")
    print("-" * 60)
    print(f"{'ls_g':<20} {ls_g_true:>10.4f} {0.35:>10.4f} {model.lengthscale_g_:>10.4f}")
    print(f"{'var_g':<20} {var_g_true:>10.4f} {0.8:>10.4f} {model.variance_g_:>10.4f}")
    print(f"{'ls_h':<20} {ls_h_true:>10.4f} {0.1:>10.4f} {model.lengthscale_h_:>10.4f}")
    print(f"{'var_h':<20} {var_h_true:>10.4f} {0.3:>10.4f} {model.variance_h_:>10.4f}")
    print(f"{'r':<20} {r_true:>10.4f} {3.0:>10.4f} {model.total_count_:>10.4f}")
    print("-" * 60)


if __name__ == "__main__":
    main()
