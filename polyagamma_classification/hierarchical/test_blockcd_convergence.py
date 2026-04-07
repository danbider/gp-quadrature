"""
Test that block CD and joint solve converge to the same result.

Uses synthetic data with known structure (small enough for both methods).
"""
import sys
import time
import numpy as np
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "hierarchical"))

from hierarchical.pg_hierarchical import HierarchicalPGNegBinRegressor
from hierarchical.pg_hierarchical_blockcd import HierarchicalPGNegBinRegressorBlockCD


def make_synthetic_data(*, n_per_loc=200, n_locations=3, seed=42):
    """Generate synthetic NB data with known global + local GP structure."""
    rng = np.random.default_rng(seed)

    # Random Fourier features to sample GPs
    def rff_sample(x, *, ls, var, n_feat=500, rng_key=None):
        r = rng_key if rng_key is not None else rng
        W = r.standard_normal((n_feat, 1)) / ls
        b = r.uniform(0, 2 * np.pi, n_feat)
        Z = np.sqrt(2 * var / n_feat) * np.cos(x @ W.T + b)
        return Z @ r.standard_normal(n_feat)

    # Generate x-values per location (overlapping)
    xs = []
    locs = []
    for ell in range(n_locations):
        x_ell = rng.uniform(0, 2, size=n_per_loc)
        xs.append(x_ell)
        locs.append(np.full(n_per_loc, ell))

    X = np.concatenate(xs).reshape(-1, 1)
    locations = np.concatenate(locs)

    # Sample global and local GPs
    g = rff_sample(X, ls=0.3, var=1.5, rng_key=np.random.default_rng(seed))
    h_all = np.zeros(len(X))
    for ell in range(n_locations):
        mask = locations == ell
        h_ell = rff_sample(X[mask], ls=0.1, var=0.5, rng_key=np.random.default_rng(seed + ell + 1))
        h_all[mask] = h_ell

    f = g + h_all
    r_true = 5.0

    # Sample NB counts
    p = 1.0 / (1.0 + np.exp(-f))
    y = rng.negative_binomial(r_true, 1 - p)

    return X, y.astype(float), locations, f, g, h_all


# Generate data
print("=== Generating synthetic data ===")
X, y, locations, f_true, g_true, h_true = make_synthetic_data(n_per_loc=200, n_locations=5)
print(f"n = {len(y)}, locations = {len(np.unique(locations))}")
print(f"y: mean={y.mean():.1f}, max={y.max():.0f}")

# Shared hyperparameters
kwargs = dict(
    lengthscale_g_init=0.3,
    variance_g_init=1.5,
    lengthscale_h_init=0.1,
    variance_h_init=0.5,
    total_count=5.0,
    learn_total_count=False,
    max_iter=20,
    e_step_iters=3,
    lr=0.03,
    cg_tol=1e-6,
    seed=42,
    verbose=1,
)

# Fit with joint solver
print("\n=== Joint solver ===")
t0 = time.perf_counter()
model_joint = HierarchicalPGNegBinRegressor(**kwargs)
model_joint.fit(X, y, locations)
t_joint = time.perf_counter() - t0
print(f"Time: {t_joint:.1f}s")

# Fit with block CD
print("\n=== Block CD solver ===")
t0 = time.perf_counter()
model_cd = HierarchicalPGNegBinRegressorBlockCD(cd_sweeps=5, **kwargs)
model_cd.fit(X, y, locations)
t_cd = time.perf_counter() - t0
print(f"Time: {t_cd:.1f}s")

# Compare results
print("\n=== Comparison ===")
print(f"{'':20s} {'Joint':>10s} {'Block CD':>10s} {'Diff':>10s}")
print(f"{'ls_g':20s} {model_joint.lengthscale_g_:10.4f} {model_cd.lengthscale_g_:10.4f} {abs(model_joint.lengthscale_g_ - model_cd.lengthscale_g_):10.4f}")
print(f"{'var_g':20s} {model_joint.variance_g_:10.4f} {model_cd.variance_g_:10.4f} {abs(model_joint.variance_g_ - model_cd.variance_g_):10.4f}")
print(f"{'ls_h':20s} {model_joint.lengthscale_h_:10.4f} {model_cd.lengthscale_h_:10.4f} {abs(model_joint.lengthscale_h_ - model_cd.lengthscale_h_):10.4f}")
print(f"{'var_h':20s} {model_joint.variance_h_:10.4f} {model_cd.variance_h_:10.4f} {abs(model_joint.variance_h_ - model_cd.variance_h_):10.4f}")

mean_corr = np.corrcoef(model_joint.mean_, model_cd.mean_)[0, 1]
mean_mae = np.mean(np.abs(model_joint.mean_ - model_cd.mean_))
g_corr = np.corrcoef(model_joint.g_hat_, model_cd.g_hat_)[0, 1]
g_mae = np.mean(np.abs(model_joint.g_hat_ - model_cd.g_hat_))

print(f"\n{'Posterior mean corr':20s} {mean_corr:10.6f}")
print(f"{'Posterior mean MAE':20s} {mean_mae:10.6f}")
print(f"{'g_hat corr':20s} {g_corr:10.6f}")
print(f"{'g_hat MAE':20s} {g_mae:10.6f}")

# Correlation with ground truth
corr_joint = np.corrcoef(f_true, model_joint.mean_)[0, 1]
corr_cd = np.corrcoef(f_true, model_cd.mean_)[0, 1]
print(f"\n{'Corr w/ true f':20s} {corr_joint:10.6f} {corr_cd:10.6f}")

print(f"\n{'Speedup':20s} {t_joint/t_cd:10.2f}x")
