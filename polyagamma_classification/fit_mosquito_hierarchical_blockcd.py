"""
Fit hierarchical PG NB-GP to mosquito ovitrap egg counts — block CD variant.

Identical hyperparameters to fit_mosquito_hierarchical.py for comparison.
"""
import sys, time
import numpy as np
import pandas as pd
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "hierarchical"))

from hierarchical.pg_hierarchical_blockcd import HierarchicalPGNegBinRegressorBlockCD

# ---- Load data ----
df = pd.read_pickle(ROOT / "data" / "mosquito" / "italy_ovitrap.pkl")
df = df.dropna(subset=['value'])

X = (df['week'].values / 52.0).reshape(-1, 1).astype(np.float64)
y = df['value'].values.astype(np.float64)
locations = df['site_idx'].values.astype(np.int64)

n_sites = len(np.unique(locations))
n = len(y)

print(f"Fitting: n={n:,}, L={n_sites} sites, X in [{X.min():.2f}, {X.max():.2f}]")
print(f"y: mean={y.mean():.1f}, max={y.max():.0f}")

# ---- Fit with block CD ----
t_start = time.perf_counter()
model = HierarchicalPGNegBinRegressorBlockCD(
    lengthscale_g_init=0.15,
    variance_g_init=2.0,
    lengthscale_h_init=0.06,
    variance_h_init=1.0,
    total_count=3.0,
    learn_total_count=True,
    total_count_lr=0.02,
    max_iter=15,
    e_step_iters=3,
    cd_sweeps=5,
    lr=0.05,
    cg_tol=1e-5,
    seed=42,
    verbose=1,
)
model.fit(X, y, locations)
elapsed = time.perf_counter() - t_start

print(f"\n--- Fit completed in {elapsed:.1f}s ({elapsed/15:.1f}s per iter) ---")
print(f"Global kernel:  ls={model.lengthscale_g_:.4f} ({model.lengthscale_g_*52:.1f} weeks)")
print(f"Local kernel:   ls={model.lengthscale_h_:.4f} ({model.lengthscale_h_*52:.1f} weeks)")
print(f"Dispersion r:   {model.total_count_:.3f}")

# Compare with joint solver baseline: 1331.1s (88.7s per iter) for 15 iterations
print(f"\nJoint solver baseline: 1331.1s (88.7s per iter)")
print(f"Block CD:              {elapsed:.1f}s ({elapsed/15:.1f}s per iter)")
print(f"Speedup:               {1331.1/elapsed:.2f}x")
