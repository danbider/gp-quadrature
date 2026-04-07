"""
Fit hierarchical PG NB-GP to mosquito data (block CD) and make figures.
"""
import sys, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# ---- Fit ----
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
print(f"\nFit: {elapsed:.1f}s ({elapsed/15:.1f}s/iter)")

r = model.total_count_
g_hat = model.g_hat_
mean = model.mean_
hist = model.history_

# ---- Figure 1: Training curves ----
fig, axes = plt.subplots(2, 3, figsize=(14, 7))

iters = [h['iter'] for h in hist]
ax = axes[0, 0]
ax.plot(iters, [h['ls_g'] * 52 for h in hist], 'b-o', ms=3, label='global')
ax.plot(iters, [h['ls_h'] * 52 for h in hist], 'r-o', ms=3, label='local')
ax.set_ylabel('Lengthscale (weeks)')
ax.legend(fontsize=8)
ax.set_title('Lengthscales')

ax = axes[0, 1]
ax.plot(iters, [h['var_g'] for h in hist], 'b-o', ms=3, label='global')
ax.plot(iters, [h['var_h'] for h in hist], 'r-o', ms=3, label='local')
ax.set_ylabel('Variance')
ax.legend(fontsize=8)
ax.set_title('Kernel variances')

ax = axes[0, 2]
ax.plot(iters, [h['r'] for h in hist], 'k-o', ms=3)
ax.set_ylabel('r')
ax.set_title('Dispersion')

ax = axes[1, 0]
ax.plot(iters, [h['mae'] for h in hist], 'g-o', ms=3)
ax.set_ylabel('MAE')
ax.set_xlabel('Iteration')
ax.set_title('Predicted count MAE')

ax = axes[1, 1]
ax.plot(iters, [h['e_cg'] for h in hist], 'b-o', ms=3, label='E-step')
ax.plot(iters, [h['m_cg'] for h in hist], 'r-o', ms=3, label='M-step')
ax.set_ylabel('CG iterations')
ax.set_xlabel('Iteration')
ax.legend(fontsize=8)
ax.set_title('CG iterations')

ax = axes[1, 2]
ax.plot(iters, [h['elapsed'] for h in hist], 'k-o', ms=3)
ax.set_ylabel('Cumulative time (s)')
ax.set_xlabel('Iteration')
ax.set_title('Wall clock')

fig.suptitle(f'Mosquito ovitrap — Block CD fit (n={n:,}, L={n_sites}, {elapsed:.0f}s)', fontsize=13)
fig.tight_layout()
fig.savefig(ROOT / 'mosquito_blockcd_training.png', dpi=150, bbox_inches='tight')
print("Saved mosquito_blockcd_training.png")

# ---- Figure 2: Global function + selected sites ----
t_years = X.ravel()
sort_idx = np.argsort(t_years)
t_sorted = t_years[sort_idx]
g_sorted = g_hat[sort_idx]

# Pick a few representative sites (most data)
site_counts = pd.Series(locations).value_counts()
top_sites = site_counts.head(6).index.values

fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
axes_flat = axes.ravel()

for i, site in enumerate(top_sites):
    ax = axes_flat[i]
    mask = locations == site
    t_s = t_years[mask]
    y_s = y[mask]
    f_s = mean[mask]
    h_s = model.h_hats_.get(site, np.zeros_like(f_s))

    order = np.argsort(t_s)
    t_s, y_s, f_s, h_s = t_s[order], y_s[order], f_s[order], h_s[order]

    # Predicted NB mean
    pred_mean = r * np.exp(f_s)

    ax.scatter(t_s * 52, y_s, s=5, alpha=0.3, c='gray', label='obs')
    ax.plot(t_s * 52, pred_mean, 'b-', lw=1.2, label='predicted mean')
    ax.set_ylabel('Egg count')
    ax.set_title(f'Site {site} (n={mask.sum()})')
    if i == 0:
        ax.legend(fontsize=7, loc='upper right')

axes[-1, 0].set_xlabel('Week')
axes[-1, 1].set_xlabel('Week')

fig.suptitle('Mosquito ovitrap — Predicted counts by site', fontsize=13)
fig.tight_layout()
fig.savefig(ROOT / 'mosquito_blockcd_sites.png', dpi=150, bbox_inches='tight')
print("Saved mosquito_blockcd_sites.png")

# ---- Figure 3: Global function g(t) ----
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t_sorted * 52, g_sorted, 'b-', lw=1.5)
ax.set_xlabel('Week')
ax.set_ylabel('g(t)')
ax.set_title(f'Global trend g(t) — ls={model.lengthscale_g_*52:.1f} weeks, var={model.variance_g_:.2f}')
ax.axhline(0, color='gray', ls='--', lw=0.5)
fig.tight_layout()
fig.savefig(ROOT / 'mosquito_blockcd_global.png', dpi=150, bbox_inches='tight')
print("Saved mosquito_blockcd_global.png")

plt.close('all')
