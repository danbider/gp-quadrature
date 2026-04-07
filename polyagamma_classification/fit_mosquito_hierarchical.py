"""
Fit hierarchical PG NB-GP to mosquito ovitrap egg counts.

Model: y_{s,t} ~ NB(r, sigma(f_s(t)))
       f_s(t) = g(t) + h_s(t)
       g ~ GP(0, K_g)    -- shared seasonal phenology
       h_s ~ GP(0, K_h)  -- site-specific deviation

X = week of year (normalized to [0,1])
locations = site index
"""
import sys, time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "hierarchical"))

from hierarchical.pg_hierarchical import HierarchicalPGNegBinRegressor

# ---- Load data ----
df = pd.read_pickle(ROOT / "data" / "mosquito" / "italy_ovitrap.pkl")

# Use week-of-year as input, pooling across years
X = (df['week'].values / 52.0).reshape(-1, 1).astype(np.float64)
y = df['value'].values.astype(np.float64)
locations = df['site_idx'].values.astype(np.int64)

n_sites = len(np.unique(locations))
n = len(y)

print(f"n = {n} observations")
print(f"Sites = {n_sites}")
print(f"y: mean={y.mean():.1f}, median={np.median(y):.0f}, max={y.max():.0f}")
print(f"X range: [{X.min():.3f}, {X.max():.3f}]")
print(f"Fraction zeros: {(y==0).sum()/n:.2%}")

# ---- Fit ----
t_start = time.perf_counter()
model = HierarchicalPGNegBinRegressor(
    lengthscale_g_init=0.15,     # ~8 weeks — broad seasonal structure
    variance_g_init=2.0,
    lengthscale_h_init=0.06,     # ~3 weeks — finer site-specific variation
    variance_h_init=1.0,
    total_count=3.0,             # mosquito counts are very overdispersed
    learn_total_count=True,
    total_count_lr=0.02,
    max_iter=30,
    e_step_iters=3,
    lr=0.05,
    cg_tol=1e-5,
    seed=42,
    verbose=1,
)
model.fit(X, y, locations)
elapsed = time.perf_counter() - t_start

print(f"\n--- Fit completed in {elapsed:.1f}s ---")
print(f"Global kernel:  ls={model.lengthscale_g_:.4f} ({model.lengthscale_g_*52:.1f} weeks), "
      f"var={model.variance_g_:.4f}")
print(f"Local kernel:   ls={model.lengthscale_h_:.4f} ({model.lengthscale_h_*52:.1f} weeks), "
      f"var={model.variance_h_:.4f}")
print(f"Dispersion r:   {model.total_count_:.3f}")

# ---- Extract decomposition ----
mean_full = model.mean_
g_hat = model.g_hat_

# Get site info for labeling
site_info = df.groupby('site_idx').agg(
    region=('Region', 'first'),
    lat=('latitude', 'first'),
    lon=('longitude', 'first'),
    mean_val=('value', 'mean'),
    site_id=('ID', 'first'),
).sort_values('mean_val', ascending=False)

# ---- Plot ----
fig, axes = plt.subplots(3, 1, figsize=(14, 12))
weeks = np.arange(1, 53)
week_frac = weeks / 52.0

# Panel 1: Global phenology g(t)
ax = axes[0]
# Evaluate g at a regular grid of weeks
# Find the closest data points to each week
g_by_week = []
for w in weeks:
    mask = df['week'].values == w
    if mask.any():
        g_by_week.append(g_hat[mask].mean())
    else:
        g_by_week.append(np.nan)
g_by_week = np.array(g_by_week)

r = model.total_count_
rate_g = r * np.exp(g_by_week)
ax.plot(weeks, rate_g, 'k-', lw=3, label='Shared phenology g(t)')
ax.set_ylabel('Expected eggs (shared)')
ax.set_title(f'Shared seasonal phenology — {n_sites} sites, n={n}')
ax.legend()
ax.set_xlim(1, 52)

# Panel 2: Site-specific curves for a selection of sites
ax = axes[1]
# Pick 5 high-count Emilia-Romagna sites and 3 Trento sites
emilia_sites = site_info[site_info['region'] == 'Emilia-Romagna'].head(3).index.tolist()
trento_sites = site_info[site_info['region'] == 'Autonomous Province of Trento'].head(3).index.tolist()
selected_sites = emilia_sites + trento_sites

colors_e = ['#d62728', '#e377c2', '#ff7f0e']
colors_t = ['#1f77b4', '#17becf', '#2ca02c']
colors = colors_e + colors_t

for idx, site in enumerate(selected_sites):
    site_mask = locations == site
    site_mean = mean_full[site_mask]
    site_weeks = df.loc[df['site_idx'] == site, 'week'].values

    # Average f across years for each week
    f_by_week = []
    for w in weeks:
        wmask = site_weeks == w
        if wmask.any():
            f_by_week.append(site_mean[wmask].mean())
        else:
            f_by_week.append(np.nan)
    f_by_week = np.array(f_by_week)
    rate_site = r * np.exp(f_by_week)

    region = site_info.loc[site, 'region']
    label = f"Site {site_info.loc[site, 'site_id']} ({region[:7]})"
    ax.plot(weeks, rate_site, '-', color=colors[idx], lw=2, label=label, alpha=0.8)

ax.plot(weeks, rate_g, 'k--', lw=2, alpha=0.5, label='Shared g(t)')
ax.set_ylabel('Expected eggs')
ax.set_title('Per-site fitted curves f_s(t) = g(t) + h_s(t)')
ax.legend(fontsize=7, ncol=2)
ax.set_xlim(1, 52)

# Panel 3: Site deviations h_s(t) for same sites
ax = axes[2]
for idx, site in enumerate(selected_sites):
    site_mask = locations == site
    site_h = mean_full[site_mask] - g_hat[site_mask]
    site_weeks = df.loc[df['site_idx'] == site, 'week'].values

    h_by_week = []
    for w in weeks:
        wmask = site_weeks == w
        if wmask.any():
            h_by_week.append(site_h[wmask].mean())
        else:
            h_by_week.append(np.nan)
    h_by_week = np.array(h_by_week)

    region = site_info.loc[site, 'region']
    label = f"Site {site_info.loc[site, 'site_id']} ({region[:7]})"
    ax.plot(weeks, h_by_week, '-', color=colors[idx], lw=2, label=label, alpha=0.8)

ax.axhline(0, color='gray', ls=':', alpha=0.5)
ax.set_ylabel('Latent deviation h_s(t)')
ax.set_xlabel('Week of year')
ax.set_title('Site-specific deviations from shared phenology')
ax.legend(fontsize=7, ncol=2)
ax.set_xlim(1, 52)

# Add month labels
for ax in axes:
    month_weeks = [1, 5, 9, 14, 18, 22, 27, 31, 36, 40, 44, 48]
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticks(month_weeks)
    ax.set_xticklabels(month_names)

plt.tight_layout()
outpath = ROOT / "data" / "mosquito" / "mosquito_hierarchical_fit.png"
fig.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"\nSaved figure to {outpath}")
