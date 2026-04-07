"""
Fit hierarchical PG NB-GP to IBL CA1 neurons: global temporal GP + trial-type local GPs.

Data structure:
    - Each neuron has spike counts in 10ms bins across 2s trials
    - Trials are labeled as left (-1) or right (1) choice
    - We flatten across trials within each type:
        X = time-within-trial (in seconds), shape (n_trials * n_bins,)
        y = spike count per bin
        locations = 0 (left) or 1 (right)
"""
import sys, pickle, math
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "hierarchical"))

from hierarchical.pg_hierarchical import HierarchicalPGNegBinRegressor

# ---- Load data ----
with open(ROOT / "data" / "ibl_ca1" / "ibl_ca1_trial_aligned.pkl", "rb") as f:
    dat = pickle.load(f)

counts = dat["counts"]        # {cluster_id: (n_trials, n_bins)}
choices = dat["choices"]       # -1 or 1
bin_size = dat["bin_size"]     # 0.01 s
trial_dur = dat["trial_duration"]  # 2.0 s
clusters = dat["selected_clusters"]

print(f"Session: {dat['session']}")
print(f"Region: {dat['region']}")
print(f"Clusters: {clusters}")
print(f"Trials: {len(choices)} (L={sum(choices==-1)}, R={sum(choices==1)})")

# ---- Pick a single neuron to fit ----
# Use the highest-firing one
clu = clusters[0]
Y_full = counts[clu]  # (n_trials, n_bins)
n_trials, n_bins = Y_full.shape

# Time axis in seconds
t_bins = np.arange(n_bins) * bin_size + bin_size / 2  # bin centers

# ---- Use ALL trials ----
trial_sel = np.arange(n_trials)
Y = Y_full  # (537, 200)
choice_sel = choices

# Flatten: each (trial, bin) pair becomes a data point
n_total = len(trial_sel) * n_bins
X = np.tile(t_bins, len(trial_sel)).reshape(-1, 1)  # (n_total, 1)
y = Y.ravel().astype(float)  # (n_total,)
locations = np.repeat((choice_sel == 1).astype(int), n_bins)  # 0=left, 1=right

print(f"\nFitting cluster {clu}")
print(f"n = {n_total} ({len(trial_sel)} trials x {n_bins} bins)")
print(f"X range: [{X.min():.3f}, {X.max():.3f}]")
print(f"y: mean={y.mean():.3f}, max={y.max():.0f}")
print(f"Locations: {np.bincount(locations)}")

# ---- Fit the hierarchical model ----
model = HierarchicalPGNegBinRegressor(
    lengthscale_g_init=0.3,     # global: broad temporal structure
    variance_g_init=1.0,
    lengthscale_h_init=0.1,     # local: finer trial-type differences
    variance_h_init=0.5,
    total_count=5.0,
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

# ---- Extract results and plot ----
print(f"\n--- Learned parameters ---")
print(f"Global kernel:  ls={model.lengthscale_g_:.4f}, var={model.variance_g_:.4f}")
print(f"Local kernel:   ls={model.lengthscale_h_:.4f}, var={model.variance_h_:.4f}")
print(f"Dispersion r:   {model.total_count_:.3f}")

# Reconstruct per-condition posterior means
# For plotting, average the posterior f values back to (n_trials, n_bins)
mean_full = model.mean_  # (n_total,)
g_hat = model.g_hat_     # (n_total,)

# Reshape to (n_selected_trials, n_bins)
mean_mat = mean_full.reshape(len(trial_sel), n_bins)
g_mat = g_hat.reshape(len(trial_sel), n_bins)

# Separate by condition
left_mask = choice_sel == -1
right_mask = choice_sel == 1

# Average posterior mean across trials (in f-space)
f_left = mean_mat[left_mask].mean(axis=0)
f_right = mean_mat[right_mask].mean(axis=0)
g_avg = g_mat.mean(axis=0)  # global component (same for all)

# Convert to firing rate (spikes/bin): E[y] = r * exp(f) approximately
r = model.total_count_
rate_left = r * np.exp(f_left)
rate_right = r * np.exp(f_right)
rate_global = r * np.exp(g_avg)

# Empirical PSTHs for comparison
psth_left = Y_full[choices == -1].mean(axis=0)
psth_right = Y_full[choices == 1].mean(axis=0)

# ---- Plot ----
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Panel 1: Empirical PSTHs vs model fits
ax = axes[0]
ax.plot(t_bins, psth_left, 'b-', alpha=0.5, label='Empirical PSTH (left)')
ax.plot(t_bins, psth_right, 'r-', alpha=0.5, label='Empirical PSTH (right)')
ax.plot(t_bins, rate_left, 'b-', lw=2, label='Model fit (left)')
ax.plot(t_bins, rate_right, 'r-', lw=2, label='Model fit (right)')
ax.set_ylabel('Spikes / 10ms bin')
ax.set_title(f'CA1 Neuron {clu} — IBL Left/Right Choice')
ax.legend()

# Panel 2: Decomposition in f-space
ax = axes[1]
ax.plot(t_bins, g_avg, 'k-', lw=2, label='g(t) — global')
# Local components: h_left and h_right
h_left = f_left - g_avg
h_right = f_right - g_avg
ax.plot(t_bins, h_left, 'b--', lw=2, label='h_left(t)')
ax.plot(t_bins, h_right, 'r--', lw=2, label='h_right(t)')
ax.axhline(0, color='gray', ls=':', alpha=0.5)
ax.set_ylabel('Latent f')
ax.set_title('Additive decomposition: f = g(t) + h_choice(t)')
ax.legend()

# Panel 3: Trial-type difference
ax = axes[2]
ax.plot(t_bins, h_right - h_left, 'purple', lw=2, label='h_right - h_left')
ax.fill_between(t_bins, h_right - h_left, alpha=0.3, color='purple')
ax.axhline(0, color='gray', ls=':', alpha=0.5)
ax.set_ylabel('Latent difference')
ax.set_xlabel('Time from stimulus onset (s)')
ax.set_title('Choice selectivity (right - left)')
ax.legend()

plt.tight_layout()
fig.savefig(ROOT / "data" / "ibl_ca1" / "ibl_ca1_hierarchical_fit.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"\nSaved figure to data/ibl_ca1/ibl_ca1_hierarchical_fit.png")
