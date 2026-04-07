"""
Fit hierarchical PG NB-GP to a choice-selective PO neuron from IBL.
"""
import sys, pickle, math, time
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "hierarchical"))

from one.api import ONE
from iblatlas.regions import BrainRegions
from hierarchical.pg_hierarchical import HierarchicalPGNegBinRegressor

# ---- Load data ----
one = ONE(base_url='https://openalyx.internationalbrainlab.org',
          username='intbrainlab', password='international')
eid = 'ebce500b-c530-47de-8cb1-963c552703ea'
coll = 'alf/probe00/pykilosort'

spike_times = one.load_dataset(eid, 'spikes.times.npy', collection=coll)
spike_clusters = one.load_dataset(eid, 'spikes.clusters.npy', collection=coll)
trials = one.load_dataset(eid, '_ibl_trials.table.pqt', collection='alf')

# Trial info
valid = (~trials['stimOn_times'].isna()) & (trials['choice'].isin([-1, 1]))
trials_v = trials[valid]
stim_times = trials_v['stimOn_times'].values
choices = trials_v['choice'].values
go_cue = trials_v['goCue_times'].values
response = trials_v['response_times'].values

# Epoch timing (median relative to stim onset)
median_go = np.nanmedian(go_cue - stim_times)
median_resp = np.nanmedian(response - stim_times)

BIN_SIZE = 0.01
TRIAL_DURATION = 2.0
N_BINS = int(TRIAL_DURATION / BIN_SIZE)
t_bins = np.arange(N_BINS) * BIN_SIZE + BIN_SIZE / 2

# ---- Build trial-aligned counts for PO cluster 759 ----
clu = 759
clu_spikes = spike_times[spike_clusters == clu]
n_trials = len(stim_times)

Y = np.zeros((n_trials, N_BINS), dtype=np.int32)
for i, t0 in enumerate(stim_times):
    trial_spikes = clu_spikes[(clu_spikes >= t0) & (clu_spikes < t0 + TRIAL_DURATION)]
    bin_idx = ((trial_spikes - t0) / BIN_SIZE).astype(int)
    bin_idx = bin_idx[bin_idx < N_BINS]
    for b in bin_idx:
        Y[i, b] += 1

print(f"PO Cluster {clu}: {n_trials} trials x {N_BINS} bins = {n_trials * N_BINS} points")
print(f"Mean count/bin = {Y.mean():.3f}, max = {Y.max()}")
print(f"Left trials: {(choices==-1).sum()}, Right trials: {(choices==1).sum()}")

# ---- Flatten ----
X = np.tile(t_bins, n_trials).reshape(-1, 1)
y = Y.ravel().astype(float)
locations = np.repeat((choices == 1).astype(int), N_BINS)

# ---- Fit ----
t_start = time.perf_counter()
model = HierarchicalPGNegBinRegressor(
    lengthscale_g_init=0.3,
    variance_g_init=1.0,
    lengthscale_h_init=0.1,
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
elapsed = time.perf_counter() - t_start

print(f"\n--- Fit completed in {elapsed:.1f}s ---")
print(f"Global kernel:  ls={model.lengthscale_g_:.4f}, var={model.variance_g_:.4f}")
print(f"Local kernel:   ls={model.lengthscale_h_:.4f}, var={model.variance_h_:.4f}")
print(f"Dispersion r:   {model.total_count_:.3f}")

# ---- Extract decomposition ----
mean_full = model.mean_
g_hat = model.g_hat_

mean_mat = mean_full.reshape(n_trials, N_BINS)
g_mat = g_hat.reshape(n_trials, N_BINS)

left_mask = choices == -1
right_mask = choices == 1

f_left = mean_mat[left_mask].mean(axis=0)
f_right = mean_mat[right_mask].mean(axis=0)
g_avg = g_mat.mean(axis=0)

r = model.total_count_
rate_left = r * np.exp(f_left)
rate_right = r * np.exp(f_right)

# Empirical PSTHs (smoothed with 50ms Gaussian for comparison)
from scipy.ndimage import gaussian_filter1d
sigma_smooth = 0.05 / BIN_SIZE  # 50ms in bins
psth_left_raw = Y[left_mask].mean(axis=0)
psth_right_raw = Y[right_mask].mean(axis=0)
psth_left_smooth = gaussian_filter1d(psth_left_raw, sigma_smooth)
psth_right_smooth = gaussian_filter1d(psth_right_raw, sigma_smooth)

h_left = f_left - g_avg
h_right = f_right - g_avg

# ---- Plot ----
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Epoch shading helper
def add_epochs(ax):
    ax.axvline(0, color='green', ls='--', alpha=0.7, label='Stim onset')
    ax.axvline(median_resp, color='orange', ls='--', alpha=0.7,
               label=f'Median response ({median_resp:.2f}s)')
    # Shade stimulus-to-response as "decision period"
    ax.axvspan(0, median_resp, alpha=0.05, color='green')
    ax.axvspan(median_resp, TRIAL_DURATION, alpha=0.05, color='orange')

# Panel 1: Empirical PSTHs vs model fits
ax = axes[0]
ax.plot(t_bins, psth_left_raw, 'b-', alpha=0.3, lw=0.5)
ax.plot(t_bins, psth_right_raw, 'r-', alpha=0.3, lw=0.5)
ax.plot(t_bins, psth_left_smooth, 'b-', alpha=0.6, lw=1, label='Smoothed PSTH (left)')
ax.plot(t_bins, psth_right_smooth, 'r-', alpha=0.6, lw=1, label='Smoothed PSTH (right)')
ax.plot(t_bins, rate_left, 'b-', lw=2.5, label='Model fit (left)')
ax.plot(t_bins, rate_right, 'r-', lw=2.5, label='Model fit (right)')
add_epochs(ax)
ax.set_ylabel('Spikes / 10ms bin')
ax.set_title(f'PO Neuron {clu} — IBL Left/Right Choice ({n_trials} trials, n={n_trials*N_BINS})')
ax.legend(loc='upper right', fontsize=8)

# Panel 2: Additive decomposition
ax = axes[1]
ax.plot(t_bins, g_avg, 'k-', lw=2.5, label='g(t) — global')
ax.plot(t_bins, h_left, 'b--', lw=2, label='h_left(t)')
ax.plot(t_bins, h_right, 'r--', lw=2, label='h_right(t)')
ax.axhline(0, color='gray', ls=':', alpha=0.5)
add_epochs(ax)
ax.set_ylabel('Latent f')
ax.set_title('Additive decomposition: f = g(t) + h_choice(t)')
ax.legend(loc='upper right', fontsize=8)

# Panel 3: Choice selectivity
ax = axes[2]
diff = h_right - h_left
ax.plot(t_bins, diff, 'purple', lw=2.5, label='h_right − h_left')
ax.fill_between(t_bins, diff, alpha=0.3, color='purple')
ax.axhline(0, color='gray', ls=':', alpha=0.5)
add_epochs(ax)
ax.set_ylabel('Latent difference')
ax.set_xlabel('Time from stimulus onset (s)')
ax.set_title('Choice selectivity (right − left)')
ax.legend(loc='upper right', fontsize=8)

plt.tight_layout()
outpath = ROOT / "data" / "ibl_ca1" / "ibl_po759_hierarchical_fit.png"
fig.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"\nSaved figure to {outpath}")
