"""
Presentation-quality 7-region mosquito hierarchical GP plot.

Fits HierarchicalPGNegBinRegressor with region (not site) as location,
then produces a large-font, high-contrast figure suitable for slides.
"""
import sys, os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from hierarchical.pg_hierarchical import HierarchicalPGNegBinRegressor

# ---- Load & prepare data ----
df = pd.read_pickle(ROOT / "data" / "mosquito" / "italy_ovitrap.pkl")
df = df.dropna(subset=["value"])

# Map region strings to integer indices
region_names_sorted = sorted(df["Region"].unique())
region_to_idx = {r: i for i, r in enumerate(region_names_sorted)}
df["region_idx"] = df["Region"].map(region_to_idx)

# Shorten region names for legend
short_names = {
    "Autonomous Province of Trento": "Trento",
    "Emilia-Romagna": "Emilia-R.",
    "Lazio": "Lazio",
    "Puglia": "Puglia",
    "Sicily": "Sicily",
    "Tuscany": "Tuscany",
    "Veneto": "Veneto",
}

X = (df["week"].values / 52.0).reshape(-1, 1).astype(np.float64)
y = df["value"].values.astype(np.float64)
locations = df["region_idx"].values.astype(np.int64)
n = len(y)
n_regions = len(region_names_sorted)

print(f"Fitting: n={n:,}, L={n_regions} regions")

# ---- Fit ----
t_start = time.perf_counter()
model = HierarchicalPGNegBinRegressor(
    lengthscale_g_init=0.15,
    variance_g_init=2.0,
    lengthscale_h_init=0.15,
    variance_h_init=1.0,
    total_count=0.5,
    learn_total_count=True,
    total_count_lr=0.02,
    total_count_update_freq=3,
    max_iter=30,
    e_step_iters=1,
    final_e_step_iters=2,
    lr_g=0.01,
    lr_h=0.05,
    n_e_probes=1,
    n_m_probes=1,
    cg_tol=1e-5,
    spectral_eps=1e-4,
    trunc_eps=1e-4,
    seed=42,
    verbose=1,
    device="cpu",
)
model.fit(X, y, locations)
elapsed = time.perf_counter() - t_start
print(f"\nFit: {elapsed:.1f}s")

# ---- Extract fitted quantities ----
r_learned = model.total_count_
g_hat = model.g_hat_
mean_full = model.mean_

weeks = np.arange(1, 53)

# Average g(t) by week
g_by_week = np.array([
    g_hat[df["week"].values == w].mean() if (df["week"].values == w).any() else np.nan
    for w in weeks
])
rate_g = r_learned * np.exp(g_by_week)

# Per-region fitted curves and deviations
region_rates = {}
region_devs = {}
for rname in region_names_sorted:
    ridx = region_to_idx[rname]
    mask = locations == ridx
    f_region = mean_full[mask]
    g_region = g_hat[mask]
    h_region = f_region - g_region
    wk_region = df.loc[df["region_idx"] == ridx, "week"].values

    f_by_week = np.array([
        f_region[wk_region == w].mean() if (wk_region == w).any() else np.nan
        for w in weeks
    ])
    h_by_week = np.array([
        h_region[wk_region == w].mean() if (wk_region == w).any() else np.nan
        for w in weeks
    ])
    region_rates[rname] = r_learned * np.exp(f_by_week)
    region_devs[rname] = h_by_week

# ---- Presentation-quality palette ----
region_colors = {
    "Autonomous Province of Trento": "#1f77b4",
    "Emilia-Romagna": "#2ca02c",
    "Lazio": "#d62728",
    "Puglia": "#ff7f0e",
    "Sicily": "#e377c2",
    "Tuscany": "#8c564b",
    "Veneto": "#9467bd",
}

# Month tick positions and labels
month_weeks = [1, 5, 9, 14, 18, 22, 27, 31, 36, 40, 44, 48]
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ==========================================================================
# FIGURE: 2-panel presentation version (shared phenology + per-region curves)
# ==========================================================================
fig, axes = plt.subplots(
    2, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [1, 1.2]},
)

# --- Panel 1: Shared phenology g(t) ---
ax = axes[0]
ax.plot(weeks, rate_g, "k-", lw=4, label="Shared phenology $g(t)$")
ax.fill_between(weeks, 0, rate_g, color="black", alpha=0.07)
ax.set_ylabel("Expected eggs (shared)", fontsize=18)
ax.set_title(
    f"Shared Italian mosquito phenology — {n_regions} regions, "
    f"n={n:,}, fit in {elapsed:.0f}s",
    fontsize=20, fontweight="bold", pad=14,
)
ax.legend(fontsize=16, loc="upper right", framealpha=0.9)
ax.set_xlim(1, 52)
ax.set_ylim(bottom=0)
ax.tick_params(labelsize=15)
ax.set_xticks(month_weeks)
ax.set_xticklabels(month_names, fontsize=15)
ax.grid(axis="y", alpha=0.25)

# --- Panel 2: Per-region fitted curves ---
ax = axes[1]
for rname in region_names_sorted:
    sname = short_names[rname]
    ax.plot(
        weeks, region_rates[rname],
        "-", color=region_colors[rname], lw=3, alpha=0.85,
        label=sname,
    )
ax.plot(weeks, rate_g, "k--", lw=2.5, alpha=0.45, label="Shared $g(t)$")
ax.set_ylabel("Expected eggs", fontsize=18)
ax.set_title(
    r"Per-region fitted curves $f_r(t) = g(t) + h_r(t)$",
    fontsize=20, fontweight="bold", pad=14,
)
ax.legend(fontsize=14, loc="upper right", framealpha=0.9, ncol=2)
ax.set_xlim(1, 52)
ax.set_ylim(bottom=0)
ax.tick_params(labelsize=15)
ax.set_xticks(month_weeks)
ax.set_xticklabels(month_names, fontsize=15)
ax.grid(axis="y", alpha=0.25)

fig.tight_layout(h_pad=3)
outpath = ROOT / "data" / "mosquito" / "mosquito_7region_presentation.png"
fig.savefig(outpath, dpi=200, bbox_inches="tight", facecolor="white")
print(f"\nSaved: {outpath}")


# ==========================================================================
# FIGURE: 3-panel version (add deviations panel)
# ==========================================================================
fig3, axes3 = plt.subplots(
    3, 1, figsize=(16, 14),
    gridspec_kw={"height_ratios": [1, 1.2, 1]},
)

# Panel 1: shared phenology
ax = axes3[0]
ax.plot(weeks, rate_g, "k-", lw=4, label="Shared phenology $g(t)$")
ax.fill_between(weeks, 0, rate_g, color="black", alpha=0.07)
ax.set_ylabel("Expected eggs (shared)", fontsize=18)
ax.set_title(
    f"Shared Italian mosquito phenology — {n_regions} regions, "
    f"n={n:,}, fit in {elapsed:.0f}s",
    fontsize=20, fontweight="bold", pad=14,
)
ax.legend(fontsize=16, loc="upper right", framealpha=0.9)
ax.set_xlim(1, 52)
ax.set_ylim(bottom=0)
ax.tick_params(labelsize=15)
ax.set_xticks(month_weeks)
ax.set_xticklabels(month_names, fontsize=15)
ax.grid(axis="y", alpha=0.25)

# Panel 2: per-region curves
ax = axes3[1]
for rname in region_names_sorted:
    sname = short_names[rname]
    ax.plot(
        weeks, region_rates[rname],
        "-", color=region_colors[rname], lw=3, alpha=0.85,
        label=sname,
    )
ax.plot(weeks, rate_g, "k--", lw=2.5, alpha=0.45, label="Shared $g(t)$")
ax.set_ylabel("Expected eggs", fontsize=18)
ax.set_title(
    r"Per-region fitted curves $f_r(t) = g(t) + h_r(t)$",
    fontsize=20, fontweight="bold", pad=14,
)
ax.legend(fontsize=14, loc="upper right", framealpha=0.9, ncol=2)
ax.set_xlim(1, 52)
ax.set_ylim(bottom=0)
ax.tick_params(labelsize=15)
ax.set_xticks(month_weeks)
ax.set_xticklabels(month_names, fontsize=15)
ax.grid(axis="y", alpha=0.25)

# Panel 3: deviations h_r(t)
ax = axes3[2]
for rname in region_names_sorted:
    sname = short_names[rname]
    ax.plot(
        weeks, region_devs[rname],
        "-", color=region_colors[rname], lw=3, alpha=0.85,
        label=sname,
    )
ax.axhline(0, color="gray", ls="--", lw=1.5, alpha=0.5)
ax.set_ylabel("Latent deviation $h_r(t)$", fontsize=18)
ax.set_xlabel("Week of year", fontsize=18)
ax.set_title(
    "Region-specific deviations from shared phenology",
    fontsize=20, fontweight="bold", pad=14,
)
ax.legend(fontsize=14, loc="upper right", framealpha=0.9, ncol=2)
ax.set_xlim(1, 52)
ax.tick_params(labelsize=15)
ax.set_xticks(month_weeks)
ax.set_xticklabels(month_names, fontsize=15)
ax.grid(axis="y", alpha=0.25)

fig3.tight_layout(h_pad=3)
outpath3 = ROOT / "data" / "mosquito" / "mosquito_7region_presentation_3panel.png"
fig3.savefig(outpath3, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {outpath3}")

plt.close("all")
