"""
Benchmark GPBoost negative-binomial GP on the grid-cell dataset.
Tests multiple approximation strategies to be fair to GPBoost.

NOTE: With 48k training points, GPBoost can be very slow. We cap iterations
and test multiple approximations to find the best speed/quality tradeoff.
"""
import os, sys, time
from pathlib import Path
import numpy as np
import h5py
import gpboost as gpb

# ── data loading (from notebook) ─────────────────────────────────────
ROOT = Path(__file__).resolve().parent
NWB_PATH = ROOT / "data" / "dandi_000582" / "sub-11265_ses-07020602_behavior+ecephys.nwb"
assert NWB_PATH.exists(), f"NWB not found: {NWB_PATH}"

def extract_unit_spike_times(spike_times, spike_times_index, unit_index):
    start = 0 if unit_index == 0 else int(spike_times_index[unit_index - 1])
    stop = int(spike_times_index[unit_index])
    return spike_times[start:stop]

with h5py.File(NWB_PATH, "r") as f:
    position = f["processing/behavior/Position/SpatialSeriesLED1/data"][:]
    position_t = f["processing/behavior/Position/SpatialSeriesLED1/timestamps"][:]
    spike_times_data = f["units/spike_times"][:]
    spike_times_index = f["units/spike_times_index"][:]

neuron = 7
bin_size = 0.01
holdout_stride = 5

unit_spikes = extract_unit_spike_times(spike_times_data, spike_times_index, neuron)
t0, t1 = float(position_t[0]), float(position_t[-1])
n_bins = int(np.floor((t1 - t0) / bin_size))
edges = t0 + np.arange(n_bins + 1) * bin_size
centers = edges[:-1] + 0.5 * bin_size

counts_all = np.histogram(unit_spikes, bins=edges)[0].astype(np.float64)
position_interp = np.column_stack([
    np.interp(centers, position_t, position[:, 0]),
    np.interp(centers, position_t, position[:, 1]),
])

coord_mins = position_interp.min(axis=0)
coord_maxs = position_interp.max(axis=0)
coord_span = np.where(coord_maxs > coord_mins, coord_maxs - coord_mins, 1.0)
X_all = 2.0 * (position_interp - coord_mins) / coord_span - 1.0

mask_test = (np.arange(n_bins) % holdout_stride) == 0
mask_train = ~mask_test
X_train = X_all[mask_train]
y_train = counts_all[mask_train]
X_test = X_all[mask_test]
y_test = counts_all[mask_test]

# Add tiny jitter to break coordinate ties (required by Vecchia for non-Gaussian)
rng = np.random.RandomState(42)
X_train_jit = X_train + rng.randn(*X_train.shape) * 1e-8
X_test_jit = X_test + rng.randn(*X_test.shape) * 1e-8

print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
print(f"Mean count: {y_train.mean():.5f}, Frac zero: {np.mean(y_train == 0):.4f}")
print("=" * 70)


def run_config(name, model_kw, optim_kw):
    """Fit one GPBoost config and return results dict."""
    print(f"\n{'─' * 70}")
    print(f"Config: {name}")
    print(f"  Model: {model_kw}")
    print(f"  Optim: {optim_kw}")
    print(f"{'─' * 70}")

    try:
        gp_model = gpb.GPModel(
            gp_coords=X_train_jit,
            cov_function="matern",
            cov_fct_shape=1.5,
            likelihood="negative_binomial",
            **model_kw,
        )
        gp_model.set_optim_params(params=optim_kw)

        t_fit = time.time()
        gp_model.fit(y=y_train)
        fit_time = time.time() - t_fit

        cov_pars = gp_model.get_cov_pars()
        print(f"\nFit time: {fit_time:.2f} s")
        print(f"Covariance params:\n{cov_pars}")

        t_pred = time.time()
        pred = gp_model.predict(
            gp_coords_pred=X_test_jit,
            predict_var=True,
            predict_response=True,
        )
        pred_time = time.time() - t_pred

        pred_mu = pred["mu"]
        mae = float(np.mean(np.abs(pred_mu - y_test)))

        print(f"Predict time: {pred_time:.2f} s")
        print(f"Total time (fit+pred): {fit_time + pred_time:.2f} s")
        print(f"Test MAE: {mae:.5f}")
        print(f"Pred mean: {pred_mu.mean():.5f}, Obs mean: {y_test.mean():.5f}")

        return dict(
            fit_time=fit_time, pred_time=pred_time, mae=mae,
            pred_mean=float(pred_mu.mean()), obs_mean=float(y_test.mean()),
        )
    except Exception as e:
        print(f"FAILED: {e}")
        return dict(error=str(e))


# ── Run configs one at a time, starting with fastest ─────────────────
results = {}

# 1) FITC — inducing-point method, typically fastest for large N
results["FITC (m=500)"] = run_config(
    "FITC (m=500)",
    model_kw=dict(gp_approx="fitc", num_ind_points=500),
    optim_kw=dict(maxit=200, trace=True),
)

# 2) Vecchia with fewer neighbors — should be faster
results["Vecchia (k=15)"] = run_config(
    "Vecchia (k=15)",
    model_kw=dict(gp_approx="vecchia", num_neighbors=15),
    optim_kw=dict(maxit=200, trace=True),
)

# 3) Full-scale Vecchia (VIF) — hybrid method
results["Full-scale Vecchia (m=200, k=30)"] = run_config(
    "Full-scale Vecchia (m=200, k=30)",
    model_kw=dict(gp_approx="full_scale_vecchia", num_ind_points=200, num_neighbors=30),
    optim_kw=dict(maxit=200, trace=True),
)

# 4) Vecchia k=30 (the "default" good setting)
results["Vecchia (k=30)"] = run_config(
    "Vecchia (k=30)",
    model_kw=dict(gp_approx="vecchia", num_neighbors=30),
    optim_kw=dict(maxit=200, trace=True),
)

# ── Summary ──────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("SUMMARY")
print(f"{'=' * 70}")
for name, res in results.items():
    if "error" in res:
        print(f"  {name}: FAILED — {res['error'][:80]}")
    else:
        total = res['fit_time'] + res['pred_time']
        print(f"  {name}: fit={res['fit_time']:.1f}s  pred={res['pred_time']:.1f}s  total={total:.1f}s  MAE={res['mae']:.5f}")
