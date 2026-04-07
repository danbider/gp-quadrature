"""
Benchmark: sdmTMB (SPDE mesh + NB2) on the grid-cell dataset.
Mirrors data loading from pg_negative_binomial_learn_r_grid_cell_demo.ipynb
and test_gpboost_grid_cell.py.

sdmTMB fits a spatial GLMM using TMB + SPDE (Lindgren et al. 2011),
which is the standard "low-d structure-exploiting" baseline for spatial
count data.  It supports nbinom2 (NB2) and Poisson likelihoods.

Usage:
    python benchmark_sdmtmb_grid_cell.py [--cutoff 0.05] [--family nbinom2]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np


# ── data loading (same as test_gpboost_grid_cell.py) ────────────────
ROOT = Path(__file__).resolve().parent
NWB_PATH = ROOT / "data" / "dandi_000582" / "sub-11265_ses-07020602_behavior+ecephys.nwb"
assert NWB_PATH.exists(), f"NWB not found: {NWB_PATH}"


def extract_unit_spike_times(spike_times, spike_times_index, unit_index):
    start = 0 if unit_index == 0 else int(spike_times_index[unit_index - 1])
    stop = int(spike_times_index[unit_index])
    return spike_times[start:stop]


def load_grid_cell_data(neuron=7, bin_size=0.01, holdout_stride=5, use_all_data=False):
    with h5py.File(NWB_PATH, "r") as f:
        position = f["processing/behavior/Position/SpatialSeriesLED1/data"][:]
        position_t = f["processing/behavior/Position/SpatialSeriesLED1/timestamps"][:]
        spike_times = f["units/spike_times"][:]
        spike_times_index = f["units/spike_times_index"][:]

    unit_spikes = extract_unit_spike_times(spike_times, spike_times_index, neuron)
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

    if use_all_data:
        X_train = X_all
        y_train = counts_all
        X_test = np.empty((0, 2), dtype=np.float64)
        y_test = np.empty((0,), dtype=np.float64)
    else:
        mask_test = (np.arange(n_bins) % holdout_stride) == 0
        mask_train = ~mask_test
        X_train = X_all[mask_train]
        y_train = counts_all[mask_train]
        X_test = X_all[mask_test]
        y_test = counts_all[mask_test]
    return (
        X_train, y_train,
        X_test, y_test,
        X_all, counts_all,
    )


# ── R script template ──────────────────────────────────────────────
R_SCRIPT = r"""
library(sdmTMB)

args <- commandArgs(trailingOnly = TRUE)
train_csv <- args[1]
test_csv  <- args[2]
grid_csv  <- args[3]
out_json  <- args[4]
mesh_cutoff <- as.numeric(args[5])
family_name <- args[6]

# Read data
train <- read.csv(train_csv)
test  <- read.csv(test_csv)
grid  <- read.csv(grid_csv)

# Build SPDE mesh
mesh <- make_mesh(train, xy_cols = c("x", "y"), cutoff = mesh_cutoff)
cat(sprintf("Mesh: %d vertices, cutoff=%.4f\n", mesh$mesh$n, mesh_cutoff))

# Choose family
fam <- switch(family_name,
    "nbinom2"  = nbinom2(),
    "nbinom1"  = nbinom1(),
    "poisson"  = poisson(),
    stop(paste("Unknown family:", family_name))
)
cat(sprintf("Family: %s\n", family_name))

# Fit
cat("Fitting sdmTMB ...\n")
t0 <- proc.time()
fit <- sdmTMB(
    count ~ 1,
    data = train,
    mesh = mesh,
    family = fam,
    spatial = "on"
)
fit_time <- (proc.time() - t0)[["elapsed"]]
cat(sprintf("Fit time: %.2f s\n", fit_time))

# Summary
cat("\n--- Model summary ---\n")
print(summary(fit))

# Predict on test set (skip if empty)
if (nrow(test) > 0) {
    cat("\nPredicting on test set ...\n")
    t0 <- proc.time()
    pred_test <- predict(fit, newdata = test)
    pred_time_test <- (proc.time() - t0)[["elapsed"]]
    test_mu  <- exp(pred_test$est)
    test_mae <- mean(abs(test_mu - test$count))
    cat(sprintf("Test MAE: %.5f\n", test_mae))
    cat(sprintf("Predicted test mean: %.5f\n", mean(test_mu)))
    cat(sprintf("Observed  test mean: %.5f\n", mean(test$count)))
    cat(sprintf("Prediction time (test): %.2f s\n", pred_time_test))
} else {
    cat("\nNo test set (training on all data)\n")
    pred_time_test <- NA
    test_mu <- numeric(0)
    test_mae <- NA
}

# Predict on grid
cat("Predicting on grid ...\n")
t0 <- proc.time()
pred_grid <- predict(fit, newdata = grid)
pred_time_grid <- (proc.time() - t0)[["elapsed"]]
grid_mu  <- exp(pred_grid$est)
cat(sprintf("Prediction time (grid): %.2f s\n", pred_time_grid))

# Extract tidy parameter estimates
tidy_fixed  <- tidy(fit, effects = "fixed")
tidy_ran    <- tidy(fit, effects = "ran_pars")
cat("\n--- Tidy parameters ---\n")
print(tidy_fixed)
print(tidy_ran)

# Extract key GP parameters
matern_range <- tidy_ran$estimate[tidy_ran$term == "range"]
spatial_sd   <- tidy_ran$estimate[tidy_ran$term == "sigma_O"]
dispersion   <- if ("phi" %in% tidy_ran$term) tidy_ran$estimate[tidy_ran$term == "phi"] else NA
intercept    <- tidy_fixed$estimate[tidy_fixed$term == "(Intercept)"]

# Optimizer iterations and convergence
nlminb_iters <- fit$model$iterations
nlminb_evals <- fit$model$evaluations
nlminb_conv  <- fit$model$convergence  # 0 = converged
nlminb_msg   <- fit$model$message
cat(sprintf("\nOptimizer: nlminb\n"))
cat(sprintf("  Convergence code:    %d (%s)\n", nlminb_conv, nlminb_msg))
cat(sprintf("  Iterations:          %d\n", nlminb_iters))
cat(sprintf("  Function evaluations: %d\n", nlminb_evals[["function"]]))
cat(sprintf("  Gradient evaluations: %d\n", nlminb_evals[["gradient"]]))

cat(sprintf("\nGP parameters:\n"))
cat(sprintf("  Matern range:    %.4f\n", matern_range))
cat(sprintf("  Spatial SD:      %.4f\n", spatial_sd))
cat(sprintf("  Matern variance: %.4f  (SD^2)\n", spatial_sd^2))
cat(sprintf("  Dispersion (phi): %.4f\n", dispersion))
cat(sprintf("  Intercept:       %.4f\n", intercept))

# Write structured output
result <- list(
    fit_time_sec       = fit_time,
    nlminb_iterations  = nlminb_iters,
    nlminb_fn_evals    = nlminb_evals[["function"]],
    nlminb_grad_evals  = nlminb_evals[["gradient"]],
    pred_time_test_sec = pred_time_test,
    pred_time_grid_sec = pred_time_grid,
    test_mae           = test_mae,
    pred_test_mean     = mean(test_mu),
    obs_test_mean      = mean(test$count),
    mesh_n_vertices    = mesh$mesh$n,
    mesh_cutoff        = mesh_cutoff,
    family             = family_name,
    matern_range       = matern_range,
    spatial_sd         = spatial_sd,
    matern_variance    = spatial_sd^2,
    dispersion_phi     = dispersion,
    intercept          = intercept,
    test_pred_mu       = as.list(test_mu),
    grid_pred_mu       = as.list(grid_mu),
    grid_pred_latent   = as.list(pred_grid$est)
)
writeLines(jsonlite::toJSON(result, auto_unbox = TRUE, digits = 8), out_json)
cat(sprintf("\nResults written to %s\n", out_json))
"""


def run_sdmtmb_benchmark(cutoff=0.05, family="nbinom2", use_all_data=False):
    X_train, y_train, X_test, y_test, X_all, y_all = load_grid_cell_data(use_all_data=use_all_data)

    # Evaluation grid (same as notebook)
    grid_size = 36
    gx, gy = np.meshgrid(
        np.linspace(-1, 1, grid_size),
        np.linspace(-1, 1, grid_size),
        indexing="ij",
    )
    X_grid = np.column_stack([gx.ravel(), gy.ravel()])

    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}, Grid: {X_grid.shape[0]}")
    print(f"Mean count: {y_train.mean():.5f}, Frac zero: {np.mean(y_train == 0):.4f}")
    print(f"Family: {family}, Mesh cutoff: {cutoff}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        train_csv = tmpdir / "train.csv"
        test_csv = tmpdir / "test.csv"
        grid_csv = tmpdir / "grid.csv"
        out_json = tmpdir / "result.json"
        r_script = tmpdir / "run_sdmtmb.R"

        # Write CSVs — sdmTMB expects a data.frame with named columns
        # For the grid (no counts), add a dummy count column
        np.savetxt(
            train_csv,
            np.column_stack([X_train, y_train]),
            delimiter=",",
            header="x,y,count",
            comments="",
        )
        np.savetxt(
            test_csv,
            np.column_stack([X_test, y_test]),
            delimiter=",",
            header="x,y,count",
            comments="",
        )
        np.savetxt(
            grid_csv,
            np.column_stack([X_grid, np.zeros(X_grid.shape[0])]),
            delimiter=",",
            header="x,y,count",
            comments="",
        )
        r_script.write_text(R_SCRIPT)

        # Check jsonlite is available
        subprocess.run(
            ["R", "-e", "if (!requireNamespace('jsonlite', quietly=TRUE)) install.packages('jsonlite', repos='https://cloud.r-project.org')"],
            check=True,
            capture_output=True,
        )

        # Run R
        print("\n--- Running R/sdmTMB ---")
        t_total = time.time()
        proc = subprocess.run(
            [
                "Rscript", "--vanilla", str(r_script),
                str(train_csv), str(test_csv), str(grid_csv), str(out_json),
                str(cutoff), family,
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        wall_time = time.time() - t_total

        print(proc.stdout)
        if proc.stderr:
            # Filter out R startup noise
            stderr_lines = [
                l for l in proc.stderr.splitlines()
                if not any(skip in l for skip in ["package", "built under"])
            ]
            if stderr_lines:
                print("STDERR:", "\n".join(stderr_lines))

        if proc.returncode != 0:
            print(f"R exited with code {proc.returncode}")
            sys.exit(1)

        # Parse results
        with open(out_json) as f:
            result = json.load(f)

        # Drop large arrays for console summary
        summary = {k: v for k, v in result.items()
                   if k not in ("test_pred_mu", "grid_pred_mu", "grid_pred_latent")}
        summary["total_wall_time_sec"] = wall_time

        print("\n=== sdmTMB Benchmark Summary ===")
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        # Parameter comparison with PG-GP notebook results
        print("\n=== Parameter Comparison: sdmTMB vs PG-GP ===")
        print(f"  {'Parameter':<25} {'sdmTMB (SPDE)':<18} {'PG-GP (notebook)':<18}")
        print(f"  {'-'*25} {'-'*18} {'-'*18}")
        print(f"  {'Matern range / lscale':<25} {result.get('matern_range', float('nan')):<18.4f} {'0.3827':<18}")
        print(f"  {'Spatial var (SD^2)':<25} {result.get('matern_variance', float('nan')):<18.4f} {'4.3241':<18}")
        print(f"  {'Dispersion / r':<25} {result.get('dispersion_phi', float('nan')):<18.4f} {'1.3817':<18}")
        print(f"  {'Fit time (s)':<25} {result['fit_time_sec']:<18.2f} {'11.55':<18}")
        test_mae_str = f"{result['test_mae']:<18.5f}" if result['test_mae'] is not None else "N/A (all-data)    "
        print(f"  {'Test MAE':<25} {test_mae_str} {'0.04531':<18}")

        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sdmTMB SPDE benchmark on grid-cell data")
    parser.add_argument("--cutoff", type=float, default=0.05,
                        help="SPDE mesh cutoff (smaller = finer mesh, slower)")
    parser.add_argument("--family", type=str, default="nbinom2",
                        choices=["nbinom2", "nbinom1", "poisson"],
                        help="Count likelihood family")
    parser.add_argument("--all-data", action="store_true",
                        help="Train on all data (no holdout)")
    args = parser.parse_args()
    run_sdmtmb_benchmark(cutoff=args.cutoff, family=args.family, use_all_data=args.all_data)
