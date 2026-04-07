from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "pg_negative_binomial_learn_r_usgs_earthquakes_demo.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


cells = [
    md(
        """
        # Polyagamma GP Negative Binomial Demo On USGS Earthquake Counts

        This notebook adapts the learn-`r` negative-binomial Poly-Gamma GP demo
        to a spatial earthquake-count benchmark built from raw USGS events.

        The pipeline is:

        - load 2020 earthquakes in the requested California-ish bounding box
        - bin the raw events into a fine longitude/latitude grid
        - use the grid-cell count as the negative-binomial response
        - use the cell centroid `(longitude, latitude)` as the 2D GP input

        This is a better GP-style count example than hour-of-day alone because
        the underlying predictor space is genuinely continuous before we
        aggregate it.
        """
    ),
    code(
        """
        import os
        import sys
        import time
        from pathlib import Path
        from urllib.request import urlretrieve

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

        ROOT = Path.cwd().resolve()
        PARENT = ROOT.parent
        if str(PARENT) not in sys.path:
            sys.path.append(str(PARENT))

        from pg_classifier import PolyagammaGPNegativeBinomialRegressor

        plt.style.use("seaborn-v0_8-whitegrid")
        np.set_printoptions(suppress=True, precision=4)
        """
    ),
    md(
        """
        ## Download And Cache The USGS Query

        This is the exact query requested in the prompt. We cache the CSV
        locally so the notebook stays reproducible.
        """
    ),
    code(
        """
        QUERY_URL = (
            "https://earthquake.usgs.gov/fdsnws/event/1/query.csv"
            "?starttime=2020-01-01"
            "&endtime=2020-12-31"
            "&minmagnitude=2.5"
            "&minlatitude=32"
            "&maxlatitude=42"
            "&minlongitude=-125"
            "&maxlongitude=-114"
            "&orderby=time"
        )

        CSV_PATH = ROOT / "data" / "usgs_earthquakes" / "usgs_2020_ca_m2p5.csv"
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

        if not CSV_PATH.exists():
            print(f"Downloading {CSV_PATH.name} ...")
            urlretrieve(QUERY_URL, CSV_PATH)

        print(f"CSV path: {CSV_PATH}")
        print(f"Size: {CSV_PATH.stat().st_size / 1e6:.2f} MB")
        """
    ),
    md(
        """
        ## Load Events And Aggregate Them Into Spatial Counts

        We keep all grid cells, including zeros. That is important here because
        the count surface is sparse and the empty cells carry information.

        By default the notebook trains on all grid cells. If you want a simple
        holdout, set `use_all_data_for_training = False`.
        """
    ),
    code(
        """
        LON_MIN, LON_MAX = -125.0, -114.0
        LAT_MIN, LAT_MAX = 32.0, 42.0
        N_LON_BINS = 36
        N_LAT_BINS = 36

        use_all_data_for_training = True
        holdout_stride = 5

        df = pd.read_csv(CSV_PATH)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.sort_values("time").reset_index(drop=True)

        lon_edges = np.linspace(LON_MIN, LON_MAX, N_LON_BINS + 1)
        lat_edges = np.linspace(LAT_MIN, LAT_MAX, N_LAT_BINS + 1)
        counts_grid, _, _ = np.histogram2d(
            df["longitude"].to_numpy(),
            df["latitude"].to_numpy(),
            bins=[lon_edges, lat_edges],
        )

        lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
        lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
        grid_lon, grid_lat = np.meshgrid(lon_centers, lat_centers, indexing="ij")

        X_raw_np = np.column_stack([grid_lon.reshape(-1), grid_lat.reshape(-1)])
        y_all_np = counts_grid.reshape(-1).astype(np.float64)

        x_min = X_raw_np.min(axis=0)
        x_max = X_raw_np.max(axis=0)
        x_span = np.where(x_max > x_min, x_max - x_min, 1.0)
        X_all_np = 2.0 * (X_raw_np - x_min) / x_span - 1.0

        if use_all_data_for_training:
            mask_train = np.ones(X_all_np.shape[0], dtype=bool)
            mask_test = np.zeros(X_all_np.shape[0], dtype=bool)
            X_train_np = X_all_np
            y_train_np = y_all_np
            X_test_np = np.empty((0, 2), dtype=np.float64)
            y_test_np = np.empty((0,), dtype=np.float64)
        else:
            mask_test = (np.arange(X_all_np.shape[0]) % holdout_stride) == 0
            mask_train = np.logical_not(mask_test)
            X_train_np = X_all_np[mask_train]
            y_train_np = y_all_np[mask_train]
            X_test_np = X_all_np[mask_test]
            y_test_np = y_all_np[mask_test]

        display_lon = np.linspace(LON_MIN, LON_MAX, 72)
        display_lat = np.linspace(LAT_MIN, LAT_MAX, 72)
        dlon, dlat = np.meshgrid(display_lon, display_lat, indexing="ij")
        X_display_raw = np.column_stack([dlon.reshape(-1), dlat.reshape(-1)])
        X_display_np = 2.0 * (X_display_raw - x_min) / x_span - 1.0

        counts_grid_img = counts_grid.astype(np.float64)
        log_counts_grid_img = np.log1p(counts_grid_img)

        print(f"Events: {df.shape[0]}")
        print(f"Longitude range: {df['longitude'].min():.3f} to {df['longitude'].max():.3f}")
        print(f"Latitude range: {df['latitude'].min():.3f} to {df['latitude'].max():.3f}")
        print(f"Cells: {X_all_np.shape[0]}")
        print(f"Mean count per cell: {y_all_np.mean():.4f}")
        print(f"Max count in a cell: {y_all_np.max():.0f}")
        print(f"Fraction of zero-count cells: {np.mean(y_all_np == 0):.4f}")
        print(f"Train cells: {X_train_np.shape[0]}")
        print(f"Test cells: {X_test_np.shape[0]}")
        if use_all_data_for_training:
            print("Training mode: all cells used for fitting")
        else:
            print(f"Training mode: stride holdout with every {holdout_stride}th cell in test")
        """
    ),
    md(
        """
        ## Visualize The Raw Events And The Binned Count Surface

        This shows the raw earthquake locations, the resulting spatial count
        grid, and the distribution of counts across cells.
        """
    ),
    code(
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
        extent = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]

        scatter = axes[0].scatter(
            df["longitude"],
            df["latitude"],
            c=df["mag"],
            cmap="viridis",
            s=18,
            alpha=0.65,
            edgecolors="none",
        )
        axes[0].set_title("Raw earthquake events")
        axes[0].set_xlabel("Longitude")
        axes[0].set_ylabel("Latitude")
        fig.colorbar(scatter, ax=axes[0], fraction=0.046, label="Magnitude")

        im1 = axes[1].imshow(
            log_counts_grid_img.T,
            origin="lower",
            extent=extent,
            cmap="magma",
            aspect="auto",
        )
        axes[1].set_title("log1p(count) on the 36x36 grid")
        axes[1].set_xlabel("Longitude")
        axes[1].set_ylabel("Latitude")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, label="log1p(count)")

        axes[2].hist(y_all_np, bins=np.arange(0, min(60, int(y_all_np.max()) + 2)) - 0.5, color="#1982c4", edgecolor="black")
        axes[2].set_yscale("log")
        axes[2].set_title("Cell-count distribution")
        axes[2].set_xlabel("Count per cell")
        axes[2].set_ylabel("Number of cells (log scale)")
        """
    ),
    md(
        """
        ## Fit `PolyagammaGPNegativeBinomialRegressor` With Learned `r`

        The model sees the spatial cell centroids as inputs and the annual event
        counts inside those cells as targets.
        """
    ),
    code(
        """
        reg = PolyagammaGPNegativeBinomialRegressor(
            total_count=2.0,
            learn_total_count=True,
            total_count_lr=0.05,
            total_count_update_frequency=1,
            total_count_quadrature_nodes=16,
            lengthscale_init=0.18,
            variance_init=1.0,
            max_iter=25,
            e_step_iters=1,
            final_e_step_iters=2,
            rho0=1.0,
            gamma=1e-3,
            lr=0.04,
            n_e_probes=1,
            n_m_probes=1,
            cg_tol=1e-5,
            nufft_eps=1e-4,
            spectral_eps=1e-4,
            trunc_eps=1e-4,
            prediction_batch_size=256,
            predictive_variance_method="chebyshev",
            predictive_variance_chebyshev_nodes=7,
            use_exact_weighted_toeplitz_operator=True,
            random_state=0,
            device="cpu",
            store_history=True,
            verbose=1,
        )

        t_fit = time.time()
        reg.fit(X_train_np, y_train_np)
        fit_time = time.time() - t_fit

        pred_train = reg.predict_mean_count(X_train_np)
        if X_test_np.shape[0] > 0:
            pred_test = reg.predict_mean_count(X_test_np)
        else:
            pred_test = None

        print(f"Fit time: {fit_time:.2f} s")
        print(f"Learned lengthscale: {reg.lengthscale_:.4f}")
        print(f"Learned variance: {reg.variance_:.4f}")
        print(f"Learned total_count r: {reg.total_count_:.4f}")
        print(f"Training MAE: {np.mean(np.abs(pred_train - y_train_np)):.5f}")
        print(f"Mean predicted training count: {pred_train.mean():.5f}")
        print(f"Mean observed training count: {y_train_np.mean():.5f}")
        if pred_test is not None:
            print(f"Test MAE: {np.mean(np.abs(pred_test - y_test_np)):.5f}")
        else:
            print("Held-out test split: disabled")
        """
    ),
    md(
        """
        ## Compare The Empirical Count Grid With The GP Fit

        We evaluate the GP on a denser longitude/latitude grid for smoother
        visualization, then compare the predicted mean count surface with the
        empirical aggregated counts.
        """
    ),
    code(
        """
        latent_display = reg.decision_function(X_display_np)
        var_display = reg.predictive_variance(X_display_np)
        mean_count_display = reg.total_count_ * np.exp(latent_display + 0.5 * var_display)

        latent_display_img = latent_display.reshape(display_lon.size, display_lat.size)
        var_display_img = var_display.reshape(display_lon.size, display_lat.size)
        mean_count_display_img = mean_count_display.reshape(display_lon.size, display_lat.size)
        log_mean_count_display_img = np.log1p(mean_count_display_img)

        history = reg.history_
        iters = np.array([row["iter"] for row in history], dtype=float)
        lengthscales = np.array([row["lengthscale"] for row in history], dtype=float)
        variances = np.array([row["variance"] for row in history], dtype=float)
        total_counts = np.array([row["total_count"] for row in history], dtype=float)
        count_mae = np.array([row["mean_count_mae"] for row in history], dtype=float)
        """
    ),
    code(
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
        extent = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]

        im0 = axes[0].imshow(
            log_counts_grid_img.T,
            origin="lower",
            extent=extent,
            cmap="magma",
            aspect="auto",
        )
        axes[0].set_title("Empirical log1p(count)")
        axes[0].set_xlabel("Longitude")
        axes[0].set_ylabel("Latitude")
        fig.colorbar(im0, ax=axes[0], fraction=0.046)

        im1 = axes[1].imshow(
            log_mean_count_display_img.T,
            origin="lower",
            extent=extent,
            cmap="magma",
            aspect="auto",
        )
        axes[1].set_title("Predicted log1p(mean count)")
        axes[1].set_xlabel("Longitude")
        axes[1].set_ylabel("Latitude")
        fig.colorbar(im1, ax=axes[1], fraction=0.046)

        im2 = axes[2].imshow(
            var_display_img.T,
            origin="lower",
            extent=extent,
            cmap="cividis",
            aspect="auto",
        )
        axes[2].set_title("Predictive latent variance")
        axes[2].set_xlabel("Longitude")
        axes[2].set_ylabel("Latitude")
        fig.colorbar(im2, ax=axes[2], fraction=0.046)
        """
    ),
    code(
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

        axes[0].plot(lengthscales, variances, color="#1982c4", marker="o", linewidth=2)
        axes[0].scatter(lengthscales[0], variances[0], color="#ff595e", s=70, label="Start")
        axes[0].scatter(lengthscales[-1], variances[-1], color="#8ac926", s=70, label="Final")
        axes[0].set_title("Variance vs lengthscale")
        axes[0].set_xlabel("Lengthscale")
        axes[0].set_ylabel("Variance")
        axes[0].legend(loc="best")

        axes[1].plot(iters, total_counts, marker="o", color="#ff595e")
        axes[1].axhline(2.0, color="black", linestyle="--", linewidth=1.25, label="Initial r")
        axes[1].set_title("Learned total_count trajectory")
        axes[1].set_xlabel("Outer iteration")
        axes[1].legend(loc="best")

        axes[2].plot(iters, count_mae, marker="^", color="#8ac926")
        axes[2].set_title("Training mean-count MAE")
        axes[2].set_xlabel("Outer iteration")
        """
    ),
    code(
        """
        summary = {
            "n_events": int(df.shape[0]),
            "n_cells": int(X_all_np.shape[0]),
            "used_all_data_for_training": bool(use_all_data_for_training),
            "fit_time_sec": fit_time,
            "lengthscale": float(reg.lengthscale_),
            "variance": float(reg.variance_),
            "total_count": float(reg.total_count_),
            "train_mae": float(np.mean(np.abs(pred_train - y_train_np))),
            "mean_pred_train": float(pred_train.mean()),
            "mean_obs_train": float(y_train_np.mean()),
            "test_mae": None if pred_test is None else float(np.mean(np.abs(pred_test - y_test_np))),
        }
        summary
        """
    ),
]


nb = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "file_extension": ".py",
        },
    },
)

NOTEBOOK_PATH.write_text(nbf.writes(nb))
print(f"Wrote {NOTEBOOK_PATH}")
