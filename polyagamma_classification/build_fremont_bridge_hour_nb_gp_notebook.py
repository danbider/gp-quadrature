from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "pg_negative_binomial_learn_r_fremont_bridge_hour_demo.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


cells = [
    md(
        """
        # Polyagamma GP Negative Binomial Demo On Seattle Fremont Bridge Counts

        This notebook adapts the learn-`r` negative-binomial Poly-Gamma GP demo
        to the Seattle Fremont Bridge bicycle counter.

        In this first Seattle version, the predictor is just the hour of day:

        - one row = one hour
        - predictor = hour of day
        - response = total bicycles counted during that hour

        The hour predictor is linearly normalized onto `[-1, 1]` to match the
        scaling convention used in the synthetic notebook.

        This notebook was validated with the repo-local virtualenv at
        `../venv/bin/python`.
        """
    ),
    code(
        """
        import os
        import sys
        import time
        from pathlib import Path
        from urllib.request import urlretrieve

        import matplotlib.dates as mdates
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
        ## Download And Cache The CSV

        The dataset is the public Seattle Fremont Bridge hourly bicycle count
        feed. We cache the CSV locally so the notebook stays reproducible even
        if the remote endpoint is slow.
        """
    ),
    code(
        """
        FREMONT_URL = "https://data.seattle.gov/api/views/65db-xm6k/rows.csv?accessType=DOWNLOAD"
        CSV_PATH = ROOT / "data" / "seattle_fremont_bridge" / "fremont_bridge_bicycle_counts.csv"
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

        if not CSV_PATH.exists():
            print(f"Downloading {CSV_PATH.name} ...")
            urlretrieve(FREMONT_URL, CSV_PATH)

        print(f"CSV path: {CSV_PATH}")
        print(f"Size: {CSV_PATH.stat().st_size / 1e6:.2f} MB")
        """
    ),
    md(
        """
        ## Load The Hourly Counts

        The total-count column already represents one-hour bins, so unlike the
        grid-cell NWB example there is no extra spike binning step here. Each
        row is directly one negative-binomial count observation.

        By default the notebook trains on all rows. If you want a simple
        stride-based holdout later, set `use_all_data_for_training = False`.
        """
    ),
    code(
        """
        TOTAL_COL = "Fremont Bridge Sidewalks, south of N 34th St Total"
        DATE_FMT = "%m/%d/%Y %I:%M:%S %p"

        use_all_data_for_training = True
        holdout_stride = 24

        df = pd.read_csv(CSV_PATH)
        df["Date"] = pd.to_datetime(df["Date"], format=DATE_FMT)
        df = df.sort_values("Date").dropna(subset=[TOTAL_COL]).reset_index(drop=True)
        df[TOTAL_COL] = df[TOTAL_COL].round().astype(np.int64)

        y_all_np = df[TOTAL_COL].astype(np.float64).to_numpy()
        hour_raw_np = df["Date"].dt.hour.astype(np.float64).to_numpy().reshape(-1, 1)

        x_min = hour_raw_np.min(axis=0)
        x_max = hour_raw_np.max(axis=0)
        x_span = np.where(x_max > x_min, x_max - x_min, 1.0)
        X_all_np = 2.0 * (hour_raw_np - x_min) / x_span - 1.0

        if use_all_data_for_training:
            mask_train = np.ones(df.shape[0], dtype=bool)
            mask_test = np.zeros(df.shape[0], dtype=bool)
            X_train_np = X_all_np
            y_train_np = y_all_np
            X_test_np = np.empty((0, 1), dtype=np.float64)
            y_test_np = np.empty((0,), dtype=np.float64)
        else:
            mask_test = (np.arange(df.shape[0]) % holdout_stride) == 0
            mask_train = np.logical_not(mask_test)
            X_train_np = X_all_np[mask_train]
            y_train_np = y_all_np[mask_train]
            X_test_np = X_all_np[mask_test]
            y_test_np = y_all_np[mask_test]

        hour_grid = np.linspace(0.0, 23.0, 241)
        X_grid_np = 2.0 * ((hour_grid.reshape(-1, 1) - x_min) / x_span) - 1.0

        daily_totals = (
            df.set_index("Date")[TOTAL_COL]
            .resample("D")
            .sum(min_count=1)
            .dropna()
        )
        hour_stats = (
            df.assign(hour=df["Date"].dt.hour)
            .groupby("hour")[TOTAL_COL]
            .agg(["mean", "median", "std", "count"])
            .reset_index()
        )

        print(f"Rows after dropping missing totals: {df.shape[0]}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Mean hourly count: {y_all_np.mean():.3f}")
        print(f"Hourly count variance: {y_all_np.var():.3f}")
        print(f"Fraction of zero-count rows: {np.mean(y_all_np == 0):.5f}")
        print(f"Train rows: {X_train_np.shape[0]}")
        print(f"Test rows: {X_test_np.shape[0]}")
        print(f"Normalized predictor bounds: min={X_all_np.min(axis=0)}, max={X_all_np.max(axis=0)}")
        if use_all_data_for_training:
            print("Training mode: all rows used for fitting")
        else:
            print(f"Training mode: stride holdout with every {holdout_stride}th row in test")
        """
    ),
    md(
        """
        ## Visualize The Raw Hourly Data

        Since the CSV is already hourly, the raw view focuses on:

        - full-history daily totals
        - a one-week high-traffic window of hourly rows
        - the empirical average count by hour of day
        """
    ),
    code(
        """
        preview_hours = 7 * 24
        rolling_hourly_sum = np.convolve(y_all_np, np.ones(preview_hours, dtype=np.float64), mode="valid")
        preview_start = int(np.argmax(rolling_hourly_sum))
        preview_slice = slice(preview_start, preview_start + preview_hours)

        df_preview = df.iloc[preview_slice].copy()
        date_preview = df_preview["Date"].to_numpy()
        count_preview = df_preview[TOTAL_COL].to_numpy()

        print(f"Preview window: {df_preview['Date'].iloc[0]} to {df_preview['Date'].iloc[-1]}")
        print(f"Rows in preview window: {df_preview.shape[0]}")
        print(f"Mean count in preview window: {count_preview.mean():.2f}")
        print(f"Max count in preview window: {count_preview.max():.0f}")

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True)

        axes[0].plot(daily_totals.index, daily_totals.values, color="#1982c4", linewidth=1.0)
        axes[0].set_title("Daily Total Bicycle Counts Across The Full Dataset")
        axes[0].set_ylabel("Bikes / day")
        axes[0].xaxis.set_major_locator(mdates.YearLocator())
        axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        axes[1].bar(
            df_preview["Date"],
            count_preview,
            width=1.0 / 30.0,
            color="#6a4c93",
            edgecolor="black",
            linewidth=0.35,
        )
        axes[1].set_title("Hourly Rows In A High-Traffic One-Week Window")
        axes[1].set_ylabel("Bikes / hour")
        axes[1].xaxis.set_major_locator(mdates.DayLocator(interval=1))
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

        axes[2].plot(hour_stats["hour"], hour_stats["mean"], marker="o", color="#ff595e", label="Empirical mean")
        axes[2].plot(hour_stats["hour"], hour_stats["median"], marker="s", color="#8ac926", label="Empirical median")
        axes[2].fill_between(
            hour_stats["hour"],
            np.maximum(hour_stats["mean"] - hour_stats["std"], 0.0),
            hour_stats["mean"] + hour_stats["std"],
            color="#1982c4",
            alpha=0.15,
            label="Mean ± 1 std",
        )
        axes[2].set_title("Raw Counts Summarized By Hour Of Day")
        axes[2].set_xlabel("Hour of day")
        axes[2].set_ylabel("Bikes / hour")
        axes[2].set_xticks(np.arange(0, 24, 2))
        axes[2].legend(loc="best")
        """
    ),
    md(
        """
        ## Fit `PolyagammaGPNegativeBinomialRegressor` With Learned `r`

        This is a deliberately simple 1D setup: the GP only sees hour of day,
        so it learns a shared hourly profile across the full history.
        """
    ),
    code(
        """
        reg = PolyagammaGPNegativeBinomialRegressor(
            total_count=20.0,
            learn_total_count=True,
            total_count_lr=0.05,
            total_count_update_frequency=1,
            total_count_quadrature_nodes=16,
            lengthscale_init=0.20,
            variance_init=1.0,
            max_iter=20,
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
        ## Compare The Learned Hourly Curve Against The Raw Data

        We evaluate the posterior mean count curve on a dense hour grid, then
        compare it with the empirical hourly averages from the raw dataset.
        """
    ),
    code(
        """
        latent_grid = reg.decision_function(X_grid_np)
        var_grid = reg.predictive_variance(X_grid_np)
        mean_count_grid = reg.total_count_ * np.exp(latent_grid + 0.5 * var_grid)

        rng = np.random.default_rng(0)
        sample_size = min(5000, df.shape[0])
        sample_idx = np.sort(rng.choice(df.shape[0], size=sample_size, replace=False))
        hour_sample = hour_raw_np[sample_idx, 0]
        count_sample = y_all_np[sample_idx]

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
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)

        axes[0].scatter(hour_sample, count_sample, s=8, alpha=0.10, color="0.35", label="Hourly rows")
        axes[0].plot(hour_stats["hour"], hour_stats["mean"], marker="o", color="#ff595e", linewidth=2, label="Empirical mean")
        axes[0].plot(hour_grid, mean_count_grid, color="#1982c4", linewidth=2.5, label="PG NB GP mean")
        axes[0].set_title("Hourly Count Curve")
        axes[0].set_xlabel("Hour of day")
        axes[0].set_ylabel("Bikes / hour")
        axes[0].set_xticks(np.arange(0, 24, 2))
        axes[0].legend(loc="best")

        axes[1].plot(hour_grid, latent_grid, color="#6a4c93", linewidth=2, label="Latent mean")
        axes[1].plot(hour_grid, var_grid, color="#8ac926", linewidth=2, label="Latent variance")
        axes[1].set_title("Latent GP Diagnostics")
        axes[1].set_xlabel("Hour of day")
        axes[1].set_ylabel("Value")
        axes[1].set_xticks(np.arange(0, 24, 2))
        axes[1].legend(loc="best")
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
        axes[1].axhline(20.0, color="black", linestyle="--", linewidth=1.25, label="Initial r")
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
            "n_rows": int(df.shape[0]),
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
