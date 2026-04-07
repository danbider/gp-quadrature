from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "slc17a7_pg_nb_demo.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


cells = [
    md(
        """
        # SLC17A7 Gene Expression With The New `PolyagammaGPNegativeBinomialRegressor`

        The earlier transcriptomics classification workflow threw away count
        information by thresholding `y_slc17a7.pt` into expressed vs not expressed.
        This notebook keeps the count structure instead:

        - inputs are the normalized 2D bead coordinates from `../x.pt`
        - `../y_slc17a7.pt` stores `log1p(raw_count)`
        - we recover integer counts with `round(expm1(y_log))`
        - the fit uses `PolyagammaGPNegativeBinomialRegressor`

        The plotting coordinates still come from `../adata_spatial.pt`, so the
        final maps land back on the original tissue geometry in microns.

        This notebook was smoke-tested with the repo-local virtualenv at
        `../venv/bin/python`. The workspace's default Anaconda Python crashed on
        the heavier real-data fits in this stack.
        """
    ),
    code(
        """
        import os
        import sys
        import time
        from pathlib import Path

        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import numpy as np
        import torch
        from matplotlib import patheffects as pe
        from matplotlib.colors import Normalize
        from matplotlib.font_manager import FontProperties
        from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

        ROOT = Path.cwd().resolve()
        PARENT = ROOT.parent
        if str(PARENT) not in sys.path:
            sys.path.append(str(PARENT))

        from pg_classifier import PolyagammaGPNegativeBinomialRegressor

        plt.style.use("seaborn-v0_8-white")
        mpl.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.sans-serif": ["DejaVu Sans"],
                "axes.linewidth": 1.2,
                "axes.labelsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
            }
        )
        np.set_printoptions(suppress=True, precision=4)
        torch.set_default_dtype(torch.float64)
        """
    ),
    md(
        """
        ## Load The Saved SLC17A7 Tensor And Recover Counts

        The saved target tensor is in log-count space. Since all unique values are
        exact `log1p(k)` values, we can safely invert it back to integer counts
        before fitting the negative-binomial model.
        """
    ),
    code(
        """
        X_train_np = torch.load(PARENT / "x.pt", map_location="cpu").numpy().astype(np.float64)
        y_log_np = torch.load(PARENT / "y_slc17a7.pt", map_location="cpu").numpy().astype(np.float64)
        spatial_xy = np.asarray(
            torch.load(PARENT / "adata_spatial.pt", map_location="cpu", weights_only=False),
            dtype=np.float64,
        )

        recovered_counts = np.rint(np.expm1(y_log_np)).astype(np.int64)
        reconstruction_ok = np.allclose(np.expm1(y_log_np), recovered_counts, atol=1e-5)

        print(f"Training points: {X_train_np.shape[0]:,}")
        print(f"Input dimension: {X_train_np.shape[1]}")
        print(f"Recovered counts are integer-consistent: {reconstruction_ok}")
        print(f"Count range: [{recovered_counts.min()}, {recovered_counts.max()}]")
        print(f"Mean count: {recovered_counts.mean():.4f}")
        print(f"Count variance: {recovered_counts.var():.4f}")
        print(f"Fraction of zero counts: {np.mean(recovered_counts == 0):.4f}")
        print(f"Normalized x bounds: min={X_train_np.min(axis=0)}, max={X_train_np.max(axis=0)}")
        print(f"Spatial bounds (µm): min={spatial_xy.min(axis=0)}, max={spatial_xy.max(axis=0)}")
        """
    ),
    code(
        """
        def style_spatial_axis(ax):
            ax.set_xlabel("x (µm)", fontweight="bold", labelpad=6)
            ax.set_ylabel("y (µm)", fontweight="bold", labelpad=6)
            ax.set_aspect("equal")
            ax.tick_params(length=4, width=1.2)


        def add_spatial_annotations(ax, xy):
            sb = AnchoredSizeBar(
                ax.transData,
                size=200,
                label="200 µm",
                loc="lower right",
                pad=0.5,
                borderpad=1,
                sep=6,
                frameon=False,
                size_vertical=4,
                fontproperties=FontProperties(size=10, weight="bold"),
            )
            sb.txt_label.set_path_effects([pe.withStroke(linewidth=2, foreground="black")])
            ax.add_artist(sb)

            arrow_start = (
                xy[:, 0].min() + 0.06 * np.ptp(xy[:, 0]),
                xy[:, 1].min() + 0.06 * np.ptp(xy[:, 1]),
            )
            arrow_end = (arrow_start[0], arrow_start[1] + 150)

            ax.annotate(
                "",
                xy=arrow_end,
                xytext=arrow_start,
                arrowprops=dict(
                    facecolor="white",
                    edgecolor="black",
                    linewidth=1.5,
                    headwidth=20,
                    headlength=30,
                    width=6,
                ),
            )
            ax.text(
                arrow_end[0] - 20,
                arrow_end[1] + 15,
                "Dorsal",
                color="white",
                fontsize=8,
                weight="bold",
                path_effects=[pe.withStroke(linewidth=1.4, foreground="black")],
            )


        def plot_expression_map(ax, xy, values, *, norm, title):
            scatter = ax.scatter(
                xy[:, 0],
                xy[:, 1],
                c=values,
                norm=norm,
                cmap="viridis",
                s=12,
                alpha=0.85,
                edgecolors="none",
            )
            style_spatial_axis(ax)
            add_spatial_annotations(ax, xy)
            ax.set_title(title, fontsize=13, fontweight="bold")
            return scatter
        """
    ),
    md(
        """
        ## Raw `log1p(count)` Map

        This is the count-preserving view of the saved tensor before fitting the
        negative-binomial regressor.
        """
    ),
    code(
        """
        raw_log_norm = Normalize(
            vmin=np.percentile(y_log_np, 2),
            vmax=np.percentile(y_log_np, 98),
        )

        fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
        scatter = plot_expression_map(
            ax,
            spatial_xy,
            y_log_np,
            norm=raw_log_norm,
            title="Raw SLC17A7 log1p(count)",
        )
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Expression level (log1p count)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## Fit The Updated Negative-Binomial PG Regressor

        This uses the count-model path in `pg_classifier.py`, including the
        learned negative-binomial shape parameter `r` (`total_count` in the code).
        """
    ),
    code(
        """
        fit_device = "cuda" if torch.cuda.is_available() else "cpu"
        init_total_count = 1.0

        reg = PolyagammaGPNegativeBinomialRegressor(
            total_count=init_total_count,
            learn_total_count=True,
            total_count_lr=0.05,
            total_count_update_frequency=1,
            total_count_quadrature_nodes=16,
            lengthscale_init=0.20,
            variance_init=1.0,
            max_iter=50,
            e_step_iters=1,
            final_e_step_iters=2,
            rho0=0.7,
            gamma=1e-3,
            lr=0.05,
            n_e_probes=4,
            n_m_probes=8,
            cg_tol=1e-6,
            nufft_eps=1e-7,
            spectral_eps=1e-4,
            trunc_eps=1e-4,
            prediction_batch_size=256,
            predictive_variance_method="chebyshev",
            predictive_variance_chebyshev_nodes=7,
            use_exact_weighted_toeplitz_operator=True,
            random_state=0,
            device=fit_device,
            store_history=True,
            verbose=0,
        )

        t0 = time.time()
        reg.fit(X_train_np, recovered_counts)
        fit_time = time.time() - t0

        mean_count = reg.predict_response_mean(X_train_np)
        latent_mean = reg.decision_function(X_train_np)
        latent_var_raw = reg.predictive_variance(X_train_np)
        latent_var = np.clip(latent_var_raw, 0.0, None)
        n_negative_var = int(np.sum(latent_var_raw < 0.0))
        log_mean_count = np.log1p(mean_count)

        print(f"Fit device: {fit_device}")
        print(f"Fit time: {fit_time:.2f} s")
        print(f"Learned lengthscale: {reg.lengthscale_:.4f}")
        print(f"Learned variance: {reg.variance_:.4f}")
        print(f"Learned total_count r: {reg.total_count_:.4f}")
        print(f"Training mean absolute error: {reg.training_mean_absolute_error_:.4f}")
        print(f"Predicted mean-count range: [{mean_count.min():.4f}, {mean_count.max():.4f}]")
        if n_negative_var:
            print(f"Clipped {n_negative_var} slightly negative predictive variances to zero for reporting.")
        print(f"Latent variance range: [{latent_var.min():.4f}, {latent_var.max():.4f}]")
        """
    ),
    md(
        """
        ## Predicted Mean Count Map

        To compare directly against the saved target tensor, the visualization
        below shows `log1p(predicted mean count)`.
        """
    ),
    code(
        """
        shared_log_norm = Normalize(
            vmin=min(np.percentile(y_log_np, 2), np.percentile(log_mean_count, 2)),
            vmax=max(np.percentile(y_log_np, 98), np.percentile(log_mean_count, 98)),
        )

        fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
        scatter = plot_expression_map(
            ax,
            spatial_xy,
            log_mean_count,
            norm=shared_log_norm,
            title="Updated PG NB fit: log1p(mean count)",
        )
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Posterior mean (log1p count)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## Raw Versus Fitted Count Map

        The panels below compare the observed `log1p(count)` field against the
        fitted `log1p(mean count)` from the negative-binomial model using a shared
        color scale.
        """
    ),
    code(
        """
        fig, axes = plt.subplots(1, 2, figsize=(13, 6), dpi=150, constrained_layout=True)

        scatter0 = plot_expression_map(
            axes[0],
            spatial_xy,
            y_log_np,
            norm=shared_log_norm,
            title="Raw SLC17A7 log1p(count)",
        )
        cbar0 = fig.colorbar(scatter0, ax=axes[0], fraction=0.046, pad=0.04)
        cbar0.set_label("log1p count", fontsize=11, fontweight="bold")

        scatter1 = plot_expression_map(
            axes[1],
            spatial_xy,
            log_mean_count,
            norm=shared_log_norm,
            title="Updated PG NB fit: log1p(mean count)",
        )
        cbar1 = fig.colorbar(scatter1, ax=axes[1], fraction=0.046, pad=0.04)
        cbar1.set_label("log1p mean count", fontsize=11, fontweight="bold")

        plt.show()
        """
    ),
    md(
        """
        ## Fit Diagnostics

        The regressor stores the learned kernel path, the evolving dispersion
        parameter, and the count-fit metric for each outer iteration.
        """
    ),
    code(
        """
        history = reg.history_
        outer_iters = np.arange(len(history), dtype=np.int64)
        lengthscales = np.array([record["lengthscale"] for record in history], dtype=np.float64)
        variances = np.array([record["variance"] for record in history], dtype=np.float64)
        total_counts = np.array([record["total_count"] for record in history], dtype=np.float64)
        mean_count_mae = np.array([record["mean_count_mae"] for record in history], dtype=np.float64)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

        axes[0].plot(outer_iters, lengthscales, marker="o", label="lengthscale")
        axes[0].plot(outer_iters, variances, marker="s", label="variance")
        axes[0].set_xlabel("History record")
        axes[0].set_ylabel("Value")
        axes[0].set_title("Kernel hyperparameter path")
        axes[0].legend(loc="best")

        axes[1].plot(outer_iters, total_counts, marker="o", color="#ff595e")
        axes[1].axhline(init_total_count, color="black", linestyle="--", linewidth=1.2)
        axes[1].set_xlabel("History record")
        axes[1].set_ylabel("total_count r")
        axes[1].set_title("Negative-binomial shape path")

        axes[2].plot(outer_iters, mean_count_mae, marker="o", color="#1982c4")
        axes[2].set_xlabel("History record")
        axes[2].set_ylabel("Mean-count MAE")
        axes[2].set_title("Training count error")

        plt.show()
        """
    ),
]


nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.13",
    },
}

NOTEBOOK_PATH.write_text(nbf.writes(nb))
print(f"Wrote {NOTEBOOK_PATH}")
