from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "slc17a7_pg_classifier_demo.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


cells = [
    md(
        """
        # SLC17A7 Gene Expression With The New `PolyagammaGPClassifier`

        This notebook recreates the transcriptomics classification figure from the
        older EFGP workflow, but it uses the updated library estimator in
        [`pg_classifier.py`](./pg_classifier.py):

        - inputs are the normalized 2D bead coordinates from `../x.pt`
        - labels come from `../y_slc17a7.pt`
        - expression is binarized the same way as the earlier notebook: `log1p(count) > 0`
        - the classifier fit is done with `PolyagammaGPClassifier`

        The micron-scale plotting coordinates come from `../adata_spatial.pt`, so
        the final figure lands back on the original tissue geometry instead of the
        normalized `[-1, 1]^2` training domain.

        This notebook was smoke-tested with the repo-local virtualenv at
        `../venv/bin/python`. The workspace's default Anaconda Python crashed on
        this real-data fit.
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
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
        from matplotlib.font_manager import FontProperties
        from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

        ROOT = Path.cwd().resolve()
        PARENT = ROOT.parent
        if str(PARENT) not in sys.path:
            sys.path.append(str(PARENT))

        from pg_classifier import PolyagammaGPClassifier

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
        ## Load The Transcriptomics Dataset

        The fit uses the same normalized coordinates as the old classification
        notebook, but the visualization uses the saved micron coordinates from the
        AnnData object.
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

        expression_threshold = 1e-3
        y_train_np = (y_log_np > expression_threshold).astype(np.int64)

        print(f"Training points: {X_train_np.shape[0]:,}")
        print(f"Input dimension: {X_train_np.shape[1]}")
        print(f"Positive class fraction: {y_train_np.mean():.4f}")
        print(f"Positive spots: {int(y_train_np.sum()):,}")
        print(f"Normalized x bounds: min={X_train_np.min(axis=0)}, max={X_train_np.max(axis=0)}")
        print(f"Spatial bounds (µm): min={spatial_xy.min(axis=0)}, max={spatial_xy.max(axis=0)}")
        print(f"Raw log1p expression range: [{y_log_np.min():.4f}, {y_log_np.max():.4f}]")
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


        def plot_raw_expression_status(ax, xy, labels):
            cmap = ListedColormap(["#445a9c", "#f2ef00"])
            norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
            ax.scatter(
                xy[:, 0],
                xy[:, 1],
                c=labels,
                cmap=cmap,
                norm=norm,
                s=8,
                alpha=0.9,
                edgecolors="none",
            )
            style_spatial_axis(ax)
            add_spatial_annotations(ax, xy)
            return ScalarMappable(norm=norm, cmap=cmap)


        def plot_posterior_mean(ax, xy, posterior_prob):
            norm = Normalize(
                vmin=np.percentile(posterior_prob, 2),
                vmax=np.percentile(posterior_prob, 98),
            )
            scatter = ax.scatter(
                xy[:, 0],
                xy[:, 1],
                c=posterior_prob,
                norm=norm,
                cmap="viridis",
                s=12,
                alpha=0.85,
                edgecolors="none",
            )
            style_spatial_axis(ax)
            add_spatial_annotations(ax, xy)
            return scatter
        """
    ),
    md(
        """
        ## Raw Binary Expression Status

        This mirrors the left panel from the earlier transcriptomics slide: each
        spot is marked as expressed vs not expressed after the same thresholding
        step used in the old notebook.
        """
    ),
    code(
        """
        fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
        sm = plot_raw_expression_status(ax, spatial_xy, y_train_np)
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1])
        cbar.ax.set_yticklabels(["not expressed", "expressed"])
        cbar.set_label("Gene status", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## Fit The Updated `PolyagammaGPClassifier`

        This uses the consolidated estimator path from `pg_classifier.py`:

        - full-batch PG E-step
        - feature-space M-step
        - exact weighted Toeplitz training operator
        - current sklearn-style prediction API

        The default iteration count matches the older slide fit.
        """
    ),
    code(
        """
        fit_device = "cuda" if torch.cuda.is_available() else "cpu"

        clf = PolyagammaGPClassifier(
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
        clf.fit(X_train_np, y_train_np)
        fit_time = time.time() - t0

        posterior_prob = clf.predict_proba(X_train_np)[:, 1]
        latent_mean = clf.decision_function(X_train_np)
        latent_var_raw = clf.predictive_variance(X_train_np)
        latent_var = np.clip(latent_var_raw, 0.0, None)
        n_negative_var = int(np.sum(latent_var_raw < 0.0))

        print(f"Fit device: {fit_device}")
        print(f"Fit time: {fit_time:.2f} s")
        print(f"Learned lengthscale: {clf.lengthscale_:.4f}")
        print(f"Learned variance: {clf.variance_:.4f}")
        print(f"Approx training accuracy: {clf.training_accuracy_:.4f}")
        print(
            "Posterior probability range: "
            f"[{posterior_prob.min():.4f}, {posterior_prob.max():.4f}]"
        )
        if n_negative_var:
            print(f"Clipped {n_negative_var} slightly negative predictive variances to zero for reporting.")
        print(
            "Latent variance range: "
            f"[{latent_var.min():.4f}, {latent_var.max():.4f}]"
        )
        """
    ),
    md(
        """
        ## Posterior Mean On The Original Tissue Coordinates

        This is the updated replacement for the older `SLC17A7_Classifier.png`
        figure: same dataset, same micron-space coordinates, but fit with the new
        `PolyagammaGPClassifier` machinery.
        """
    ),
    code(
        """
        fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
        scatter = plot_posterior_mean(ax, spatial_xy, posterior_prob)
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Posterior mean", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## Slide-Style Comparison Figure

        This side-by-side layout makes it easy to compare the raw expression map
        with the fitted posterior mean from the updated classifier.
        """
    ),
    code(
        """
        fig, axes = plt.subplots(1, 2, figsize=(13, 6), dpi=150, constrained_layout=True)

        sm = plot_raw_expression_status(axes[0], spatial_xy, y_train_np)
        cbar0 = fig.colorbar(sm, ax=axes[0], fraction=0.046, pad=0.04, ticks=[0, 1])
        cbar0.ax.set_yticklabels(["not expressed", "expressed"])
        cbar0.set_label("Gene status", fontsize=11, fontweight="bold")
        axes[0].set_title("Raw SLC17A7 status", fontsize=13, fontweight="bold")

        scatter = plot_posterior_mean(axes[1], spatial_xy, posterior_prob)
        cbar1 = fig.colorbar(scatter, ax=axes[1], fraction=0.046, pad=0.04)
        cbar1.set_label("Posterior mean", fontsize=11, fontweight="bold")
        axes[1].set_title("Updated PG classifier posterior mean", fontsize=13, fontweight="bold")

        plt.show()
        """
    ),
    md(
        """
        ## Fit Diagnostics

        The estimator stores one history record per outer iteration plus a final
        post-fit E-step record. The plots below expose the learned hyperparameter
        trajectory and the approximate training accuracy over the fit.
        """
    ),
    code(
        """
        history = clf.history_
        outer_iters = np.arange(len(history), dtype=np.int64)
        lengthscales = np.array([record["lengthscale"] for record in history], dtype=np.float64)
        variances = np.array([record["variance"] for record in history], dtype=np.float64)
        approx_acc = np.array([record["approx_accuracy"] for record in history], dtype=np.float64)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

        axes[0].plot(outer_iters, lengthscales, marker="o", label="lengthscale")
        axes[0].plot(outer_iters, variances, marker="s", label="variance")
        axes[0].set_xlabel("History record")
        axes[0].set_ylabel("Value")
        axes[0].set_title("Kernel hyperparameter path")
        axes[0].legend(loc="best")

        axes[1].plot(outer_iters, approx_acc, marker="o", color="#1982c4")
        axes[1].set_xlabel("History record")
        axes[1].set_ylabel("Approx accuracy")
        axes[1].set_title("Approximate training accuracy")
        axes[1].set_ylim(0.0, 1.0)

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
