from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit, log_expit


def rbf_kernel(x: np.ndarray, variance: float, lengthscale: float) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    sqdist = (x - x.T) ** 2
    return variance * np.exp(-0.5 * sqdist / (lengthscale**2))


def normal_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    z = (x - mean) / std
    return np.exp(-0.5 * z**2) / (std * np.sqrt(2.0 * np.pi))


def pg_mean_from_latent(latent_value: float) -> float:
    abs_value = abs(float(latent_value))
    if abs_value < 1e-10:
        return 0.25
    return float(np.tanh(abs_value / 2.0) / (2.0 * abs_value))


def choose_synthetic_example(
    x_grid: np.ndarray,
    variance: float,
    lengthscale: float,
) -> dict[str, float | int | np.ndarray]:
    covariance = rbf_kernel(x_grid, variance=variance, lengthscale=lengthscale)
    covariance += 1e-9 * np.eye(x_grid.size)
    search_region = np.abs(x_grid) < 1.2

    for seed in range(5000):
        rng = np.random.default_rng(seed)
        f_curve = rng.multivariate_normal(np.zeros(x_grid.size), covariance)
        candidate_idx = np.where(search_region & (f_curve > 1.6))[0]
        if candidate_idx.size == 0:
            continue

        obs_idx = int(candidate_idx[np.argmax(f_curve[candidate_idx])])
        f_obs = float(f_curve[obs_idx])
        p_obs = float(expit(f_obs))
        y_obs = int(rng.binomial(1, p_obs))
        if y_obs == 1:
            return {
                "seed": seed,
                "obs_idx": obs_idx,
                "f_curve": f_curve,
                "f_obs": f_obs,
                "p_obs": p_obs,
                "y_obs": y_obs,
            }

    raise RuntimeError("Could not find a simple synthetic GP classification example.")


def scalar_exact_posterior_density(
    f_grid: np.ndarray,
    *,
    prior_variance: float,
    y_obs: int,
) -> np.ndarray:
    prior = normal_pdf(f_grid, mean=0.0, std=np.sqrt(prior_variance))
    if y_obs == 1:
        log_likelihood = log_expit(f_grid)
    else:
        log_likelihood = log_expit(-f_grid)
    unnormalized = prior * np.exp(log_likelihood)
    return unnormalized / np.trapezoid(unnormalized, f_grid)


def make_figure(output_stem: Path) -> None:
    variance = 12.0
    lengthscale = 1.0
    x_grid = np.linspace(-2.7, 2.7, 450)

    demo = choose_synthetic_example(x_grid, variance=variance, lengthscale=lengthscale)
    x_obs = float(x_grid[int(demo["obs_idx"])])
    f_curve = np.asarray(demo["f_curve"])
    f_obs = float(demo["f_obs"])
    p_obs = float(demo["p_obs"])
    y_obs = int(demo["y_obs"])

    prior_variance = variance
    kappa = y_obs - 0.5
    omega = pg_mean_from_latent(f_obs)
    conditional_variance = 1.0 / (1.0 / prior_variance + omega)
    conditional_mean = conditional_variance * kappa

    density_extent = 3.2 * np.sqrt(prior_variance)
    f_density_grid = np.linspace(-density_extent, density_extent, 1600)
    prior_density = normal_pdf(f_density_grid, mean=0.0, std=np.sqrt(prior_variance))
    exact_posterior = scalar_exact_posterior_density(
        f_density_grid,
        prior_variance=prior_variance,
        y_obs=y_obs,
    )
    conditional_density = normal_pdf(
        f_density_grid,
        mean=conditional_mean,
        std=np.sqrt(conditional_variance),
    )
    exact_mean = np.trapezoid(f_density_grid * exact_posterior, f_density_grid)
    exact_var = np.trapezoid((f_density_grid - exact_mean) ** 2 * exact_posterior, f_density_grid)
    matched_gaussian = normal_pdf(f_density_grid, mean=exact_mean, std=np.sqrt(exact_var))
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "figure.titlesize": 20,
            "legend.fontsize": 12,
        }
    )

    fig = plt.figure(figsize=(13.5, 5.8))
    gs = fig.add_gridspec(1, 2, wspace=0.28)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1], sharey=ax_left)

    prior_color = "#7c7c7c"
    nonconj_color = "#2563eb"
    conj_color = "#1f8a5b"

    y_max = 1.12 * max(prior_density.max(), exact_posterior.max(), conditional_density.max())

    ax_left.plot(f_density_grid, prior_density, color=prior_color, lw=2.0, ls="--", label="Gaussian prior")
    ax_left.plot(f_density_grid, exact_posterior, color=nonconj_color, lw=3.0, label="exact posterior")
    ax_left.plot(
        f_density_grid,
        matched_gaussian,
        color="#111111",
        lw=1.8,
        ls=":",
        label="matched Gaussian",
    )
    ax_left.fill_between(f_density_grid, 0.0, exact_posterior, color=nonconj_color, alpha=0.14)
    ax_left.set_xlim(f_density_grid.min(), f_density_grid.max())
    ax_left.set_ylim(0.0, y_max)
    ax_left.set_xlabel("$f_*$")
    ax_left.set_ylabel("density")
    ax_left.set_title("No PG", pad=14)
    ax_left.legend(loc="upper left", frameon=True)

    ax_right.plot(f_density_grid, prior_density, color=prior_color, lw=2.0, ls="--", label="Gaussian prior")
    ax_right.plot(f_density_grid, conditional_density, color=conj_color, lw=3.0, label="conditional posterior")
    ax_right.fill_between(f_density_grid, 0.0, conditional_density, color=conj_color, alpha=0.14)
    ax_right.set_xlim(f_density_grid.min(), f_density_grid.max())
    ax_right.set_ylim(0.0, y_max)
    ax_right.set_xlabel("$f_*$")
    ax_right.set_title("Given omega", pad=14)
    ax_right.legend(loc="upper left", frameon=True)

    fig.suptitle("Conditional Conjugacy", y=0.97)
    fig.subplots_adjust(top=0.80, left=0.07, right=0.98, bottom=0.14)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=220, pad_inches=0.2)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a simple presentation figure for PG conditional conjugacy.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pg_conditional_conjugacy_demo"),
        help="Output stem for the exported figure.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_figure(args.output)
