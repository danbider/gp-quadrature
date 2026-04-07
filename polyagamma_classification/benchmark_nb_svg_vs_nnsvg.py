from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.optimize as opt
import scipy.sparse as sp
import scipy.special as sps
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.distributions import NegativeBinomial

from pg_classifier import PolyagammaGPNegativeBinomialRegressor

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent

if str(REPO_ROOT) not in __import__("sys").path:
    __import__("sys").path.append(str(REPO_ROOT))

from vanilla_gp_sampling import sample_gp_spectral_approx


@dataclass
class MouseSpatialData:
    counts: sp.csr_matrix
    genes: pd.Series
    coords_raw: np.ndarray
    coords_norm: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the NB PG spatial model against nnSVG on synthetic genes placed on real coordinates."
    )
    parser.add_argument("--n-spots", type=int, default=1200)
    parser.add_argument("--n-genes", type=int, default=12)
    parser.add_argument("--svg-fraction", type=float, default=0.33)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=12)
    parser.add_argument("--e-step-iters", type=int, default=1)
    parser.add_argument("--final-e-step-iters", type=int, default=2)
    parser.add_argument("--n-e-probes", type=int, default=3)
    parser.add_argument("--n-m-probes", type=int, default=6)
    parser.add_argument("--prediction-batch-size", type=int, default=128)
    parser.add_argument("--install-r-packages", action="store_true")
    parser.add_argument("--skip-nnsvg", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "svg_synthetic_benchmark")
    return parser.parse_args()


def load_mouse_spatial_data() -> MouseSpatialData:
    counts = sio.mmread("/Users/colecitrenbaum/counts.mtx").tocsr()
    genes = pd.read_csv("/Users/colecitrenbaum/genes.tsv", header=None, sep="\t")[0].astype(str)
    barcodes = pd.read_csv("/Users/colecitrenbaum/barcodes.tsv", header=None, sep="\t")[0].astype(str)
    loc = (
        pd.read_csv("/Users/colecitrenbaum/location.tsv", sep="\t")
        .set_index("barcode")
        .loc[barcodes]
    )
    coords_raw = loc[["x", "y"]].to_numpy(dtype=np.float64)
    mins = coords_raw.min(axis=0)
    maxs = coords_raw.max(axis=0)
    span = np.where(maxs > mins, maxs - mins, 1.0)
    coords_norm = 2.0 * (coords_raw - mins) / span - 1.0
    return MouseSpatialData(
        counts=counts,
        genes=genes,
        coords_raw=coords_raw,
        coords_norm=coords_norm,
    )


def compute_gene_statistics(counts: sp.csr_matrix) -> pd.DataFrame:
    n_spots = counts.shape[1]
    mean = np.asarray(counts.mean(axis=1)).ravel()
    sq_mean = np.asarray(counts.power(2).mean(axis=1)).ravel()
    var = np.maximum(sq_mean - mean**2, 0.0)
    nonzero_frac = counts.getnnz(axis=1) / float(n_spots)
    overdisp = np.maximum(var - mean, 1e-8)
    size_mom = np.clip((mean**2) / overdisp, 0.1, 100.0)
    size_mom[var <= mean + 1e-8] = 100.0
    return pd.DataFrame(
        {
            "mean_count": mean,
            "var_count": var,
            "nonzero_frac": nonzero_frac,
            "size_mom": size_mom,
        }
    )


def sample_spot_subset(coords_norm: np.ndarray, n_spots: int, seed: int) -> np.ndarray:
    n_total = coords_norm.shape[0]
    if n_spots >= n_total:
        return np.arange(n_total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, size=n_spots, replace=False))


def checkerboard_holdout(coords_norm: np.ndarray, n_bins: int = 6) -> np.ndarray:
    scaled = (coords_norm + 1.0) / 2.0
    scaled = np.clip(scaled, 0.0, np.nextafter(1.0, 0.0))
    bins = np.floor(scaled * n_bins).astype(np.int64)
    return ((bins[:, 0] + bins[:, 1]) % 2) == 0


def nb_logpmf(y: np.ndarray, mu: np.ndarray, total_count: float) -> np.ndarray:
    mu_safe = np.clip(mu.astype(np.float64), 1e-10, None)
    r = float(max(total_count, 1e-10))
    y = y.astype(np.float64)
    return (
        sps.gammaln(y + r)
        - sps.gammaln(r)
        - sps.gammaln(y + 1.0)
        + r * (math.log(r) - np.log(r + mu_safe))
        + y * (np.log(mu_safe) - np.log(r + mu_safe))
    )


def fit_intercept_only_nb(y_train: np.ndarray) -> tuple[float, float]:
    mu_hat = float(max(np.mean(y_train), 1e-8))
    var_hat = float(np.var(y_train))
    if var_hat <= mu_hat + 1e-6:
        return mu_hat, 100.0

    def objective(log_r: float) -> float:
        r = math.exp(log_r)
        return -float(nb_logpmf(y_train, np.full_like(y_train, mu_hat, dtype=np.float64), r).sum())

    result = opt.minimize_scalar(objective, bounds=(-4.0, 6.0), method="bounded")
    if not result.success:
        r_hat = max((mu_hat**2) / max(var_hat - mu_hat, 1e-6), 0.1)
    else:
        r_hat = math.exp(float(result.x))
    return mu_hat, float(np.clip(r_hat, 0.1, 100.0))


def simulate_synthetic_panel(
    *,
    mouse: MouseSpatialData,
    gene_stats: pd.DataFrame,
    subset_idx: np.ndarray,
    n_genes: int,
    svg_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    template_pool = gene_stats.index[
        (gene_stats["mean_count"] >= 0.05)
        & (gene_stats["mean_count"] <= 2.0)
        & (gene_stats["nonzero_frac"] >= 0.02)
        & (gene_stats["nonzero_frac"] <= 0.7)
    ].to_numpy()
    if template_pool.size < n_genes:
        raise RuntimeError("Template pool is too small for the requested benchmark size.")

    template_indices = rng.choice(template_pool, size=n_genes, replace=False)
    truth = np.zeros(n_genes, dtype=bool)
    truth[: max(1, int(round(n_genes * svg_fraction)))] = True
    rng.shuffle(truth)

    effect_grid = np.array([0.5, 0.9, 1.3], dtype=np.float64)
    lengthscale_grid = np.array([0.08, 0.16, 0.28], dtype=np.float64)

    x_subset = mouse.coords_norm[subset_idx]
    x_subset_t = torch.as_tensor(x_subset, dtype=torch.float64)

    panel_meta: list[dict[str, object]] = []
    panel_counts = np.zeros((n_genes, subset_idx.size), dtype=np.int64)

    for gene_pos, (template_idx, is_svg) in enumerate(zip(template_indices, truth, strict=True)):
        template_name = str(mouse.genes.iloc[int(template_idx)])
        template_mu = float(gene_stats.loc[int(template_idx), "mean_count"])
        template_r = float(gene_stats.loc[int(template_idx), "size_mom"])
        effect_size = float(rng.choice(effect_grid)) if is_svg else 0.0
        lengthscale = float(rng.choice(lengthscale_grid)) if is_svg else np.nan

        if is_svg:
            latent = sample_gp_spectral_approx(
                x_subset_t,
                num_samples=1,
                length_scale=lengthscale,
                variance=1.0,
                spectral_eps=1e-4,
                trunc_eps=1e-4,
                nufft_eps=1e-7,
                seed=seed + 1000 + gene_pos,
            ).detach().cpu().numpy()
            latent = (latent - latent.mean()) / (latent.std() + 1e-8)
            intercept = math.log(max(template_mu, 1e-8) / template_r) - math.log(np.mean(np.exp(effect_size * latent)))
            logits = intercept + effect_size * latent
        else:
            latent = np.zeros(subset_idx.size, dtype=np.float64)
            intercept = math.log(max(template_mu, 1e-8) / template_r)
            logits = np.full(subset_idx.size, intercept, dtype=np.float64)

        counts = NegativeBinomial(
            total_count=torch.tensor(template_r, dtype=torch.float64),
            logits=torch.as_tensor(logits, dtype=torch.float64),
        ).sample().cpu().numpy().astype(np.int64)

        gene_id = f"sim_gene_{gene_pos:03d}"
        panel_meta.append(
            {
                "gene_id": gene_id,
                "template_gene": template_name,
                "template_index": int(template_idx),
                "template_mean_count": template_mu,
                "template_size": template_r,
                "is_svg": bool(is_svg),
                "effect_size": effect_size,
                "lengthscale": lengthscale,
            }
        )
        panel_counts[gene_pos] = counts

    return pd.DataFrame(panel_meta), panel_counts


def run_our_model(
    *,
    x_norm: np.ndarray,
    gene_meta: pd.DataFrame,
    counts: np.ndarray,
    test_mask: np.ndarray,
    args: argparse.Namespace,
) -> pd.DataFrame:
    train_mask = ~test_mask
    x_train = x_norm[train_mask]
    x_test = x_norm[test_mask]

    results: list[dict[str, float | str | bool]] = []
    for gene_pos, row in gene_meta.iterrows():
        y = counts[gene_pos].astype(np.float64)
        y_train = y[train_mask]
        y_test = y[test_mask]

        mu_null, r_null = fit_intercept_only_nb(y_train)
        null_train_ll = float(nb_logpmf(y_train, np.full_like(y_train, mu_null), r_null).sum())
        null_test_ll = float(nb_logpmf(y_test, np.full_like(y_test, mu_null), r_null).sum())

        init_total_count = float(np.clip(r_null, 0.2, 10.0))
        reg = PolyagammaGPNegativeBinomialRegressor(
            total_count=init_total_count,
            learn_total_count=True,
            total_count_lr=0.05,
            total_count_update_frequency=1,
            total_count_quadrature_nodes=16,
            lengthscale_init=0.20,
            variance_init=1.0,
            max_iter=args.max_iter,
            e_step_iters=args.e_step_iters,
            final_e_step_iters=args.final_e_step_iters,
            rho0=0.7,
            gamma=1e-3,
            lr=0.05,
            n_e_probes=args.n_e_probes,
            n_m_probes=args.n_m_probes,
            cg_tol=1e-6,
            nufft_eps=1e-7,
            spectral_eps=1e-4,
            trunc_eps=1e-4,
            prediction_batch_size=args.prediction_batch_size,
            predictive_variance_method="chebyshev",
            predictive_variance_chebyshev_nodes=5,
            use_exact_weighted_toeplitz_operator=True,
            random_state=args.seed + gene_pos,
            device="cpu",
            store_history=True,
            verbose=0,
        )

        t0 = time.time()
        reg.fit(x_train, y_train)
        runtime_sec = time.time() - t0

        mu_train = np.clip(reg.predict_response_mean(x_train), 1e-10, None)
        spatial_train_ll = float(nb_logpmf(y_train, mu_train, reg.total_count_).sum())
        latent_test = reg.decision_function(x_test)
        variance_test = np.clip(reg.predictive_variance(x_test), 0.0, None)
        mu_test = reg.total_count_ * np.exp(latent_test + 0.5 * variance_test)
        spatial_test_ll = float(nb_logpmf(y_test, mu_test, reg.total_count_).sum())

        results.append(
            {
                "gene_id": row["gene_id"],
                "our_score_holdout": spatial_test_ll - null_test_ll,
                "our_score_train": spatial_train_ll - null_train_ll,
                "our_lr_like_stat": 2.0 * (spatial_train_ll - null_train_ll),
                "our_null_train_loglik": null_train_ll,
                "our_spatial_train_loglik": spatial_train_ll,
                "our_null_test_loglik": null_test_ll,
                "our_spatial_test_loglik": spatial_test_ll,
                "our_runtime_sec": runtime_sec,
                "our_total_count": float(reg.total_count_),
                "our_lengthscale": float(reg.lengthscale_),
                "our_variance": float(reg.variance_),
                "our_null_mu": mu_null,
                "our_null_size": r_null,
                "test_mean_count_pred": float(np.mean(mu_test)),
            }
        )

    return pd.DataFrame(results)


def save_simulation_dataset(
    *,
    output_dir: Path,
    subset_idx: np.ndarray,
    mouse: MouseSpatialData,
    gene_meta: pd.DataFrame,
    counts: np.ndarray,
    test_mask: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    spot_ids = np.array([f"spot_{i:04d}" for i in range(subset_idx.size)], dtype=object)
    spots_df = pd.DataFrame(
        {
            "spot_id": spot_ids,
            "original_index": subset_idx,
            "x": mouse.coords_raw[subset_idx, 0],
            "y": mouse.coords_raw[subset_idx, 1],
            "x_norm": mouse.coords_norm[subset_idx, 0],
            "y_norm": mouse.coords_norm[subset_idx, 1],
            "is_test": test_mask.astype(int),
        }
    )
    spots_df.to_csv(output_dir / "spots.csv", index=False)
    gene_meta.to_csv(output_dir / "gene_metadata.csv", index=False)
    counts_df = pd.DataFrame(counts, index=gene_meta["gene_id"], columns=spot_ids)
    counts_df.to_csv(output_dir / "sim_counts.csv")


def run_nnsvg(output_dir: Path, install_r_packages: bool) -> pd.DataFrame:
    nnsvg_out = output_dir / "nnsvg_results.csv"
    cmd = [
        "Rscript",
        str(ROOT / "run_nnsvg_synthetic_benchmark.R"),
        str(output_dir),
        str(nnsvg_out),
        "true" if install_r_packages else "false",
    ]
    subprocess.run(cmd, check=True)
    return pd.read_csv(nnsvg_out)


def compute_metrics(scores: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    metrics = {
        "auroc": float(roc_auc_score(truth, scores)),
        "auprc": float(average_precision_score(truth, scores)),
    }
    k = int(np.sum(truth))
    if k > 0:
        topk_idx = np.argsort(scores)[::-1][:k]
        metrics["precision_at_k"] = float(np.mean(truth[topk_idx]))
    else:
        metrics["precision_at_k"] = float("nan")
    return metrics


def make_plots(results: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    colors = np.where(results["is_svg"], "#d62728", "#1f77b4")
    axes[0].scatter(results["nnsvg_LR_stat"], results["our_lr_like_stat"], c=colors, s=60)
    axes[0].set_xlabel("nnSVG LR_stat")
    axes[0].set_ylabel("Our train LR-like stat")
    axes[0].set_title("nnSVG vs our train score")

    order = np.argsort(results["our_score_holdout"].to_numpy())[::-1]
    ranked = results.iloc[order].reset_index(drop=True)
    axes[1].bar(
        np.arange(ranked.shape[0]),
        ranked["our_score_holdout"],
        color=np.where(ranked["is_svg"], "#d62728", "#1f77b4"),
    )
    axes[1].set_xlabel("Genes ranked by our holdout score")
    axes[1].set_ylabel("Our held-out NB gain")
    axes[1].set_title("Our holdout ranking")

    fig.savefig(output_dir / "score_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    mouse = load_mouse_spatial_data()
    gene_stats = compute_gene_statistics(mouse.counts)

    subset_idx = sample_spot_subset(mouse.coords_norm, args.n_spots, args.seed)
    subset_norm = mouse.coords_norm[subset_idx]
    test_mask = checkerboard_holdout(subset_norm)
    if np.sum(test_mask) == 0 or np.sum(~test_mask) == 0:
        raise RuntimeError("Holdout split produced an empty train or test partition.")

    gene_meta, sim_counts = simulate_synthetic_panel(
        mouse=mouse,
        gene_stats=gene_stats,
        subset_idx=subset_idx,
        n_genes=args.n_genes,
        svg_fraction=args.svg_fraction,
        seed=args.seed,
    )

    output_dir = args.output_dir / f"pilot_seed_{args.seed}_spots_{args.n_spots}_genes_{args.n_genes}"
    save_simulation_dataset(
        output_dir=output_dir,
        subset_idx=subset_idx,
        mouse=mouse,
        gene_meta=gene_meta,
        counts=sim_counts,
        test_mask=test_mask,
    )

    our_results = run_our_model(
        x_norm=subset_norm,
        gene_meta=gene_meta,
        counts=sim_counts,
        test_mask=test_mask,
        args=args,
    )
    our_results.to_csv(output_dir / "our_model_results.csv", index=False)

    merged = gene_meta.merge(our_results, on="gene_id", how="left")
    summary: dict[str, object] = {
        "seed": args.seed,
        "n_spots": int(args.n_spots),
        "n_genes": int(args.n_genes),
        "n_svg": int(np.sum(gene_meta["is_svg"])),
        "n_train_spots": int(np.sum(~test_mask)),
        "n_test_spots": int(np.sum(test_mask)),
    }

    our_holdout_metrics = compute_metrics(merged["our_score_holdout"].to_numpy(), merged["is_svg"].to_numpy())
    our_train_metrics = compute_metrics(merged["our_lr_like_stat"].to_numpy(), merged["is_svg"].to_numpy())
    summary["our_holdout_metrics"] = our_holdout_metrics
    summary["our_train_lr_metrics"] = our_train_metrics

    if args.skip_nnsvg:
        merged.to_csv(output_dir / "benchmark_results.csv", index=False)
        with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))
        print(f"Wrote benchmark outputs to {output_dir}")
        return

    nnsvg_results = run_nnsvg(output_dir, install_r_packages=args.install_r_packages)
    nnsvg_results = nnsvg_results.rename(
        columns={
            "LR_stat": "nnsvg_LR_stat",
            "pval": "nnsvg_pval",
            "padj": "nnsvg_padj",
            "prop_sv": "nnsvg_prop_sv",
        }
    )
    merged = merged.merge(
        nnsvg_results[
            [col for col in nnsvg_results.columns if col.startswith("nnsvg_")] + ["gene_id"]
        ],
        on="gene_id",
        how="left",
    )

    nnsvg_metrics = compute_metrics(merged["nnsvg_LR_stat"].to_numpy(), merged["is_svg"].to_numpy())
    summary["nnsvg_metrics"] = nnsvg_metrics
    summary["mean_our_runtime_sec"] = float(np.mean(merged["our_runtime_sec"]))
    summary["nnsvg_runtime_total_sec"] = float(merged["nnsvg_runtime_total_sec"].dropna().iloc[0])

    merged.to_csv(output_dir / "benchmark_results.csv", index=False)
    make_plots(merged, output_dir)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Wrote benchmark outputs to {output_dir}")


if __name__ == "__main__":
    main()
