from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt
import scipy.sparse as sp
import scipy.special as sps

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pg_classifier import PolyagammaGPNegativeBinomialRegressor


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "data" / "visium_breast_cancer_dispersion_screen"
DEFAULT_SAMPLE_ID = "V1_Breast_Cancer_Block_A_Section_1"
DOMAIN_ORDER = ["Invasive", "In situ", "Non-tumor"]

# Published in the SpaMetric breast-cancer tutorial, which says the mapping
# follows pathologist annotation from Fu et al. This mapping is only fully
# defensible when the Leiden labels come from that same clustering workflow.
PUBLISHED_SPMETRIC_LEIDEN_DOMAIN_MAP: dict[str, str] = {
    "0": "Non-tumor",
    "1": "Invasive",
    "2": "Invasive",
    "3": "Invasive",
    "4": "In situ",
    "5": "In situ",
    "6": "Non-tumor",
    "7": "Non-tumor",
    "8": "Invasive",
    "9": "In situ",
    "10": "Non-tumor",
}

DEFAULT_GENE_PANEL: tuple[tuple[str, str], ...] = (
    ("ERBB2", "positive_control"),
    ("ESR1", "positive_control"),
    ("EPCAM", "positive_control"),
    ("KRT8", "positive_control"),
    ("MUC1", "positive_control"),
    ("CD3D", "positive_control"),
    ("CD8A", "positive_control"),
    ("IGKC", "positive_control"),
    ("GAPDH", "negative_control"),
    ("ACTB", "negative_control"),
    ("RPL13A", "negative_control"),
    ("RPLP0", "negative_control"),
    ("EEF1A1", "negative_control"),
    ("HLA-B", "exploratory"),
    ("HLA-A", "exploratory"),
    ("CXCL14", "exploratory"),
    ("CCND1", "exploratory"),
    ("C1QA", "exploratory"),
    ("APOE", "exploratory"),
    ("COX6C", "exploratory"),
)

REFERENCE_LINKS = {
    "scanpy_dataset_docs": "https://scanpy.readthedocs.io/en/latest/generated/scanpy.datasets.visium_sge.html",
    "scanpy_release_notes_visium_sge": "https://scanpy.readthedocs.io/en/1.9.x/release-notes/1.7.0.html",
    "spametric_breast_cancer_tutorial": "https://spametric.readthedocs.io/en/stable/tutorials/breast_cancer.html",
    "tenx_breast_cancer_dataset": "https://www.10xgenomics.com/resources/datasets/human-breast-cancer-block-a-section-1-1-standard-1-1-0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a quick domain-specific NB-dispersion screen on the Visium breast "
            "cancer sample using the PolyagammaGPNegativeBinomialRegressor."
        )
    )
    parser.add_argument("--adata-path", type=Path, default=None)
    parser.add_argument("--visium-dir", type=Path, default=None)
    parser.add_argument("--sample-id", type=str, default=DEFAULT_SAMPLE_ID)
    parser.add_argument("--annotation-csv", type=Path, default=None)
    parser.add_argument("--annotation-obs-column", type=str, default=None)
    parser.add_argument("--annotation-barcode-column", type=str, default="barcode")
    parser.add_argument("--annotation-domain-column", type=str, default="domain")
    parser.add_argument("--use-published-spametric-map", action="store_true")
    parser.add_argument("--compute-scanpy-leiden", action="store_true")
    parser.add_argument("--leiden-key", type=str, default="leiden")
    parser.add_argument("--leiden-resolution", type=float, default=1.5)
    parser.add_argument("--hvg-count", type=int, default=2000)
    parser.add_argument("--genes", nargs="*", default=None)
    parser.add_argument("--gene-panel-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--e-step-iters", type=int, default=1)
    parser.add_argument("--final-e-step-iters", type=int, default=2)
    parser.add_argument("--n-e-probes", type=int, default=3)
    parser.add_argument("--n-m-probes", type=int, default=6)
    parser.add_argument("--prediction-batch-size", type=int, default=128)
    parser.add_argument("--lengthscale-init", type=float, default=0.20)
    parser.add_argument("--variance-init", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--cg-tol", type=float, default=1e-6)
    parser.add_argument("--nufft-eps", type=float, default=1e-7)
    parser.add_argument("--spectral-eps", type=float, default=1e-4)
    parser.add_argument("--trunc-eps", type=float, default=1e-4)
    parser.add_argument("--total-count-quadrature-nodes", type=int, default=16)
    parser.add_argument("--min-spots-per-fit", type=int, default=120)
    parser.add_argument("--min-nonzero-spots", type=int, default=10)
    parser.add_argument("--min-total-count-sum", type=float, default=20.0)
    parser.add_argument("--min-init-total-count", type=float, default=0.2)
    parser.add_argument("--max-init-total-count", type=float, default=10.0)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def canonicalize_domain_label(value: object) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    normalized = text.casefold().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    aliases = {
        "invasive": "Invasive",
        "in situ": "In situ",
        "insitu": "In situ",
        "non tumor": "Non-tumor",
        "nontumor": "Non-tumor",
        "non tumour": "Non-tumor",
        "normal": "Non-tumor",
    }
    return aliases.get(normalized, text)


def normalize_spatial_coordinates(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.asarray(coords, dtype=np.float64)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = np.where(maxs > mins, maxs - mins, 1.0)
    coords_norm = 2.0 * (coords - mins) / span - 1.0
    return coords_norm, mins, span


def infer_domain_order(domains: pd.Series) -> list[str]:
    values = domains.dropna()
    if isinstance(values.dtype, pd.CategoricalDtype):
        categories = [str(cat) for cat in values.dtype.categories]
        present = set(map(str, values.astype(str)))
        return [cat for cat in categories if cat in present]

    ordered: list[str] = []
    seen: set[str] = set()
    for value in values.astype(str):
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def nb_logpmf(y: np.ndarray, mu: np.ndarray, total_count: float) -> np.ndarray:
    mu_safe = np.clip(np.asarray(mu, dtype=np.float64), 1e-10, None)
    y = np.asarray(y, dtype=np.float64)
    r = float(max(total_count, 1e-10))
    return (
        sps.gammaln(y + r)
        - sps.gammaln(r)
        - sps.gammaln(y + 1.0)
        + r * (math.log(r) - np.log(r + mu_safe))
        + y * (np.log(mu_safe) - np.log(r + mu_safe))
    )


def fit_intercept_only_nb(y: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y, dtype=np.float64)
    mu_hat = float(max(np.mean(y), 1e-8))
    var_hat = float(np.var(y))
    if var_hat <= mu_hat + 1e-6:
        return mu_hat, 100.0

    def objective(log_r: float) -> float:
        r = math.exp(log_r)
        return -float(nb_logpmf(y, np.full_like(y, mu_hat, dtype=np.float64), r).sum())

    result = opt.minimize_scalar(objective, bounds=(-4.0, 6.0), method="bounded")
    if not result.success:
        r_hat = max((mu_hat**2) / max(var_hat - mu_hat, 1e-6), 0.1)
    else:
        r_hat = math.exp(float(result.x))
    return mu_hat, float(np.clip(r_hat, 0.1, 100.0))


def moments_total_count(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    mean = float(max(np.mean(y), 1e-10))
    var = float(np.var(y))
    if var <= mean + 1e-8:
        return 100.0
    return float(np.clip((mean**2) / max(var - mean, 1e-8), 0.1, 100.0))


def default_gene_panel_frame() -> pd.DataFrame:
    return pd.DataFrame(DEFAULT_GENE_PANEL, columns=["requested_gene", "category"])


def load_gene_panel(args: argparse.Namespace) -> pd.DataFrame:
    if args.gene_panel_csv is not None:
        panel = pd.read_csv(args.gene_panel_csv, sep=None, engine="python")
        if "requested_gene" not in panel.columns and "gene" in panel.columns:
            panel = panel.rename(columns={"gene": "requested_gene"})
        if "requested_gene" not in panel.columns:
            raise ValueError("Gene panel file must include a 'requested_gene' or 'gene' column.")
        if "category" not in panel.columns:
            panel["category"] = "custom"
        return panel[["requested_gene", "category"]].copy()

    if args.genes:
        return pd.DataFrame(
            {"requested_gene": [str(g) for g in args.genes], "category": "custom"}
        )

    return default_gene_panel_frame()


def resolve_requested_genes(requested_genes: pd.Series, var_names: pd.Index) -> tuple[list[str], list[str]]:
    var_name_list = [str(name) for name in var_names]
    exact_names = set(var_name_list)
    folded_lookup: dict[str, str] = {}
    duplicates: set[str] = set()
    for name in var_name_list:
        key = name.casefold()
        if key in folded_lookup and folded_lookup[key] != name:
            duplicates.add(key)
        else:
            folded_lookup[key] = name

    resolved: list[str] = []
    missing: list[str] = []
    for gene in requested_genes.astype(str):
        if gene in exact_names:
            resolved.append(gene)
            continue
        key = gene.casefold()
        if key in duplicates or key not in folded_lookup:
            missing.append(gene)
            continue
        resolved.append(folded_lookup[key])
    return resolved, missing


def require_scanpy() -> Any:
    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError(
            "This workflow needs scanpy/anndata. Install scanpy plus leiden dependencies, "
            "for example: pip install 'scanpy>=1.10' 'anndata>=0.10' leidenalg igraph"
        ) from exc
    return sc


def load_adata(args: argparse.Namespace) -> Any:
    sc = require_scanpy()
    if args.adata_path is not None:
        adata = sc.read_h5ad(args.adata_path)
    elif args.visium_dir is not None:
        adata = sc.read_visium(args.visium_dir)
    else:
        adata = sc.datasets.visium_sge(sample_id=args.sample_id)
    adata.var_names_make_unique()
    return adata


def extract_count_matrix(adata: Any) -> sp.csr_matrix:
    matrix = adata.layers["counts"] if "counts" in adata.layers else adata.X
    if sp.issparse(matrix):
        counts = matrix.tocsr()
        sample = counts.data[: min(counts.nnz, 10000)]
    else:
        dense = np.asarray(matrix, dtype=np.float64)
        counts = sp.csr_matrix(dense)
        sample = dense.ravel()[: min(dense.size, 10000)]

    if sample.size and not np.allclose(sample, np.round(sample), atol=1e-6):
        raise ValueError(
            "Could not find an integer count matrix. Provide an AnnData object with raw counts "
            "in adata.layers['counts'] or adata.X."
        )
    return counts


def _compute_scanpy_leiden(
    adata: Any,
    *,
    leiden_key: str,
    resolution: float,
    hvg_count: int,
    random_state: int,
) -> Any:
    sc = require_scanpy()
    work = adata.copy()
    sc.pp.normalize_total(work, target_sum=1e4)
    sc.pp.log1p(work)
    n_top = int(min(max(hvg_count, 100), work.n_vars))
    sc.pp.highly_variable_genes(work, n_top_genes=n_top, flavor="seurat")
    if "highly_variable" in work.var and bool(np.any(work.var["highly_variable"].to_numpy())):
        work = work[:, work.var["highly_variable"].to_numpy()].copy()
    sc.pp.pca(work)
    sc.pp.neighbors(work, n_neighbors=15, n_pcs=min(30, work.obsm["X_pca"].shape[1]))
    try:
        sc.tl.leiden(
            work,
            resolution=resolution,
            key_added=leiden_key,
            random_state=random_state,
            flavor="igraph",
            directed=False,
            n_iterations=2,
        )
    except ImportError as exc:
        raise ImportError(
            "scanpy Leiden clustering requires igraph and leidenalg. "
            "Install them before using --compute-scanpy-leiden."
        ) from exc
    adata.obs[leiden_key] = work.obs[leiden_key].astype(str)
    return adata


def _load_annotation_csv(
    adata: Any,
    *,
    annotation_csv: Path,
    barcode_column: str,
    domain_column: str,
) -> pd.Series:
    frame = pd.read_csv(annotation_csv, sep=None, engine="python")
    if domain_column not in frame.columns:
        raise ValueError(f"Annotation file is missing the domain column '{domain_column}'.")
    if barcode_column in frame.columns:
        barcode_index = frame[barcode_column].astype(str)
    else:
        barcode_index = frame.index.astype(str)
    domain_values = frame[domain_column].map(canonicalize_domain_label)
    series = pd.Series(domain_values.to_numpy(), index=barcode_index, name="domain")
    aligned = series.reindex(adata.obs_names.astype(str))
    return aligned


def attach_domain_annotations(adata: Any, args: argparse.Namespace) -> tuple[pd.Series, list[str]]:
    notes: list[str] = []

    if args.annotation_obs_column is not None:
        if args.annotation_obs_column not in adata.obs.columns:
            raise ValueError(
                f"Requested annotation column '{args.annotation_obs_column}' is not present in adata.obs."
            )
        domains = adata.obs[args.annotation_obs_column].map(canonicalize_domain_label)
        notes.append(f"Used adata.obs['{args.annotation_obs_column}'] as the domain labels.")
        return pd.Series(domains.to_numpy(), index=adata.obs_names.astype(str), name="domain"), notes

    if args.annotation_csv is not None:
        domains = _load_annotation_csv(
            adata,
            annotation_csv=args.annotation_csv,
            barcode_column=args.annotation_barcode_column,
            domain_column=args.annotation_domain_column,
        )
        notes.append(f"Loaded barcode-level domain labels from {args.annotation_csv}.")
        return domains, notes

    if not args.use_published_spametric_map:
        raise ValueError(
            "No domain labels were provided. Supply --annotation-csv or --annotation-obs-column, "
            "or opt into the exploratory published fallback with --use-published-spametric-map."
        )

    if args.leiden_key not in adata.obs.columns:
        if not args.compute_scanpy_leiden:
            warnings.warn(
                "No Leiden labels were present, so a standard Scanpy Leiden solution will be computed. "
                "The published map_dict was defined for SpaMetric clusters, so these domain calls are "
                "exploratory only.",
                stacklevel=2,
            )
        adata = _compute_scanpy_leiden(
            adata,
            leiden_key=args.leiden_key,
            resolution=args.leiden_resolution,
            hvg_count=args.hvg_count,
            random_state=args.random_state,
        )
        notes.append(
            "Computed standard Scanpy Leiden clusters before applying the published SpaMetric map. "
            "Treat these domains as exploratory rather than pathologist-ground-truth."
        )
    else:
        notes.append(
            f"Used existing adata.obs['{args.leiden_key}'] labels with the published SpaMetric map_dict."
        )

    domains = adata.obs[args.leiden_key].astype(str).map(PUBLISHED_SPMETRIC_LEIDEN_DOMAIN_MAP)
    notes.append(
        "Applied the published SpaMetric Leiden->domain map attributed there to Fu et al. "
        "pathologist annotation."
    )
    return pd.Series(domains.to_numpy(), index=adata.obs_names.astype(str), name="domain"), notes


def _column_to_numpy(matrix: sp.csr_matrix, column_index: int) -> np.ndarray:
    return np.asarray(matrix.getcol(column_index).toarray()).ravel().astype(np.float64)


def fit_subset_model(
    *,
    X: np.ndarray,
    y: np.ndarray,
    gene: str,
    requested_gene: str,
    category: str,
    subset_label: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    y = np.asarray(y, dtype=np.float64)
    record: dict[str, object] = {
        "requested_gene": requested_gene,
        "gene": gene,
        "category": category,
        "subset": subset_label,
        "n_spots": int(y.size),
        "n_nonzero": int(np.count_nonzero(y)),
        "mean_count": float(np.mean(y)) if y.size else math.nan,
        "var_count": float(np.var(y)) if y.size else math.nan,
        "zero_fraction": float(np.mean(y == 0.0)) if y.size else math.nan,
        "mom_total_count": float(moments_total_count(y)) if y.size else math.nan,
        "intercept_mean": math.nan,
        "intercept_total_count": math.nan,
        "intercept_loglik": math.nan,
        "gp_total_count": math.nan,
        "gp_inverse_total_count": math.nan,
        "gp_lengthscale": math.nan,
        "gp_variance": math.nan,
        "gp_train_mae": math.nan,
        "gp_loglik": math.nan,
        "gp_loglik_gain_vs_intercept": math.nan,
        "status": "ok",
        "note": "",
    }
    if y.size == 0:
        record["status"] = "skip"
        record["note"] = "No spots were available in this subset."
        return record

    mu_null, r_null = fit_intercept_only_nb(y)
    record["intercept_mean"] = mu_null
    record["intercept_total_count"] = r_null
    record["intercept_loglik"] = float(nb_logpmf(y, np.full_like(y, mu_null), r_null).sum())

    if y.size < args.min_spots_per_fit:
        record["status"] = "skip"
        record["note"] = f"Only {y.size} spots; min_spots_per_fit={args.min_spots_per_fit}."
        return record
    if int(np.count_nonzero(y)) < args.min_nonzero_spots:
        record["status"] = "skip"
        record["note"] = (
            f"Only {int(np.count_nonzero(y))} nonzero spots; "
            f"min_nonzero_spots={args.min_nonzero_spots}."
        )
        return record
    if float(np.sum(y)) < args.min_total_count_sum:
        record["status"] = "skip"
        record["note"] = (
            f"Total count sum {float(np.sum(y)):.2f} is below "
            f"min_total_count_sum={args.min_total_count_sum:.2f}."
        )
        return record

    init_total_count = float(
        np.clip(r_null, args.min_init_total_count, args.max_init_total_count)
    )
    try:
        reg = PolyagammaGPNegativeBinomialRegressor(
            total_count=init_total_count,
            learn_total_count=True,
            total_count_lr=args.lr,
            total_count_update_frequency=1,
            total_count_quadrature_nodes=args.total_count_quadrature_nodes,
            lengthscale_init=args.lengthscale_init,
            variance_init=args.variance_init,
            max_iter=args.max_iter,
            e_step_iters=args.e_step_iters,
            final_e_step_iters=args.final_e_step_iters,
            n_e_probes=args.n_e_probes,
            n_m_probes=args.n_m_probes,
            prediction_batch_size=args.prediction_batch_size,
            lr=args.lr,
            cg_tol=args.cg_tol,
            nufft_eps=args.nufft_eps,
            spectral_eps=args.spectral_eps,
            trunc_eps=args.trunc_eps,
            random_state=args.random_state,
            verbose=args.verbose,
            store_history=True,
        )
        reg.fit(X, y)
        mu_gp = np.clip(reg.predict_mean_count(X).astype(np.float64), 1e-10, None)
        gp_ll = float(nb_logpmf(y, mu_gp, reg.total_count_).sum())
    except Exception as exc:  # pragma: no cover - real-data path can fail in compiled deps
        record["status"] = "error"
        record["note"] = f"{type(exc).__name__}: {exc}"
        return record

    record["gp_total_count"] = float(reg.total_count_)
    record["gp_inverse_total_count"] = float(1.0 / max(reg.total_count_, 1e-10))
    record["gp_lengthscale"] = float(reg.lengthscale_)
    record["gp_variance"] = float(reg.variance_)
    record["gp_train_mae"] = float(reg.training_mean_absolute_error_)
    record["gp_loglik"] = gp_ll
    record["gp_loglik_gain_vs_intercept"] = gp_ll - float(record["intercept_loglik"])
    return record


def run_dispersion_screen(adata: Any, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    counts = extract_count_matrix(adata)
    if "spatial" not in adata.obsm:
        raise ValueError("AnnData object is missing adata.obsm['spatial'].")
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    coords_norm, coord_mins, coord_span = normalize_spatial_coordinates(coords)

    domains, annotation_notes = attach_domain_annotations(adata, args)
    panel = load_gene_panel(args)
    resolved_by_row: list[str | None] = []
    missing_genes: list[str] = []
    for requested_gene in panel["requested_gene"].astype(str):
        resolved_genes, missing = resolve_requested_genes(pd.Series([requested_gene]), adata.var_names)
        resolved_by_row.append(resolved_genes[0] if resolved_genes else None)
        missing_genes.extend(missing)
    panel = panel.assign(gene=resolved_by_row)
    panel = panel.dropna(subset=["gene"]).copy()
    panel["gene"] = panel["gene"].astype(str)

    var_lookup = {str(name): idx for idx, name in enumerate(adata.var_names.astype(str))}
    records: list[dict[str, object]] = []
    observed_domains = infer_domain_order(domains)

    for _, row in panel.iterrows():
        gene = str(row["gene"])
        requested_gene = str(row["requested_gene"])
        y_all = _column_to_numpy(counts, var_lookup[gene])
        records.append(
            fit_subset_model(
                X=coords_norm,
                y=y_all,
                gene=gene,
                requested_gene=requested_gene,
                category=str(row["category"]),
                subset_label="Global",
                args=args,
            )
        )
        for domain in observed_domains:
            mask = domains.to_numpy() == domain
            records.append(
                fit_subset_model(
                    X=coords_norm[mask],
                    y=y_all[mask],
                    gene=gene,
                    requested_gene=requested_gene,
                    category=str(row["category"]),
                    subset_label=domain,
                    args=args,
                )
            )

    long_df = pd.DataFrame(records)
    wide_df = build_wide_summary(long_df)
    metadata = {
        "sample_id": args.sample_id,
        "n_spots": int(adata.n_obs),
        "n_genes_total": int(adata.n_vars),
        "n_genes_screened": int(panel.shape[0]),
        "resolved_genes": panel[["requested_gene", "gene", "category"]].to_dict(orient="records"),
        "missing_genes": missing_genes,
        "observed_domains": observed_domains,
        "annotation_notes": annotation_notes,
        "coord_mins": coord_mins.tolist(),
        "coord_span": coord_span.tolist(),
        "reference_links": REFERENCE_LINKS,
    }
    return long_df, wide_df, metadata


def build_wide_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()

    key_cols = ["requested_gene", "gene", "category"]
    total_count_wide = (
        long_df.pivot_table(index=key_cols, columns="subset", values="gp_total_count", aggfunc="first")
        .add_prefix("gp_total_count__")
        .reset_index()
    )
    inv_total_count_wide = (
        long_df.pivot_table(
            index=key_cols,
            columns="subset",
            values="gp_inverse_total_count",
            aggfunc="first",
        )
        .add_prefix("gp_inv_total_count__")
        .reset_index()
    )
    ll_gain_wide = (
        long_df.pivot_table(
            index=key_cols,
            columns="subset",
            values="gp_loglik_gain_vs_intercept",
            aggfunc="first",
        )
        .add_prefix("gp_loglik_gain__")
        .reset_index()
    )
    status_wide = (
        long_df.pivot_table(index=key_cols, columns="subset", values="status", aggfunc="first")
        .add_prefix("status__")
        .reset_index()
    )

    wide = total_count_wide.merge(inv_total_count_wide, on=key_cols, how="outer")
    wide = wide.merge(ll_gain_wide, on=key_cols, how="outer")
    wide = wide.merge(status_wide, on=key_cols, how="outer")

    observed_domains = [
        str(subset) for subset in long_df["subset"].drop_duplicates().tolist() if str(subset) != "Global"
    ]
    inv_cols = [f"gp_inv_total_count__{domain}" for domain in observed_domains if f"gp_inv_total_count__{domain}" in wide]
    total_cols = [f"gp_total_count__{domain}" for domain in observed_domains if f"gp_total_count__{domain}" in wide]

    if inv_cols:
        inv_vals = wide[inv_cols].to_numpy(dtype=np.float64)
        wide["domain_overdispersion_max_to_min"] = _rowwise_positive_max_to_min_ratio(inv_vals)
    else:
        wide["domain_overdispersion_max_to_min"] = math.nan

    if total_cols:
        total_vals = wide[total_cols].to_numpy(dtype=np.float64)
        wide["domain_total_count_max_to_min"] = _rowwise_positive_max_to_min_ratio(total_vals)
    else:
        wide["domain_total_count_max_to_min"] = math.nan

    return wide.sort_values(["category", "requested_gene"]).reset_index(drop=True)


def _rowwise_positive_max_to_min_ratio(values: np.ndarray) -> np.ndarray:
    ratios = np.full(values.shape[0], np.nan, dtype=np.float64)
    for row_index, row in enumerate(values):
        finite_positive = row[np.isfinite(row) & (row > 0.0)]
        if finite_positive.size == 0:
            continue
        ratios[row_index] = float(np.max(finite_positive) / np.min(finite_positive))
    return ratios


def save_heatmap(long_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = long_df[["requested_gene", "subset", "gp_inverse_total_count"]].copy()
    if plot_df.empty:
        return

    subset_order = ["Global"] + [domain for domain in DOMAIN_ORDER if domain in set(plot_df["subset"])]
    heatmap = (
        plot_df.pivot_table(index="requested_gene", columns="subset", values="gp_inverse_total_count", aggfunc="first")
        .reindex(columns=subset_order)
        .sort_index()
    )
    if heatmap.empty:
        return

    values = heatmap.to_numpy(dtype=np.float64)
    fig_height = max(4.0, 0.35 * heatmap.shape[0] + 1.2)
    fig, ax = plt.subplots(figsize=(7.2, fig_height), dpi=150)
    image = ax.imshow(values, aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(heatmap.shape[1]))
    ax.set_xticklabels(list(heatmap.columns), rotation=30, ha="right")
    ax.set_yticks(np.arange(heatmap.shape[0]))
    ax.set_yticklabels(list(heatmap.index))
    ax.set_title("NB overdispersion proxy by subset (1 / r)")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("1 / learned total_count")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    adata = load_adata(args)
    long_df, wide_df, metadata = run_dispersion_screen(adata, args)

    long_path = args.output_dir / "dispersion_screen_long.csv"
    wide_path = args.output_dir / "dispersion_screen_wide.csv"
    metadata_path = args.output_dir / "dispersion_screen_metadata.json"
    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    if not args.no_plots:
        save_heatmap(long_df, args.output_dir / "dispersion_overdispersion_heatmap.png")

    print(f"Wrote long results to {long_path}")
    print(f"Wrote wide results to {wide_path}")
    print(f"Wrote metadata to {metadata_path}")
    if metadata["missing_genes"]:
        print(f"Missing genes skipped: {', '.join(metadata['missing_genes'])}")
    if metadata["annotation_notes"]:
        print("Annotation notes:")
        for note in metadata["annotation_notes"]:
            print(f"  - {note}")


if __name__ == "__main__":
    main()
