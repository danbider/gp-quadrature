import math

import numpy as np
import pandas as pd

from visium_breast_cancer_dispersion_screen import (
    build_wide_summary,
    canonicalize_domain_label,
    infer_domain_order,
    normalize_spatial_coordinates,
    resolve_requested_genes,
)


def test_canonicalize_domain_label_handles_common_aliases():
    assert canonicalize_domain_label(" invasive ") == "Invasive"
    assert canonicalize_domain_label("in_situ") == "In situ"
    assert canonicalize_domain_label("non tumor") == "Non-tumor"
    assert canonicalize_domain_label(np.nan) is None


def test_normalize_spatial_coordinates_maps_box_to_minus_one_plus_one():
    coords = np.array([[10.0, 100.0], [20.0, 160.0], [30.0, 130.0]])
    coords_norm, mins, span = normalize_spatial_coordinates(coords)

    assert np.allclose(mins, [10.0, 100.0])
    assert np.allclose(span, [20.0, 60.0])
    assert np.all(coords_norm >= -1.0 - 1e-12)
    assert np.all(coords_norm <= 1.0 + 1e-12)
    assert np.allclose(coords_norm[0], [-1.0, -1.0])
    assert np.allclose(coords_norm[1], [0.0, 1.0])
    assert np.allclose(coords_norm[2], [1.0, 0.0])


def test_resolve_requested_genes_prefers_exact_then_unique_casefold():
    resolved, missing = resolve_requested_genes(
        pd.Series(["ERBB2", "esr1", "missing"]),
        pd.Index(["ERBB2", "ESR1", "GAPDH"]),
    )

    assert resolved == ["ERBB2", "ESR1"]
    assert missing == ["missing"]


def test_infer_domain_order_preserves_first_seen_for_noncategorical_labels():
    domains = pd.Series(["Tumor", "Healthy", "Tumor", "Invasive", None, "Healthy"])
    assert infer_domain_order(domains) == ["Tumor", "Healthy", "Invasive"]


def test_build_wide_summary_computes_domain_ratios():
    long_df = pd.DataFrame(
        [
            {
                "requested_gene": "ERBB2",
                "gene": "ERBB2",
                "category": "positive_control",
                "subset": "Global",
                "gp_total_count": 4.0,
                "gp_inverse_total_count": 0.25,
                "gp_loglik_gain_vs_intercept": 1.0,
                "status": "ok",
            },
            {
                "requested_gene": "ERBB2",
                "gene": "ERBB2",
                "category": "positive_control",
                "subset": "Invasive",
                "gp_total_count": 2.0,
                "gp_inverse_total_count": 0.5,
                "gp_loglik_gain_vs_intercept": 2.0,
                "status": "ok",
            },
            {
                "requested_gene": "ERBB2",
                "gene": "ERBB2",
                "category": "positive_control",
                "subset": "In situ",
                "gp_total_count": 8.0,
                "gp_inverse_total_count": 0.125,
                "gp_loglik_gain_vs_intercept": 0.5,
                "status": "ok",
            },
        ]
    )

    wide = build_wide_summary(long_df)

    assert wide.shape[0] == 1
    assert math.isclose(wide.loc[0, "domain_total_count_max_to_min"], 4.0)
    assert math.isclose(wide.loc[0, "domain_overdispersion_max_to_min"], 4.0)
