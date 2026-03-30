"""
Helpers for loading the Chicago taxi trips CSV into plug-and-play ``x, y`` arrays.

Default features are pickup centroid longitude/latitude and the default target
is trip miles, which gives a clearer spatial signal than duration on the
current sample file.
"""

from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_FILENAME = "Taxi_Trips_(2013-2023)_20260317.csv"
_DEFAULT_PATH = _REPO_ROOT / _DEFAULT_FILENAME

_FEATURE_COLUMNS = {
    "pickup": ("Pickup Centroid Longitude", "Pickup Centroid Latitude"),
    "dropoff": ("Dropoff Centroid Longitude", "Dropoff Centroid Latitude"),
}

_TARGET_COLUMNS = {
    "trip_miles": "Trip Miles",
    "trip_seconds": "Trip Seconds",
    "trip_total": "Trip Total",
    "fare": "Fare",
}


def _resolve_path(path=None):
    csv_path = _DEFAULT_PATH if path is None else Path(path)
    if not csv_path.is_absolute():
        csv_path = _REPO_ROOT / csv_path
    if not csv_path.exists():
        raise FileNotFoundError(f"Taxi trips CSV not found: {csv_path}")
    return csv_path


def _clean_numeric(series):
    return pd.to_numeric(
        series.astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce",
    )


def load_taxi(n_sub=None, seed=0, feature_set="pickup", target="trip_miles", path=None):
    """Load taxi trip features/targets as numpy arrays.

    Parameters
    ----------
    n_sub : int or None
        If given, uniformly subsample to this many valid trips.
    seed : int
        Random seed used when subsampling.
    feature_set : {'pickup', 'dropoff'}
        Which latitude/longitude pair to use for ``x``.
    target : {'trip_miles', 'trip_seconds', 'trip_total', 'fare'}
        Numeric response to use for ``y``.
    path : str or path-like or None
        Optional CSV path.

    Returns
    -------
    x : ndarray, shape (N, 2)
        Coordinates (longitude, latitude).
    y : ndarray, shape (N,)
        Target values for each trip.
    """
    if feature_set not in _FEATURE_COLUMNS:
        raise ValueError(f"Unsupported feature_set '{feature_set}'. Expected one of {sorted(_FEATURE_COLUMNS)}")
    if target not in _TARGET_COLUMNS:
        raise ValueError(f"Unsupported target '{target}'. Expected one of {sorted(_TARGET_COLUMNS)}")

    csv_path = _resolve_path(path)
    feature_cols = list(_FEATURE_COLUMNS[feature_set])
    target_col = _TARGET_COLUMNS[target]

    df = pd.read_csv(csv_path, usecols=feature_cols + [target_col])
    for col in feature_cols + [target_col]:
        df[col] = _clean_numeric(df[col])

    df = df.dropna()

    lon_col, lat_col = feature_cols
    df = df[
        df[lon_col].between(-180.0, 180.0)
        & df[lat_col].between(-90.0, 90.0)
        & (df[target_col] > 0.0)
    ]

    x = df[feature_cols].to_numpy(dtype=np.float64)
    y = df[target_col].to_numpy(dtype=np.float64)

    if n_sub is not None and n_sub < len(y):
        rng = np.random.default_rng(seed)
        inds = rng.choice(len(y), size=n_sub, replace=False)
        x = x[inds]
        y = y[inds]

    return x, y


def load_taxi_torch(n_sub=None, seed=0, feature_set="pickup", target="trip_miles", path=None):
    """Same as ``load_taxi`` but returns torch tensors (float64)."""
    import torch

    x, y = load_taxi(n_sub=n_sub, seed=seed, feature_set=feature_set, target=target, path=path)
    return torch.from_numpy(x.astype(np.float64)), torch.from_numpy(y.astype(np.float64))
