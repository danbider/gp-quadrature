"""
Helpers for loading the ERA5 2m temperature dataset.

Source: ERA5 monthly averaged reanalysis on single levels (Copernicus CDS).
File: era5.nc — global 0.25° grid, 721 x 1440 = ~1M cells.
Variable: t2m (2-meter temperature in Kelvin).
"""

from pathlib import Path
import numpy as np
import h5py

_MODULE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _MODULE_DIR.parents[2]
_FILENAME = "era5.nc"
_SEARCH_DIRS = (_MODULE_DIR / "data", _MODULE_DIR, _REPO_ROOT)


def _resolve_data_path():
    tried = []
    for base in _SEARCH_DIRS:
        p = (base / _FILENAME).resolve()
        if p.exists():
            return p
        tried.append(str(p))
    raise FileNotFoundError(
        "ERA5 dataset not found. Looked in:\n  " + "\n  ".join(tried)
    )


DATA_PATH = _resolve_data_path()


def load_era5(n_sub=None, seed=0, celsius=True):
    """Load the ERA5 2m temperature dataset.

    Parameters
    ----------
    n_sub : int or None
        If given, uniformly subsample to this many points.
    seed : int
        Random seed used when subsampling.
    celsius : bool
        If True (default), convert temperature from Kelvin to Celsius.

    Returns
    -------
    x : ndarray, shape (N, 2)
        Coordinates (longitude, latitude) in degrees.
        Longitude is shifted to [-180, 180].
    y : ndarray, shape (N,)
        2m temperature (°C if celsius=True, K otherwise).
    """
    with h5py.File(DATA_PATH, "r") as ds:
        t2m = ds["t2m"][:].astype(np.float64)
        lat = ds["latitude"][:].astype(np.float64)
        lon = ds["longitude"][:].astype(np.float64)

    # Drop singleton time dim
    if t2m.ndim == 3:
        t2m = t2m[0]

    # Shift longitude from [0, 360) to [-180, 180)
    lon = np.where(lon > 180, lon - 360, lon)

    LON, LAT = np.meshgrid(lon, lat)
    x = np.column_stack([LON.ravel(), LAT.ravel()])
    y = t2m.ravel()

    if celsius:
        y = y - 273.15

    if n_sub is not None and n_sub < len(y):
        rng = np.random.default_rng(seed)
        inds = rng.choice(len(y), size=n_sub, replace=False)
        x = x[inds]
        y = y[inds]

    return x, y


def load_era5_torch(n_sub=None, seed=0, celsius=True):
    """Same as load_era5 but returns torch tensors (float64)."""
    import torch

    x, y = load_era5(n_sub=n_sub, seed=seed, celsius=celsius)
    return torch.from_numpy(x), torch.from_numpy(y)
