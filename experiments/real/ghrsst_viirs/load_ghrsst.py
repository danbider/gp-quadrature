"""
Helpers for loading GHRSST L2P swath NetCDF files into ``x, y`` arrays.

The reader expects a single VIIRS L2P granule (e.g. ``VIIRS_NPP-STAR-L2P-v2.80``)
that has been downloaded via ``earthaccess``. Coordinates are returned as
(longitude, latitude) and SST values are converted from Kelvin to Celsius by
default.
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

_MODULE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _MODULE_DIR.parents[2]
_SEARCH_DIRS = (
    _MODULE_DIR,
    _MODULE_DIR / "data",
    _REPO_ROOT,
)


def _resolve_path(path):
    candidate = Path(path)
    if candidate.is_absolute():
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"GHRSST dataset not found: {candidate}")

    tried = []
    for base in _SEARCH_DIRS:
        p = (base / candidate).resolve()
        if p.exists():
            return p
        tried.append(str(p))
    raise FileNotFoundError(
        "GHRSST dataset not found. Looked in:\n  " + "\n  ".join(tried)
    )


def download_ghrsst(
    short_name="VIIRS_NPP-STAR-L2P-v2.80",
    temporal=("2024-06-15", "2024-06-15"),
    bounding_box=(-80.0, 25.0, -60.0, 45.0),
    count=1,
    out_dir=None,
):
    """Download GHRSST L2P granules via earthaccess.

    Requires a NASA Earthdata login. ``earthaccess.login()`` will prompt on
    the first call.
    """
    import earthaccess

    earthaccess.login()
    results = earthaccess.search_data(
        short_name=short_name,
        temporal=temporal,
        bounding_box=bounding_box,
        count=count,
    )
    out_dir = Path(out_dir) if out_dir is not None else (_MODULE_DIR / "data")
    out_dir.mkdir(parents=True, exist_ok=True)
    files = earthaccess.download(results, str(out_dir))
    return [Path(f) for f in files]


def load_ghrsst_swath(
    path,
    variable="sea_surface_temperature",
    quality_min=None,
    to_celsius=True,
    bbox=None,
):
    """Load a GHRSST L2P granule on its native swath.

    Parameters
    ----------
    path : str or path-like
        Path to a GHRSST L2P NetCDF granule.
    variable : str
        Variable to extract. Defaults to ``sea_surface_temperature``.
    quality_min : int or None
        Minimum ``quality_level`` to retain (GHRSST convention: 5 = best,
        1 = worst). When ``None`` (default), no quality filtering is applied
        beyond the file's mask/fill-value handling.
    to_celsius : bool
        If True (default) and the variable's units are Kelvin, convert to
        degrees Celsius.
    bbox : tuple or None
        Optional ``(lon_min, lat_min, lon_max, lat_max)`` bounding box to
        clip to before flattening.

    Returns
    -------
    lon : ndarray, shape (n_lat_swath, n_lon_swath)
    lat : ndarray, shape (n_lat_swath, n_lon_swath)
    values : ndarray, shape (n_lat_swath, n_lon_swath)
        ``NaN`` where pixels are flagged invalid.
    metadata : dict
    """
    dataset_path = _resolve_path(path)

    with xr.open_dataset(dataset_path, mask_and_scale=True) as ds:
        if variable not in ds.variables:
            raise KeyError(
                f"Variable '{variable}' not present in {dataset_path}. "
                f"Available: {list(ds.data_vars)}"
            )

        lat = np.asarray(ds["lat"].values, dtype=np.float64)
        lon = np.asarray(ds["lon"].values, dtype=np.float64)
        var = ds[variable].squeeze()
        values = np.asarray(var.values, dtype=np.float64)
        units = var.attrs.get("units")
        long_name = var.attrs.get("long_name", variable)

        if quality_min is not None and "quality_level" in ds.variables:
            ql = np.asarray(ds["quality_level"].squeeze().values)
            values = np.where(ql >= quality_min, values, np.nan)

        time_iso = None
        if "time" in ds.variables:
            t = ds["time"].values
            t = t.reshape(-1)[0] if t.size else None
            if t is not None:
                try:
                    time_iso = np.datetime_as_string(np.datetime64(t), unit="s")
                except (TypeError, ValueError):
                    time_iso = str(t)

        title = ds.attrs.get("title")
        ds_id = ds.attrs.get("id") or ds.attrs.get("product_name")

    if to_celsius and isinstance(units, str) and units.lower() in {"kelvin", "k"}:
        values = values - 273.15
        units = "Celsius"

    if bbox is not None:
        lon_min, lat_min, lon_max, lat_max = bbox
        in_box = (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
        values = np.where(in_box, values, np.nan)

    metadata = {
        "path": str(dataset_path),
        "variable": variable,
        "title": title,
        "id": ds_id,
        "long_name": long_name,
        "units": units,
        "time": time_iso,
    }
    return lon, lat, values, metadata


def load_ghrsst(
    path,
    variable="sea_surface_temperature",
    quality_min=5,
    to_celsius=True,
    bbox=None,
    n_sub=None,
    seed=0,
):
    """Load a GHRSST granule (or list of granules) into ``x`` and ``y`` arrays.

    Parameters
    ----------
    path : str, path-like, or sequence of those
        Path(s) to L2P NetCDF file(s). When a sequence is given, all granules
        are concatenated (after per-granule masking) into a single ``(N, 2)``
        coordinate matrix.
    variable : str
        Field to extract. Defaults to ``sea_surface_temperature``.
    quality_min : int or None
        Minimum ``quality_level`` to retain. Defaults to ``5`` (best only).
    to_celsius : bool
        Convert from Kelvin if applicable.
    bbox : tuple or None
        Optional ``(lon_min, lat_min, lon_max, lat_max)`` clipping box.
    n_sub : int or None
        Subsample to this many valid pixels (after concatenation).
    seed : int
        Random seed for subsampling.

    Returns
    -------
    x : ndarray, shape (N, 2)
        Coordinates (longitude, latitude) in decimal degrees.
    y : ndarray, shape (N,)
    """
    paths = _as_path_list(path)

    xs, ys = [], []
    for p in paths:
        lon, lat, values, _ = load_ghrsst_swath(
            path=p,
            variable=variable,
            quality_min=quality_min,
            to_celsius=to_celsius,
            bbox=bbox,
        )
        valid = np.isfinite(values)
        if not valid.any():
            continue
        xs.append(np.column_stack([lon[valid].ravel(), lat[valid].ravel()]))
        ys.append(values[valid].ravel())

    if not xs:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64)

    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)

    if n_sub is not None and n_sub < len(y):
        rng = np.random.default_rng(seed)
        inds = rng.choice(len(y), size=n_sub, replace=False)
        x = x[inds]
        y = y[inds]

    return x, y


def load_ghrsst_torch(
    path,
    variable="sea_surface_temperature",
    quality_min=5,
    to_celsius=True,
    bbox=None,
    n_sub=None,
    seed=0,
):
    """Same as ``load_ghrsst`` but returns torch tensors (float64)."""
    import torch

    x, y = load_ghrsst(
        path=path,
        variable=variable,
        quality_min=quality_min,
        to_celsius=to_celsius,
        bbox=bbox,
        n_sub=n_sub,
        seed=seed,
    )
    return torch.from_numpy(x.astype(np.float64)), torch.from_numpy(y.astype(np.float64))


def _as_path_list(path):
    if isinstance(path, (str, Path)):
        return [path]
    return list(path)
