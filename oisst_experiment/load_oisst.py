"""
Helpers for loading OISST NetCDF rasters into plug-and-play ``x, y`` arrays.

This reader uses ``h5py`` directly because the local environment does not have
the NetCDF backends that ``xarray`` would normally rely on.
"""

from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_FILENAME = "oisst-avhrr-v02r01.20260315_preliminary.nc"
_DEFAULT_PATH = _REPO_ROOT / _DEFAULT_FILENAME
_GRID_VARIABLES = {"sst", "anom", "err", "ice"}


def _decode_attr(value):
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode_attr(value.item())
        if value.size == 1:
            return _decode_attr(value.reshape(-1)[0])
        return [_decode_attr(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _resolve_path(path=None):
    dataset_path = _DEFAULT_PATH if path is None else Path(path)
    if not dataset_path.is_absolute():
        dataset_path = _REPO_ROOT / dataset_path
    if not dataset_path.exists():
        raise FileNotFoundError(f"OISST dataset not found: {dataset_path}")
    return dataset_path


def _wrap_longitudes(lon):
    lon_wrapped = ((lon + 180.0) % 360.0) - 180.0
    order = np.argsort(lon_wrapped)
    return lon_wrapped[order], order


def _open_variable(path, variable):
    if variable not in _GRID_VARIABLES:
        raise ValueError(f"Unsupported OISST variable '{variable}'. Expected one of {sorted(_GRID_VARIABLES)}")

    with h5py.File(path, "r") as f:
        lat = np.array(f["lat"], dtype=np.float32)
        lon = np.array(f["lon"], dtype=np.float32)
        raw = np.array(f[variable][0, 0, :, :], dtype=np.int16)

        var_attrs = {key: _decode_attr(val) for key, val in f[variable].attrs.items()}
        root_attrs = {key: _decode_attr(val) for key, val in f.attrs.items()}
        time_value = float(np.array(f["time"], dtype=np.float32).reshape(-1)[0])
        time_units = _decode_attr(f["time"].attrs["units"])

    fill_value = float(var_attrs["_FillValue"])
    scale_factor = float(var_attrs.get("scale_factor", 1.0))
    add_offset = float(var_attrs.get("add_offset", 0.0))

    data = raw.astype(np.float32) * scale_factor + add_offset
    data = np.where(raw == fill_value, np.nan, data)

    metadata = {
        "path": str(path),
        "variable": variable,
        "title": root_attrs.get("title"),
        "id": root_attrs.get("id"),
        "long_name": var_attrs.get("long_name"),
        "units": var_attrs.get("units"),
        "time_value": time_value,
        "time_units": time_units,
        "date": _parse_time_value(time_value, time_units),
    }
    return lon, lat, data, metadata


def _parse_time_value(time_value, time_units):
    prefix = "days since "
    if not isinstance(time_units, str) or not time_units.startswith(prefix):
        return None
    base_text = time_units[len(prefix):]
    base_time = datetime.strptime(base_text, "%Y-%m-%d %H:%M:%S")
    return base_time + timedelta(days=float(time_value))


def load_oisst_grid(variable="sst", path=None, lon_range="-180_180"):
    """Load an OISST variable on its native grid.

    Returns
    -------
    lon_grid : ndarray, shape (n_lat, n_lon)
    lat_grid : ndarray, shape (n_lat, n_lon)
    values : ndarray, shape (n_lat, n_lon)
    metadata : dict
    """
    dataset_path = _resolve_path(path)
    lon, lat, data, metadata = _open_variable(dataset_path, variable)

    if lon_range == "-180_180":
        lon, order = _wrap_longitudes(lon)
        data = data[:, order]
    elif lon_range != "0_360":
        raise ValueError("lon_range must be '-180_180' or '0_360'")

    lon_grid, lat_grid = np.meshgrid(lon.astype(np.float64), lat.astype(np.float64))
    return lon_grid, lat_grid, data.astype(np.float32), metadata


def load_oisst(variable="sst", n_sub=None, seed=0, path=None, lon_range="-180_180"):
    """Load an OISST variable into ``x`` and ``y`` arrays.

    Parameters
    ----------
    variable : {'sst', 'anom', 'err', 'ice'}
        Grid variable to load.
    n_sub : int or None
        If given, uniformly subsample to this many valid pixels.
    seed : int
        Random seed used when subsampling.
    path : str or path-like or None
        Optional path to the NetCDF file.
    lon_range : {'-180_180', '0_360'}
        Longitude convention to return.

    Returns
    -------
    x : ndarray, shape (N, 2)
        Coordinates (longitude, latitude) in decimal degrees.
    y : ndarray, shape (N,)
        Variable values at those coordinates.
    """
    lon_grid, lat_grid, values, _ = load_oisst_grid(variable=variable, path=path, lon_range=lon_range)
    valid = np.isfinite(values)
    x = np.column_stack([lon_grid[valid], lat_grid[valid]])
    y = values[valid]

    if n_sub is not None and n_sub < len(y):
        rng = np.random.default_rng(seed)
        inds = rng.choice(len(y), size=n_sub, replace=False)
        x = x[inds]
        y = y[inds]

    return x, y


def load_oisst_torch(variable="sst", n_sub=None, seed=0, path=None, lon_range="-180_180"):
    """Same as ``load_oisst`` but returns torch tensors (float64)."""
    import torch

    x, y = load_oisst(variable=variable, n_sub=n_sub, seed=seed, path=path, lon_range=lon_range)
    return torch.from_numpy(x.astype(np.float64)), torch.from_numpy(y.astype(np.float64))
