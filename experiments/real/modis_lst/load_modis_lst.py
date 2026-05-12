"""
Helpers for loading VIIRS Land Surface Temperature swath granules into
``x, y`` arrays.

Defaults to ``VNP21`` (VIIRS NPP LST 1 km swath, NetCDF4/HDF5). Also works
for ``VJ121`` (VIIRS-JPSS1) which uses the same on-disk layout. Both ship
as NetCDF4 (.nc) and read cleanly via ``xarray`` + ``netCDF4`` with no HDF4
dependency.

The MODIS Terra equivalent ``MOD11_L2`` is HDF-EOS2 (HDF4) and would need
``pyhdf``; we deliberately avoid that here.
"""

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

# Group paths inside a VNP21 / VJ121 NetCDF4 file. The product's swath name
# varies across collections (VIIRS_Swath_LSTE for VNP21 v2; some HDF-EOS
# reprocessings use VIIRS_Swath_LST under HDFEOS/SWATHS).
_DATA_GROUPS = (
    "VIIRS_Swath_LSTE/Data Fields",
    "VIIRS_Swath_LST/Data Fields",
    "HDFEOS/SWATHS/VIIRS_Swath_LSTE/Data Fields",
    "HDFEOS/SWATHS/VIIRS_Swath_LST/Data Fields",
    "Data Fields",
    None,  # flattened
)
_GEO_GROUPS = (
    "VIIRS_Swath_LSTE/Geolocation Fields",
    "VIIRS_Swath_LST/Geolocation Fields",
    "HDFEOS/SWATHS/VIIRS_Swath_LSTE/Geolocation Fields",
    "HDFEOS/SWATHS/VIIRS_Swath_LST/Geolocation Fields",
    "Geolocation Fields",
    None,
)
# Latitude/longitude variables ship under different capitalisations.
_LAT_NAMES = ("latitude", "Latitude", "lat")
_LON_NAMES = ("longitude", "Longitude", "lon")


def _resolve_path(path):
    candidate = Path(path)
    if candidate.is_absolute():
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"VIIRS LST dataset not found: {candidate}")

    tried = []
    for base in _SEARCH_DIRS:
        p = (base / candidate).resolve()
        if p.exists():
            return p
        tried.append(str(p))
    raise FileNotFoundError(
        "VIIRS LST dataset not found. Looked in:\n  " + "\n  ".join(tried)
    )


def download_viirs_lst(
    short_name="VNP21",
    temporal=("2024-06-15", "2024-06-15"),
    bounding_box=(-125.0, 25.0, -66.0, 50.0),
    count=10,
    out_dir=None,
):
    """Download VIIRS LST granules via earthaccess.

    Requires a NASA Earthdata login. ``earthaccess.login()`` will prompt on
    the first call.

    ``short_name`` defaults to ``VNP21`` (VIIRS NPP LST). Use ``VJ121`` for
    VIIRS-JPSS1.
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


# Keep the old name alive in case anyone imports it.
download_modis_lst = download_viirs_lst


def _find_latlon(ds):
    """Return (lat_name, lon_name) found in ``ds``, or (None, None)."""
    lat = next((n for n in _LAT_NAMES if n in ds.variables), None)
    lon = next((n for n in _LON_NAMES if n in ds.variables), None)
    return lat, lon


def _open_groups(path):
    """Open the data + geolocation groups of a VIIRS LST granule.

    Returns
    -------
    data_ds, geo_ds, lat_name, lon_name : xr.Dataset, xr.Dataset, str, str
    """
    last_err = None
    for data_grp, geo_grp in zip(_DATA_GROUPS, _GEO_GROUPS):
        try:
            data_ds = xr.open_dataset(
                path, mask_and_scale=True, group=data_grp,
            )
        except (OSError, KeyError, ValueError) as e:
            last_err = e
            continue

        # If the data group already carries lat/lon (flat layout),
        # geo_ds is just data_ds; otherwise pull from the sibling group.
        lat_n, lon_n = _find_latlon(data_ds)
        if lat_n is not None and lon_n is not None:
            return data_ds, data_ds, lat_n, lon_n
        try:
            geo_ds = xr.open_dataset(
                path, mask_and_scale=True, group=geo_grp,
            )
        except (OSError, KeyError, ValueError) as e:
            last_err = e
            data_ds.close()
            continue

        lat_n, lon_n = _find_latlon(geo_ds)
        if lat_n is not None and lon_n is not None:
            return data_ds, geo_ds, lat_n, lon_n
        data_ds.close()
        geo_ds.close()
        last_err = KeyError(
            f"No lat/lon under group={geo_grp!r} (looked for {_LAT_NAMES} / {_LON_NAMES})"
        )

    # Last-ditch peek to give a useful error message.
    try:
        with xr.open_dataset(path) as ds:
            available = list(ds.data_vars)
    except Exception:
        available = ["<could not enumerate>"]
    raise RuntimeError(
        f"Could not open {path} as a VIIRS LST granule. Last error: {last_err}. "
        f"Top-level data variables: {available}."
    )


def load_viirs_lst_swath(
    path,
    variable="LST",
    quality="good_only",
    to_celsius=True,
    bbox=None,
):
    """Load a single VIIRS LST granule on its native swath.

    Parameters
    ----------
    path : str or path-like
    variable : str
        Field name in the data group. ``LST`` is the daytime/nighttime LST.
    quality : {'good_only', 'good_and_other', None}
        How to filter using the ``QC`` byte (bits 0-1):
          - ``'good_only'`` keeps only QC[0:1] == 0 (default).
          - ``'good_and_other'`` keeps QC[0:1] in {0, 1} (drops cloud/missing).
          - ``None`` skips QC masking.
    to_celsius : bool
        Convert from Kelvin if the variable's units say so.
    bbox : tuple or None
        Optional ``(lon_min, lat_min, lon_max, lat_max)`` clipping box.

    Returns
    -------
    lon : ndarray, shape (along, across)
    lat : ndarray, shape (along, across)
    values : ndarray, shape (along, across) (NaN where invalid)
    metadata : dict
    """
    dataset_path = _resolve_path(path)
    data_ds, geo_ds, lat_name, lon_name = _open_groups(dataset_path)

    try:
        if variable not in data_ds.variables:
            raise KeyError(
                f"Variable '{variable}' not found. "
                f"Available: {list(data_ds.data_vars)}"
            )
        lst = data_ds[variable].squeeze()
        values = np.asarray(lst.values, dtype=np.float64)
        units = lst.attrs.get("units")
        long_name = lst.attrs.get("long_name", variable)

        if quality is not None and "QC" in data_ds.variables:
            qc = np.asarray(data_ds["QC"].squeeze().values).astype(np.int64)
            mandatory = qc & 0x03
            if quality == "good_only":
                ok = mandatory == 0
            elif quality == "good_and_other":
                ok = mandatory <= 1
            else:
                raise ValueError(f"Unknown quality option: {quality!r}")
            values = np.where(ok, values, np.nan)

        lat = np.asarray(geo_ds[lat_name].values, dtype=np.float64)
        lon = np.asarray(geo_ds[lon_name].values, dtype=np.float64)
    finally:
        data_ds.close()
        if geo_ds is not data_ds:
            geo_ds.close()

    # Handle the case where geolocation is sub-sampled (older VIIRS layouts).
    if lat.shape != values.shape:
        lat = _resample_to(lat, values.shape)
        lon = _resample_to(lon, values.shape)

    if to_celsius and isinstance(units, str) and units.lower() in {"kelvin", "k"}:
        values = values - 273.15
        units = "Celsius"

    if bbox is not None:
        lon_min, lat_min, lon_max, lat_max = bbox
        in_box = (
            (lon >= lon_min) & (lon <= lon_max)
            & (lat >= lat_min) & (lat <= lat_max)
        )
        values = np.where(in_box, values, np.nan)

    metadata = {
        "path": str(dataset_path),
        "variable": variable,
        "long_name": long_name,
        "units": units,
        "quality_filter": quality,
    }
    return lon, lat, values, metadata


# Backwards-compat shim: the old MOD11_L2 swath loader took ``quality_max``.
def load_modis_lst_swath(path, variable="LST", quality_max=1, to_celsius=True, bbox=None):
    quality = "good_and_other" if quality_max and quality_max >= 1 else "good_only"
    return load_viirs_lst_swath(
        path=path, variable=variable, quality=quality,
        to_celsius=to_celsius, bbox=bbox,
    )


def _resample_to(arr, target_shape):
    """Up-sample a coarse 2D geolocation grid to ``target_shape`` by nearest-
    neighbour."""
    h, w = arr.shape
    H, W = target_shape
    rows = np.clip(np.round(np.linspace(0, h - 1, H)).astype(int), 0, h - 1)
    cols = np.clip(np.round(np.linspace(0, w - 1, W)).astype(int), 0, w - 1)
    return arr[np.ix_(rows, cols)]


def load_viirs_lst(
    path,
    variable="LST",
    quality="good_only",
    to_celsius=True,
    bbox=None,
    n_sub=None,
    seed=0,
):
    """Load VIIRS LST granule(s) into ``x`` and ``y`` arrays.

    ``path`` may be a single file or a sequence of files (concatenated after
    masking).
    """
    paths = _as_path_list(path)

    xs, ys = [], []
    for p in paths:
        lon, lat, values, _ = load_viirs_lst_swath(
            path=p,
            variable=variable,
            quality=quality,
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


# Backwards-compat shim.
def load_modis_lst(path, variable="LST", quality_max=1, to_celsius=True, bbox=None,
                   n_sub=None, seed=0):
    quality = "good_and_other" if quality_max and quality_max >= 1 else "good_only"
    return load_viirs_lst(
        path=path, variable=variable, quality=quality,
        to_celsius=to_celsius, bbox=bbox, n_sub=n_sub, seed=seed,
    )


def load_viirs_lst_torch(
    path,
    variable="LST",
    quality="good_only",
    to_celsius=True,
    bbox=None,
    n_sub=None,
    seed=0,
):
    """Same as ``load_viirs_lst`` but returns torch tensors (float64)."""
    import torch

    x, y = load_viirs_lst(
        path=path, variable=variable, quality=quality,
        to_celsius=to_celsius, bbox=bbox, n_sub=n_sub, seed=seed,
    )
    return torch.from_numpy(x.astype(np.float64)), torch.from_numpy(y.astype(np.float64))


# Backwards-compat shim (old name used by the original notebook draft).
def load_modis_lst_torch(path, variable="LST", quality_max=1, to_celsius=True, bbox=None,
                         n_sub=None, seed=0):
    quality = "good_and_other" if quality_max and quality_max >= 1 else "good_only"
    return load_viirs_lst_torch(
        path=path, variable=variable, quality=quality,
        to_celsius=to_celsius, bbox=bbox, n_sub=n_sub, seed=seed,
    )


def _as_path_list(path):
    if isinstance(path, (str, Path)):
        return [path]
    return list(path)
