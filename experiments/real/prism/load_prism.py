"""
Helpers for loading PRISM GeoTIFF rasters into plug-and-play ``x, y`` arrays.

The default loader keeps the current mean-temperature dataset behavior intact,
and the generic helpers make it easy to point notebooks at the newer
``*_avg_30y`` directories without changing downstream code.
"""

from pathlib import Path

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATASET_DIRNAME = "prism_tmean_us_30s_202602"
_AVG_30Y_PPT_DIRNAME = "prism_ppt_us_30s_202001_avg_30y"
_AVG_30Y_TMEAN_DIRNAME = "prism_tmean_us_30s_2020_avg_30y"
_MODEL_PIXEL_SCALE_TAG = 33550
_MODEL_TIEPOINT_TAG = 33922
_GDAL_NODATA_TAG = 42113


def _resolve_dataset_dir(dataset):
    dataset_path = Path(dataset)
    if not dataset_path.is_absolute():
        dataset_path = _REPO_ROOT / dataset_path
    if not dataset_path.exists():
        raise FileNotFoundError(f"PRISM dataset directory not found: {dataset_path}")
    if not dataset_path.is_dir():
        raise NotADirectoryError(f"PRISM dataset path is not a directory: {dataset_path}")
    return dataset_path


def _find_tif_path(dataset_dir):
    tif_paths = sorted(dataset_dir.glob("*.tif"))
    if not tif_paths:
        raise FileNotFoundError(f"No .tif file found in {dataset_dir}")
    if len(tif_paths) > 1:
        raise ValueError(f"Expected one .tif file in {dataset_dir}, found {len(tif_paths)}")
    return tif_paths[0]


def _get_geotransform(img):
    pixel_scale = img.tag_v2.get(_MODEL_PIXEL_SCALE_TAG)
    tiepoint = img.tag_v2.get(_MODEL_TIEPOINT_TAG)
    if pixel_scale is None or tiepoint is None:
        raise ValueError("Missing GeoTIFF geotransform tags needed to build coordinates")

    pixel_width = float(pixel_scale[0])
    pixel_height = float(pixel_scale[1])
    origin_lon = float(tiepoint[3])
    origin_lat = float(tiepoint[4])
    return origin_lon, origin_lat, pixel_width, pixel_height


def _get_nodata_value(img):
    nodata = img.tag_v2.get(_GDAL_NODATA_TAG)
    if nodata is None:
        return None
    if isinstance(nodata, bytes):
        nodata = nodata.decode()
    return float(nodata)


def load_prism_dataset(dataset, n_sub=None, seed=0):
    """Load a PRISM GeoTIFF directory into ``x`` and ``y`` arrays.

    Parameters
    ----------
    dataset : str or path-like
        Dataset directory name relative to the repo root, or an absolute path.
    n_sub : int or None
        If given, uniformly subsample to this many valid pixels.
    seed : int
        Random seed used when subsampling.

    Returns
    -------
    x : ndarray, shape (N, 2)
        Coordinates (longitude, latitude) in NAD83 decimal degrees.
    y : ndarray, shape (N,)
        Raster values at those coordinates.
    """
    dataset_dir = _resolve_dataset_dir(dataset)
    tif_path = _find_tif_path(dataset_dir)

    img = Image.open(tif_path)
    data = np.array(img, dtype=np.float32)
    nrows, ncols = data.shape

    origin_lon, origin_lat, pixel_width, pixel_height = _get_geotransform(img)
    nodata = _get_nodata_value(img)

    col_idx = np.arange(ncols, dtype=np.float64)
    row_idx = np.arange(nrows, dtype=np.float64)
    lon = origin_lon + (col_idx + 0.5) * pixel_width
    lat = origin_lat - (row_idx + 0.5) * pixel_height
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    valid = np.isfinite(data) if nodata is None else data != nodata
    x = np.column_stack([lon_grid[valid], lat_grid[valid]])
    y = data[valid]

    if n_sub is not None and n_sub < len(y):
        rng = np.random.default_rng(seed)
        inds = rng.choice(len(y), size=n_sub, replace=False)
        x = x[inds]
        y = y[inds]

    return x, y


def load_prism(n_sub=None, seed=0):
    """Load the default PRISM mean-temperature dataset."""
    return load_prism_dataset(_DEFAULT_DATASET_DIRNAME, n_sub=n_sub, seed=seed)


def load_prism_avg_30y_ppt(n_sub=None, seed=0):
    """Load the PRISM 1991-2020 average January precipitation dataset."""
    return load_prism_dataset(_AVG_30Y_PPT_DIRNAME, n_sub=n_sub, seed=seed)


def load_prism_avg_30y_tmean(n_sub=None, seed=0):
    """Load the PRISM 1991-2020 average daily mean temperature dataset."""
    return load_prism_dataset(_AVG_30Y_TMEAN_DIRNAME, n_sub=n_sub, seed=seed)


def load_prism_dataset_torch(dataset, n_sub=None, seed=0):
    """Same as ``load_prism_dataset`` but returns torch tensors (float64)."""
    import torch

    x, y = load_prism_dataset(dataset, n_sub=n_sub, seed=seed)
    return torch.from_numpy(x.astype(np.float64)), torch.from_numpy(y.astype(np.float64))


def load_prism_torch(n_sub=None, seed=0):
    """Same as ``load_prism`` but returns torch tensors (float64)."""
    import torch

    x, y = load_prism(n_sub=n_sub, seed=seed)
    return torch.from_numpy(x.astype(np.float64)), torch.from_numpy(y.astype(np.float64))


def load_prism_avg_30y_ppt_torch(n_sub=None, seed=0):
    """Same as ``load_prism_avg_30y_ppt`` but returns torch tensors (float64)."""
    import torch

    x, y = load_prism_avg_30y_ppt(n_sub=n_sub, seed=seed)
    return torch.from_numpy(x.astype(np.float64)), torch.from_numpy(y.astype(np.float64))


def load_prism_avg_30y_tmean_torch(n_sub=None, seed=0):
    """Same as ``load_prism_avg_30y_tmean`` but returns torch tensors (float64)."""
    import torch

    x, y = load_prism_avg_30y_tmean(n_sub=n_sub, seed=seed)
    return torch.from_numpy(x.astype(np.float64)), torch.from_numpy(y.astype(np.float64))
