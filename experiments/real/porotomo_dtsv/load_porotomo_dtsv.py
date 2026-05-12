"""
Helpers for loading the PoroTomo DTSv (vertical DTS) dataset.

Source: PoroTomo 8/7 vertical DTS, processed and split into a forward
(down-going) and backward (up-going) leg of the borehole fiber. The
file ``PoroTomo_8_7_DTSv_Processed.xlsx`` ships as a 360 MB Excel
workbook with two sheets, each ~2935 depth rows by ~11235 time columns
(8.1 days at ~1-minute spacing).

We parse the xlsx once with python-calamine and cache it to a ``.npz``
under ``data/``; subsequent loads read from the cache and are essentially
instantaneous.
"""

from datetime import datetime
from pathlib import Path

import numpy as np

_MODULE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _MODULE_DIR.parents[2]
_XLSX_FILENAME = "PoroTomo_8_7_DTSv_Processed.xlsx"
_SEARCH_DIRS = (
    _MODULE_DIR / "data",
    _MODULE_DIR,
    _REPO_ROOT,
    Path.home() / "Downloads",
)
_LEG_TO_SHEET = {"forward": "Forward Length", "backward": "Backward Length"}


def _resolve_xlsx_path(path=None):
    if path is not None:
        p = Path(path).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"PoroTomo DTSv xlsx not found: {p}")
    tried = []
    for base in _SEARCH_DIRS:
        p = (base / _XLSX_FILENAME).resolve()
        if p.exists():
            return p
        tried.append(str(p))
    raise FileNotFoundError(
        "PoroTomo DTSv dataset not found. Looked in:\n  " + "\n  ".join(tried)
    )


def _cache_path(leg):
    return _MODULE_DIR / "data" / f"porotomo_dtsv_{leg}.npz"


def _build_cache(xlsx_path, leg, verbose=True):
    from python_calamine import CalamineWorkbook  # heavy import, only on cache miss

    sheet = _LEG_TO_SHEET[leg]
    if verbose:
        print(f"Parsing {xlsx_path.name} sheet '{sheet}' (one-time cache build)…")
    wb = CalamineWorkbook.from_path(str(xlsx_path))
    data = wb.get_sheet_by_name(sheet).to_python()

    # Row 0: ['', t1, t2, ...]; row 1: column-name header; rows 2+: depth, T...
    timestamps = [datetime.strptime(s, "%d/%m/%Y %H:%M:%S") for s in data[0][1:]]
    z = np.asarray([row[0] for row in data[2:]], dtype=np.float64)
    T = np.asarray([row[1:] for row in data[2:]], dtype=np.float64)

    t0 = timestamps[0]
    t_seconds = np.asarray([(ts - t0).total_seconds() for ts in timestamps], dtype=np.float64)

    cache = _cache_path(leg)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        z=z,
        t_seconds=t_seconds,
        T=T,
        start_datetime=np.array(t0.isoformat(), dtype="<U32"),
        leg=np.array(leg, dtype="<U16"),
    )
    if verbose:
        print(f"  cached to {cache} ({cache.stat().st_size/1e6:.1f} MB)")
    return cache


def load_porotomo_dtsv_grid(leg="forward", path=None, force_rebuild=False, verbose=True):
    """Load the PoroTomo DTSv dataset on its native (depth, time) grid.

    Parameters
    ----------
    leg : {'forward', 'backward'}
        Which fiber leg sheet to load.
    path : str or path-like or None
        Optional override for the xlsx path. Default: standard search dirs.
    force_rebuild : bool
        Rebuild the .npz cache even if it exists.
    verbose : bool
        Print progress messages during cache builds.

    Returns
    -------
    z : ndarray, shape (n_z,)
        Depth in metres.
    t : ndarray, shape (n_t,)
        Elapsed time in seconds since the first sample.
    T : ndarray, shape (n_z, n_t)
        Temperature in degrees Celsius.
    metadata : dict
        Acquisition metadata (start datetime, leg, spacing).
    """
    if leg not in _LEG_TO_SHEET:
        raise ValueError(f"leg must be one of {sorted(_LEG_TO_SHEET)}; got {leg!r}")

    cache = _cache_path(leg)
    if force_rebuild or not cache.exists():
        xlsx = _resolve_xlsx_path(path)
        _build_cache(xlsx, leg, verbose=verbose)

    arr = np.load(cache, allow_pickle=False)
    z = arr["z"]
    t = arr["t_seconds"]
    T = arr["T"]
    start_iso = str(arr["start_datetime"])
    metadata = {
        "cache_path": str(cache),
        "leg": str(arr["leg"]),
        "start_datetime": datetime.fromisoformat(start_iso),
        "n_z": int(T.shape[0]),
        "n_t": int(T.shape[1]),
        "z_step_m": float(np.median(np.diff(z))),
        "t_step_s": float(np.median(np.diff(t))),
    }
    return z, t, T, metadata


def load_porotomo_dtsv(leg="forward", path=None, n_sub=None, seed=0,
                       force_rebuild=False, verbose=True):
    """Load PoroTomo DTSv into ``x`` (depth, time) and ``y`` (temperature) arrays.

    Parameters
    ----------
    leg : {'forward', 'backward'}
        Which fiber leg sheet to load.
    path : str or path-like or None
        Optional override for the xlsx path.
    n_sub : int or None
        Optional uniform random subsample.
    seed : int
        RNG seed for subsampling.
    force_rebuild : bool
        Rebuild the .npz cache even if it exists.
    verbose : bool
        Print progress messages during cache builds.

    Returns
    -------
    x : ndarray, shape (N, 2)
        Coordinates ``(depth_m, time_s)``.
    y : ndarray, shape (N,)
        Temperature in degrees Celsius.
    metadata : dict
        Acquisition metadata.
    """
    z, t, T, metadata = load_porotomo_dtsv_grid(
        leg=leg, path=path, force_rebuild=force_rebuild, verbose=verbose
    )
    Z, Tg = np.meshgrid(z, t, indexing="ij")
    x = np.column_stack([Z.ravel(), Tg.ravel()])
    y = T.ravel()

    valid = np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if n_sub is not None and n_sub < len(y):
        rng = np.random.default_rng(seed)
        inds = rng.choice(len(y), size=n_sub, replace=False)
        x = x[inds]
        y = y[inds]

    return x, y, metadata


def load_porotomo_dtsv_torch(leg="forward", path=None, n_sub=None, seed=0,
                             force_rebuild=False, verbose=True):
    """Same as ``load_porotomo_dtsv`` but returns torch tensors (float64)."""
    import torch

    x, y, metadata = load_porotomo_dtsv(
        leg=leg, path=path, n_sub=n_sub, seed=seed,
        force_rebuild=force_rebuild, verbose=verbose,
    )
    return torch.from_numpy(x), torch.from_numpy(y), metadata
