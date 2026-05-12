"""
Helpers for loading the Brady PoroTomo DTS dataset.

Source: Distributed Temperature Sensing (DTS) measurements made in
Brady Hot Springs observation well 56-1 with a Silixa XT instrument
(2018 acquisition). Kratt, Coleman, Tyler, Wang, Miller, Feigl (2019).
DOI: 10.15121/1495412 — https://gdr.openei.org/submissions/1114

The .mat file contains a 2361 x 1210 temperature matrix (channel x time),
sampled every ~25 cm along the fiber and every ~1 minute. The fiber is
deployed in a double-ended configuration, so it travels down and back up
the borehole: temperature peaks at the bottom around channel ~1180. By
default we keep only the down-going leg to avoid the mirrored profile.
"""

from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import scipy.io

_MODULE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _MODULE_DIR.parents[2]
_FILENAME = "Brady_25Aug2018_ch1.mat"
_SEARCH_DIRS = (
    _MODULE_DIR / "data",
    _MODULE_DIR,
    _REPO_ROOT,
    Path.home() / "Downloads",
)


def _resolve_data_path(path=None):
    if path is not None:
        p = Path(path).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"Brady DTS file not found: {p}")
    tried = []
    for base in _SEARCH_DIRS:
        p = (base / _FILENAME).resolve()
        if p.exists():
            return p
        tried.append(str(p))
    raise FileNotFoundError(
        "Brady DTS dataset not found. Looked in:\n  " + "\n  ".join(tried)
    )


def _matlab_datenum_to_datetime(dn):
    return datetime.fromordinal(int(dn) - 366) + timedelta(days=float(dn) % 1)


def load_brady_grid(path=None, downgoing_only=True):
    """Load the Brady DTS dataset on its native (depth, time) grid.

    Returns
    -------
    z : ndarray, shape (n_z,)
        Along-fiber distance in metres for the retained channels.
    t : ndarray, shape (n_t,)
        Elapsed time in seconds since the first sample.
    T : ndarray, shape (n_z, n_t)
        Temperature in degrees Celsius.
    metadata : dict
        Acquisition metadata (start datetime, channel/time spacing, etc).
    """
    data_path = _resolve_data_path(path)
    mat = scipy.io.loadmat(data_path)

    distance = mat["distance"].ravel().astype(np.float64)  # (n_z,) metres
    datenum = mat["datetime"].ravel().astype(np.float64)   # (n_t,) MATLAB serial date
    tempC = mat["tempC"].astype(np.float64)                # (n_z, n_t)

    if downgoing_only:
        # Fiber turns around near the global temperature peak at the borehole bottom.
        mid_time = tempC.shape[1] // 2
        turnaround = int(np.argmax(tempC[:, mid_time]))
        distance = distance[: turnaround + 1]
        tempC = tempC[: turnaround + 1, :]

    seconds_per_day = 86400.0
    t_seconds = (datenum - datenum[0]) * seconds_per_day

    metadata = {
        "path": str(data_path),
        "start_datetime": _matlab_datenum_to_datetime(datenum[0]),
        "n_z": tempC.shape[0],
        "n_t": tempC.shape[1],
        "z_step_m": float(np.median(np.diff(distance))),
        "t_step_s": float(np.median(np.diff(t_seconds))),
        "downgoing_only": bool(downgoing_only),
    }
    return distance, t_seconds, tempC, metadata


def detrend_depth_polynomial(z, T, deg=3):
    """Subtract a low-order polynomial in depth at each time step.

    Removes the borehole's geothermal gradient so a stationary GP can
    model the residual fluctuations.
    """
    T_out = np.empty_like(T)
    for j in range(T.shape[1]):
        col = T[:, j]
        mask = np.isfinite(col)
        if mask.sum() > deg + 1:
            coeffs = np.polynomial.polynomial.polyfit(z[mask], col[mask], deg=deg)
            trend = np.polynomial.polynomial.polyval(z, coeffs)
            T_out[:, j] = col - trend
        else:
            T_out[:, j] = np.nan
    return T_out


def load_brady(path=None, downgoing_only=True, detrend=True, detrend_deg=3,
               n_sub=None, seed=0):
    """Load Brady DTS into ``x`` (depth, time) and ``y`` (temperature) arrays.

    Parameters
    ----------
    path : str or path-like or None
        Optional path to the .mat file. Defaults to standard search locations.
    downgoing_only : bool
        Keep only the down-going leg of the fiber (first half).
    detrend : bool
        If True, subtract a degree-``detrend_deg`` polynomial in depth at
        each time step. The geothermal gradient dominates the raw signal.
    detrend_deg : int
        Polynomial degree used for depth detrending.
    n_sub : int or None
        Optional uniform random subsample.
    seed : int
        RNG seed for subsampling.

    Returns
    -------
    x : ndarray, shape (N, 2)
        Coordinates ``(depth_m, time_s)``.
    y : ndarray, shape (N,)
        Temperature (°C) — detrended residual if ``detrend`` is True.
    metadata : dict
        Acquisition metadata.
    """
    z, t, T, metadata = load_brady_grid(path=path, downgoing_only=downgoing_only)

    if detrend:
        T = detrend_depth_polynomial(z, T, deg=detrend_deg)

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


def load_brady_torch(path=None, downgoing_only=True, detrend=True, detrend_deg=3,
                     n_sub=None, seed=0):
    """Same as ``load_brady`` but returns torch tensors (float64)."""
    import torch

    x, y, metadata = load_brady(
        path=path,
        downgoing_only=downgoing_only,
        detrend=detrend,
        detrend_deg=detrend_deg,
        n_sub=n_sub,
        seed=seed,
    )
    return torch.from_numpy(x), torch.from_numpy(y), metadata
