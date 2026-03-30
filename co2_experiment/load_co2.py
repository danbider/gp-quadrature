"""
Helpers for loading the OCO-2 CO2 dataset.

The raw data lives in ../co2_data/ (binary files produced from the dataset
described in https://www.tandfonline.com/doi/full/10.1080/01621459.2017.1419136).

co2_xs.bin   — N×2 float64 array of (latitude, longitude) on disk
co2_meas.bin — N float64 array of CO2 concentration (ppm)
"""

import os
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "co2_data")


def load_co2(n_sub=None, seed=0, lonlat=True):
    """Load the OCO-2 CO2 dataset.

    Parameters
    ----------
    n_sub : int or None
        If given, uniformly subsample to this many points.
    seed : int
        Random seed used when subsampling.
    lonlat : bool
        If True (default), return coordinates as (lon, lat) to match the
        convention in Heaton et al.  If False, return (lat, lon) as stored
        on disk.

    Returns
    -------
    x : ndarray, shape (2, N)
        Coordinates.  Row 0 is longitude (or latitude if lonlat=False),
        row 1 is latitude (or longitude if lonlat=False).
    meas : ndarray, shape (N,)
        CO2 concentration in ppm.
    """
    meas_path = os.path.join(DATA_DIR, "co2_meas.bin")
    xs_path = os.path.join(DATA_DIR, "co2_xs.bin")

    meas = np.fromfile(meas_path, dtype=np.float64)
    N = meas.shape[0]
    # MATLAB fread(fid, [N 2], 'double') reads column-major, so the binary
    # layout is [all N latitudes, then all N longitudes].
    raw = np.fromfile(xs_path, dtype=np.float64)
    x = np.column_stack([raw[:N], raw[N:]])  # columns: lat, lon

    if n_sub is not None and n_sub < N:
        rng = np.random.default_rng(seed)
        inds = rng.choice(N, size=n_sub, replace=False)
        x = x[inds]
        meas = meas[inds]

    x = x.T  # shape (2, N) — row 0 = lat, row 1 = lon
    if lonlat:
        x = x[[1, 0]]  # swap to (lon, lat)

    return x, meas


def load_co2_torch(n_sub=None, seed=0, lonlat=True):
    """Same as load_co2 but returns torch tensors (float64)."""
    import torch

    x, meas = load_co2(n_sub=n_sub, seed=seed, lonlat=lonlat)
    return torch.from_numpy(x), torch.from_numpy(meas)
