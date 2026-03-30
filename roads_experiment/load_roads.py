"""
Helpers for loading the 3D Road Network (North Jutland, Denmark) dataset.

Source: https://archive.ics.uci.edu/ml/datasets/3D+Road+Network+(North+Jutland,+Denmark)

The raw file is a headerless CSV with columns:
    OSM_ID, LONGITUDE, LATITUDE, ALTITUDE
where coordinates are in Web Mercator (Google) format and altitude is in meters.
"""

import os
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "3D_spatial_network.txt")


def load_roads(n_sub=None, seed=0):
    """Load the 3D road network dataset.

    Parameters
    ----------
    n_sub : int or None
        If given, uniformly subsample to this many points.
    seed : int
        Random seed used when subsampling.

    Returns
    -------
    x : ndarray, shape (N, 2)
        Coordinates (longitude, latitude).
    y : ndarray, shape (N,)
        Altitude in meters.
    """
    data = np.loadtxt(DATA_PATH, delimiter=",")
    # columns: osm_id, lon, lat, altitude
    lon = data[:, 1]
    lat = data[:, 2]
    alt = data[:, 3]

    x = np.column_stack([lon, lat])

    if n_sub is not None and n_sub < len(alt):
        rng = np.random.default_rng(seed)
        inds = rng.choice(len(alt), size=n_sub, replace=False)
        x = x[inds]
        alt = alt[inds]

    return x, alt


def load_roads_torch(n_sub=None, seed=0):
    """Same as load_roads but returns torch tensors (float64)."""
    import torch

    x, y = load_roads(n_sub=n_sub, seed=seed)
    return torch.from_numpy(x), torch.from_numpy(y)
