"""
Per-iteration breakdown of EFGPND hyper-learning on the VNP21 LST dataset
(~1M points, 2D), Jacobi mean-CG preconditioner.

For each Adam step we record:
  - total wall time
  - mean CG iters / trace CG iters
  - mtot per dim, total feature count M
  - hyperparameter values

For three "checkpoint" iterations (early, mid, late) we additionally record
a per-section breakdown by manually instrumenting the gradient pipeline
through compute_gradients(do_profiling=True).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "real" / "modis_lst"))

from efgpnd import EFGPND  # noqa: E402
from kernels import SquaredExponential  # noqa: E402
from load_modis_lst import load_viirs_lst_torch  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)


def load_data(n_sub: int) -> tuple[torch.Tensor, torch.Tensor]:
    data_dir = REPO / "experiments" / "real" / "modis_lst" / "data"
    granules = sorted(data_dir.glob("VNP21*.nc"))
    print(f"Loading {len(granules)} VNP21 granules ...", flush=True)
    bbox = (-125.0, 25.0, -66.0, 50.0)
    x, y = load_viirs_lst_torch(
        path=granules,
        variable="LST",
        quality="good_only",
        to_celsius=True,
        bbox=bbox,
        n_sub=n_sub,
        seed=0,
    )
    x_min, x_max = x.min(dim=0).values, x.max(dim=0).values
    x = (x - x_min) / (x_max - x_min)
    y = (y - y.mean()) / y.std()
    return x.to(torch.float32), y.to(torch.float32)


def main(
    n_sub: int = 1_000_000,
    max_iters: int = 50,
    eps: float = 1e-3,
    cg_tol: float = 1e-4,
    init_ls: float = 0.1,
    init_var: float = 1.0,
    init_noise: float = 1.0,
    lr: float = 0.1,
    out_path: Path = REPO / "scratch" / "bench_lst_grad.json",
):
    x, y = load_data(n_sub)
    print(f"x: {tuple(x.shape)}  y: {tuple(y.shape)}", flush=True)

    d = x.shape[1]
    kernel = SquaredExponential(
        dimension=d, init_lengthscale=init_ls, init_variance=init_var
    )
    model = EFGPND(
        x, y, kernel=kernel, sigmasq=init_noise, eps=eps,
        estimate_params=False,
        opts={"mean_cg_preconditioner_type": "jacobi"},
    )
    optimizer = Adam(model.parameters(), lr=lr)

    log = []
    for it in range(max_iters):
        t0 = time.perf_counter()
        optimizer.zero_grad()
        model.compute_gradients(
            trace_samples=1, cg_tol=cg_tol, noise_floor=1e-4,
        )
        optimizer.step()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        dt = time.perf_counter() - t0

        s = model.last_gradient_stats
        rec = {
            "iter": it + 1,
            "wall_sec": dt,
            "lengthscale": float(model.kernel.get_hyper("lengthscale")),
            "variance": float(model.kernel.get_hyper("variance")),
            "sigmasq": float(model._gp_params.sig2.item()),
            "mean_cg_iters": int(s.get("mean_cg_iters")),
            "trace_cg_iters": int(s.get("trace_cg_iters")),
            "trace_num_rhs": int(s.get("trace_num_rhs")),
            "mtot": s.get("mtot") if not isinstance(s.get("mtot"), tuple) else list(s.get("mtot")),
            "mtot_per_dim": list(s.get("mtot_per_dim")),
            "feature_count": int(s.get("feature_count")),
        }
        log.append(rec)

        if it % 1 == 0:
            print(
                f"iter {it+1:>3}  {dt:6.2f}s  "
                f"ℓ={rec['lengthscale']:.4g}  σf²={rec['variance']:.4g}  "
                f"σn²={rec['sigmasq']:.4g}  "
                f"cg(mean/trace)={rec['mean_cg_iters']}/{rec['trace_cg_iters']}  "
                f"mtot={rec['mtot']}  M={rec['feature_count']}",
                flush=True,
            )

        # snapshot to disk after each iter so we can monitor live
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"records": log}, indent=2))

    out_path.write_text(json.dumps({"records": log}, indent=2))
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
