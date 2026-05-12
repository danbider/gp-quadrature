"""
Compare baseline EFGPND hyper-learning vs a 'frozen-grid every K iters'
schedule on the VNP21 LST dataset (~1M points, Jacobi mean-CG preconditioner).

Idea: when the frequency grid (mtot) is held fixed across iters, the M-space
mean-CG warm start (`self._last_gradient_beta`) actually applies — it drops
mean_cg from ~100 down to ~15 (observed in bench_lst_grad).

We refresh the grid every K iters so it can still adapt to the moving
hyperparameters. Schedule:

  iter t  |  grid action
  --------+-----------------------------------
  t % K == 0  refresh the grid for the current hypers, then solve
  otherwise   reuse the previously-frozen grid (warm start kicks in)

Trace-CG has no warm start in efgpnd.py, so this experiment only exercises
the mean-CG side. The trace_cg_iters number should stay unchanged — the win
is on the mean solve.
"""

from __future__ import annotations

import argparse
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

from efgpnd import EFGPND, _resolve_grid  # noqa: E402
from kernels import SquaredExponential  # noqa: E402
from load_modis_lst import load_viirs_lst_torch  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)

BBOX = (-125.0, 25.0, -66.0, 50.0)


def load_data(n_sub: int) -> tuple[torch.Tensor, torch.Tensor]:
    data_dir = REPO / "experiments" / "real" / "modis_lst" / "data"
    granules = sorted(data_dir.glob("VNP21*.nc"))
    print(f"Loading {len(granules)} VNP21 granules ...", flush=True)
    x, y = load_viirs_lst_torch(
        path=granules, variable="LST", quality="good_only",
        to_celsius=True, bbox=BBOX, n_sub=n_sub, seed=0,
    )
    x_min, x_max = x.min(dim=0).values, x.max(dim=0).values
    x = (x - x_min) / (x_max - x_min)
    y = (y - y.mean()) / y.std()
    return x.to(torch.float32), y.to(torch.float32)


def build_frozen_grid(model: EFGPND, eps: float):
    """Build a (xis_1d_list, h_per_dim) frozen-grid tuple from the *current*
    kernel hypers and data extent."""
    xis_1d_list, h_per_dim, _, _ = _resolve_grid(
        model.kernel, model.x, eps,
        frozen_grid=None, max_mtot_1d=None,
        rdtype=model.x.dtype, device=model.x.device,
    )
    return (xis_1d_list, h_per_dim)


def run_config(label, x, y, *, refresh_every, max_iters, eps, cg_tol,
               init_ls, init_var, init_noise, lr, noise_floor,
               trace_warm_start=False):
    """Run an Adam loop. ``refresh_every`` controls grid pinning:
       0  -> baseline, never freeze (current behaviour)
       K>0 -> refresh frozen_grid every K iters; otherwise reuse it

    ``trace_warm_start=True`` enables the new trace-CG warm start
    (cached Beta_all + frozen Hutchinson probes).
    """
    d = x.shape[1]
    kernel = SquaredExponential(
        dimension=d, init_lengthscale=init_ls, init_variance=init_var
    )
    opts = {"mean_cg_preconditioner_type": "jacobi"}
    if not trace_warm_start:
        # Disable both — the warm start is meaningless if probes change every
        # iter, so we also turn off the freeze for an apples-to-apples baseline.
        opts["trace_cg_warm_start"] = False
        opts["trace_probes_freeze"] = False
    model = EFGPND(
        x, y, kernel=kernel, sigmasq=init_noise, eps=eps,
        estimate_params=False,
        opts=opts,
    )
    optimizer = Adam(model.parameters(), lr=lr)

    frozen_grid = None
    log = []
    print(f"\n=== {label}  refresh_every={refresh_every} ===", flush=True)
    for it in range(max_iters):
        if refresh_every:
            if it % refresh_every == 0:
                frozen_grid = build_frozen_grid(model, eps)

        t0 = time.perf_counter()
        optimizer.zero_grad()
        model.compute_gradients(
            trace_samples=1, cg_tol=cg_tol, noise_floor=noise_floor,
            frozen_grid=frozen_grid,
        )
        optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        s = model.last_gradient_stats
        rec = {
            "iter": it + 1,
            "wall_sec": dt,
            "grid_refreshed": bool(refresh_every and it % refresh_every == 0),
            "lengthscale": float(model.kernel.get_hyper("lengthscale")),
            "variance": float(model.kernel.get_hyper("variance")),
            "sigmasq": float(model._gp_params.sig2.item()),
            "mean_cg_iters": int(s.get("mean_cg_iters")),
            "trace_cg_iters": int(s.get("trace_cg_iters")),
            "mtot": s.get("mtot") if not isinstance(s.get("mtot"), tuple) else list(s.get("mtot")),
            "feature_count": int(s.get("feature_count")),
        }
        log.append(rec)
        flag = "*" if rec["grid_refreshed"] else " "
        print(
            f" {flag} iter {it+1:>3}  {dt:6.2f}s  "
            f"ℓ={rec['lengthscale']:.4g}  σf²={rec['variance']:.4g}  "
            f"σn²={rec['sigmasq']:.4g}  "
            f"cg(mean/trace)={rec['mean_cg_iters']}/{rec['trace_cg_iters']}  "
            f"mtot={rec['mtot']}  M={rec['feature_count']}",
            flush=True,
        )
    return log


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_sub", type=int, default=1_000_000)
    p.add_argument("--max_iters", type=int, default=50)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--cg_tol", type=float, default=1e-4)
    p.add_argument("--init_ls", type=float, default=0.1)
    p.add_argument("--init_var", type=float, default=1.0)
    p.add_argument("--init_noise", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--noise_floor", type=float, default=1e-4)
    p.add_argument("--refresh_every", type=int, nargs="+", default=[0, 5],
                   help="refresh schedules to compare; 0=baseline (never freeze)")
    p.add_argument("--also_with_trace_warm", type=int, nargs="*",
                   default=[5],
                   help="for each K listed here, also run refresh_every_K with "
                        "the trace-CG warm start enabled")
    p.add_argument("--out", type=Path,
                   default=REPO / "scratch" / "bench_lst_warmstart.json")
    args = p.parse_args()

    x, y = load_data(args.n_sub)
    print(f"x: {tuple(x.shape)}  y: {tuple(y.shape)}", flush=True)

    runs = {}
    for k in args.refresh_every:
        label = "baseline" if k == 0 else f"refresh_every_{k}"
        runs[label] = run_config(
            label, x, y, refresh_every=k,
            max_iters=args.max_iters, eps=args.eps, cg_tol=args.cg_tol,
            init_ls=args.init_ls, init_var=args.init_var,
            init_noise=args.init_noise, lr=args.lr,
            noise_floor=args.noise_floor,
            trace_warm_start=False,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"runs": runs}, indent=2))

    for k in args.also_with_trace_warm:
        label = f"refresh_every_{k}_trace_warm"
        runs[label] = run_config(
            label, x, y, refresh_every=k,
            max_iters=args.max_iters, eps=args.eps, cg_tol=args.cg_tol,
            init_ls=args.init_ls, init_var=args.init_var,
            init_noise=args.init_noise, lr=args.lr,
            noise_floor=args.noise_floor,
            trace_warm_start=True,
        )
        # snapshot to disk after each config so partial runs are usable
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"runs": runs}, indent=2))

    print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
