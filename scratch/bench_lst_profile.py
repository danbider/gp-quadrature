"""
Per-section profiling of EFGPND training on the FULL VNP21 LST dataset
(~16.5M clear-sky points, no subsample), Jacobi mean-CG preconditioner,
frozen-grid every 5 iters with mean+trace warm starts (the defaults).

For every iter we record total wall time. For a handful of `profile_iters`
we wrap `compute_gradients` in a `torch.profiler.profile` context and pull
out the per-section CPU times keyed off the `record_function` blocks in
`efgpnd_gradient_batched`:

  0_book_keeping
  1_frequency_grid_setup
  2_nufft_setup           <- build the NUFFT operator
  3_toeplitz_setup        <- includes a big NUFFT type-1 to build v_kernel
  4_solve_cg              <- mean CG (matvec = Toeplitz FFT)
  5_compute_term2
  6_monte_carlo_trace     <- builds B_all (Toeplitz on probes)
  7_batch_cg_solve        <- trace CG (matvec = batched Toeplitz FFT)
  7.5_compute_alpha
  8_gradient_calculation

That tells us exactly where the wall time goes: NUFFT setup vs Toeplitz
setup vs mean CG vs trace CG vs everything else.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.profiler import profile, ProfilerActivity

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "real" / "modis_lst"))

from efgpnd import EFGPND, _resolve_grid  # noqa: E402
from kernels import SquaredExponential  # noqa: E402
from load_modis_lst import load_viirs_lst_torch  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)

BBOX = (-125.0, 25.0, -66.0, 50.0)

SECTIONS = [
    "0_book_keeping",
    "1_frequency_grid_setup",
    "2_nufft_setup",
    "3_toeplitz_setup",
    "4_solve_cg",
    "5_compute_term2",
    "6_monte_carlo_trace",
    "7_batch_cg_solve",
    "7.5_compute_alpha",
    "8_gradient_calculation",
]


def load_data(n_sub):
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


def build_frozen_grid(model, eps):
    xis_1d_list, h_per_dim, _, _ = _resolve_grid(
        model.kernel, model.x, eps,
        rdtype=model.x.dtype, device=model.x.device,
    )
    return (xis_1d_list, h_per_dim)


def section_times_us(prof):
    """Pull per-section total CPU time (microseconds) from a torch profile."""
    out = {s: 0.0 for s in SECTIONS}
    for ev in prof.key_averages():
        if ev.key in out:
            out[ev.key] = float(ev.cpu_time_total)  # microseconds, total across calls
    return out


def main(
    n_sub=None,                     # full dataset
    max_iters=30,
    refresh_every=5,
    eps=1e-2,
    cg_tol=1e-4,
    init_ls=0.3,
    init_var=1.0,
    init_noise=1.0,
    lr=0.2,
    profile_iters=(1, 2, 6, 10, 16, 25),
    out_path=REPO / "scratch" / "bench_lst_profile_cached.json",
):
    x, y = load_data(n_sub)
    print(f"x: {tuple(x.shape)}  y: {tuple(y.shape)}", flush=True)
    print(f"Using full dataset = {x.shape[0]:,} points", flush=True)

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
    section_log = {}
    frozen_grid = None

    for it in range(max_iters):
        if refresh_every and it % refresh_every == 0:
            frozen_grid = build_frozen_grid(model, eps)
            refreshed = True
        else:
            refreshed = False

        do_prof = (it + 1) in profile_iters

        t0 = time.perf_counter()
        optimizer.zero_grad()

        if do_prof:
            # CPU-only profile so we get per-section wall time without CUDA noise.
            with profile(activities=[ProfilerActivity.CPU], record_shapes=False) as prof_ctx:
                model.compute_gradients(
                    trace_samples=1, cg_tol=cg_tol, noise_floor=1e-4,
                    frozen_grid=frozen_grid,
                )
            sec_us = section_times_us(prof_ctx)
            sec_log_entry = {k: v / 1e6 for k, v in sec_us.items()}
            section_log[it + 1] = sec_log_entry
        else:
            model.compute_gradients(
                trace_samples=1, cg_tol=cg_tol, noise_floor=1e-4,
                frozen_grid=frozen_grid,
            )

        optimizer.step()
        dt = time.perf_counter() - t0

        s = model.last_gradient_stats
        rec = {
            "iter": it + 1,
            "wall_sec": dt,
            "refreshed": refreshed,
            "lengthscale": float(model.kernel.get_hyper("lengthscale")),
            "variance": float(model.kernel.get_hyper("variance")),
            "sigmasq": float(model._gp_params.sig2.item()),
            "mean_cg_iters": int(s.get("mean_cg_iters")),
            "trace_cg_iters": int(s.get("trace_cg_iters")),
            "mtot": s.get("mtot") if not isinstance(s.get("mtot"), tuple) else list(s.get("mtot")),
            "feature_count": int(s.get("feature_count")),
        }
        log.append(rec)
        flag = "*" if refreshed else " "
        prof_flag = " [PROF]" if do_prof else ""
        print(
            f"{flag} iter {it+1:>3}  {dt:6.2f}s  "
            f"ℓ={rec['lengthscale']:.4g}  M={rec['feature_count']:>6}  "
            f"cg(m/t)={rec['mean_cg_iters']}/{rec['trace_cg_iters']}{prof_flag}",
            flush=True,
        )

        if do_prof:
            print(f"   section breakdown (s) for iter {it+1}:", flush=True)
            total = sum(section_log[it + 1].values())
            for sec in SECTIONS:
                v = section_log[it + 1][sec]
                pct = 100 * v / total if total > 0 else 0
                print(f"     {sec:<28} {v:6.3f}  ({pct:4.1f}%)", flush=True)
            print(f"     {'sum-of-sections':<28} {total:6.3f}", flush=True)

        # snapshot to disk after each iter
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(
            {"records": log, "sections": section_log}, indent=2,
        ))

    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
