"""
Compare hyperparameter optimization with:
  (a) fixed epsilon (grid rebuilt each step as kernel hypers drift)
  (b) fixed quadrature grid (xis_1d, h frozen at init)

Uses a fixed cg_tol = 1e-5 so we isolate the effect of quadrature tolerance
(eps) from CG solver tolerance.

NOTE: we read params via model._gp_params.pos each step instead of using
optimize_hyperparameters' training_log, because that log pulls from a stale
_params_dict cache on the kernel object.
"""

import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.optim import Adam

from kernels.squared_exponential import SquaredExponential
from efgpnd import EFGPND
from utils.kernels import get_xis
from vanilla_gp_sampling import sample_gp_rff

dtype = torch.float64

# --- data ---------------------------------------------------------------
n, d = 20_000, 2
TRUE_LS = 0.05
TRUE_SF2 = 1.0
TRUE_NOISE = 0.01

torch.manual_seed(1)
x = torch.rand(n, d, dtype=dtype)
f = sample_gp_rff(x, length_scale=TRUE_LS, variance=TRUE_SF2, num_features=5000, seed=0)
torch.manual_seed(1)
y = f + torch.sqrt(torch.tensor(TRUE_NOISE, dtype=dtype)) * torch.randn(n, dtype=dtype)
print(f"Data: n={n}, d={d}, y std={y.std():.3f}")

# --- config -------------------------------------------------------------
INIT_LS = 0.3
INIT_SF2 = 1.0
INIT_NOISE = 0.3
MAX_ITERS = 50
LR = 0.5
TRACE_SAMPLES = 1
NOISE_FLOOR = 1e-5
CG_TOL = 1e-5  # fixed, tight CG tolerance
FROZEN_LS_FLOOR = 0.03  # build frozen-headroom grid for this (smaller than truth 0.05)

eps_list = [1e-1, 1e-2, 1e-3, 1e-4]
MODES = ["fixed_eps", "frozen_init", "frozen_headroom"]


def read_pos(model):
    p = model._gp_params.pos.detach()
    return float(p[0]), float(p[1]), float(p[2])  # ls, sf2, noise


def run(eps, mode):
    """mode in {'fixed_eps', 'frozen_init', 'frozen_headroom'}"""
    torch.manual_seed(1)
    kernel = SquaredExponential(dimension=d, init_lengthscale=INIT_LS, init_variance=INIT_SF2)
    model = EFGPND(x, y, kernel=kernel, sigmasq=INIT_NOISE, eps=eps, estimate_params=False)

    L_val = float(torch.max(torch.max(x, dim=0).values - torch.min(x, dim=0).values))
    fg = None
    if mode == "frozen_init":
        xis_1d_f, h_f, mtot_f = get_xis(kernel_obj=kernel, eps=eps, L=L_val,
                                         use_integral=True, l2scaled=False)
        fg = (xis_1d_f.to(dtype=dtype), float(h_f))
        print(f"  frozen @ init: mtot_1d={mtot_f}, h={h_f:.4g}")
    elif mode == "frozen_headroom":
        # build grid sized for the smallest ls we expect (pessimistic)
        ker_small = SquaredExponential(dimension=d, init_lengthscale=FROZEN_LS_FLOOR,
                                        init_variance=INIT_SF2)
        xis_1d_f, h_f, mtot_f = get_xis(kernel_obj=ker_small, eps=eps, L=L_val,
                                         use_integral=True, l2scaled=False)
        fg = (xis_1d_f.to(dtype=dtype), float(h_f))
        print(f"  frozen @ headroom (ls={FROZEN_LS_FLOOR}): mtot_1d={mtot_f}, h={h_f:.4g}")

    optimizer = Adam(model._gp_params.parameters(), lr=LR)
    hist = {"ls": [], "sf2": [], "noise": [], "mtot": []}
    ls0, sf20, n0 = read_pos(model)
    hist["ls"].append(ls0); hist["sf2"].append(sf20); hist["noise"].append(n0)

    t0 = time.time()
    for it in range(MAX_ITERS):
        optimizer.zero_grad()
        model.compute_gradients(
            trace_samples=TRACE_SAMPLES,
            cg_tol=CG_TOL,
            noise_floor=NOISE_FLOOR,
            apply_gradients=True,
            frozen_grid=fg,
        )
        optimizer.step()
        lsv, sf2v, nv = read_pos(model)
        hist["ls"].append(lsv); hist["sf2"].append(sf2v); hist["noise"].append(nv)
        hist["mtot"].append(model.last_gradient_stats.get("mtot"))
    dt = time.time() - t0
    return hist, dt


results = {}
for eps in eps_list:
    results[eps] = {}
    for mode in MODES:
        print(f"\n=== eps={eps:g}  mode={mode} ===")
        h, dt = run(eps, mode)
        print(f"  final: ls={h['ls'][-1]:.4f}  sf2={h['sf2'][-1]:.4f}  "
              f"noise={h['noise'][-1]:.4f}  time={dt:.1f}s  "
              f"mtot_end={h['mtot'][-1]}")
        results[eps][mode] = (h, dt)


# --- summary ------------------------------------------------------------
print("\n==================== FINAL HYPERS ====================")
print(f"{'eps':>10} {'mode':>18} {'ls':>10} {'sf2':>10} {'noise':>10} {'mtot_end':>10} {'time(s)':>10}")
print(f"{'TRUE':>10} {'':>18} {TRUE_LS:>10.4f} {TRUE_SF2:>10.4f} {TRUE_NOISE:>10.4f}")
for eps in eps_list:
    for mode in MODES:
        h, dt = results[eps][mode]
        mtot_end = h["mtot"][-1] if h["mtot"] else None
        print(f"{eps:>10.0e} {mode:>18} {h['ls'][-1]:>10.4f} {h['sf2'][-1]:>10.4f} "
              f"{h['noise'][-1]:>10.4f} {str(mtot_end):>10} {dt:>10.1f}")


# --- plot ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
param_keys = ["ls", "sf2", "noise"]
truths = [TRUE_LS, TRUE_SF2, TRUE_NOISE]
labels = ["lengthscale", "variance (sf2)", "noise"]

mode_styles = {"fixed_eps": "--", "frozen_init": ":", "frozen_headroom": "-"}
cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(eps_list)))
for ax, key, truth, lbl in zip(axes, param_keys, truths, labels):
    for ci, eps in enumerate(eps_list):
        for mode in MODES:
            h, _ = results[eps][mode]
            ax.plot(h[key], color=cmap[ci], linestyle=mode_styles[mode], lw=1.3,
                    label=f"eps={eps:.0e} {mode}" if key == "sf2" else None)
    ax.axhline(truth, color="red", lw=0.9, label="truth" if key == "sf2" else None)
    ax.set_xlabel("iter")
    ax.set_title(lbl)
    if key != "ls":
        ax.set_yscale("log")
axes[1].legend(fontsize=6, loc="best", ncol=2)
fig.suptitle(f"EFGP hyperopt: fixed-eps vs frozen quadrature grid  (n={n}, d={d}, cg_tol={CG_TOL:g})")
plt.tight_layout()
plt.savefig("frozen_quad_grid_comparison.png", dpi=130)
print("\nSaved plot to frozen_quad_grid_comparison.png")
