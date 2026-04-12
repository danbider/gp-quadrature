"""
Degradation test: ls_design=0.03, truth ls varies from 0.05 down to 0.005.
For true_ls < ls_design, both capped and frozen modes under-resolve the
spectral support. Does the optimizer still recover ls, sf2, noise?

Modes (all at eps=1e-4):
  (a) fixed_eps          — adaptive, oracle baseline
  (b) capped             — adaptive with M_cap from (eps, ls_design)
  (c) frozen_headroom    — static grid from (eps, ls_design)
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

# --- fixed config -------------------------------------------------------
n, d = 50_000, 2
TRUE_SF2 = 1.0
TRUE_NOISE = 0.01
LS_DESIGN = 0.03
EPS = 1e-4
CG_TOL = 1e-5
INIT_LS, INIT_SF2, INIT_NOISE = 0.3, 1.0, 0.3
MAX_ITERS = 80
LR = 0.3
TRACE_SAMPLES = 1
NOISE_FLOOR = 1e-5

true_ls_list = [0.05, 0.02, 0.01, 0.005]
MODES = ["fixed_eps", "capped", "frozen_headroom"]


def make_data(true_ls, seed=1):
    torch.manual_seed(seed)
    xx = torch.rand(n, d, dtype=dtype)
    fv = sample_gp_rff(xx, length_scale=true_ls, variance=TRUE_SF2,
                       num_features=5000, seed=0)
    torch.manual_seed(seed)
    yy = fv + torch.sqrt(torch.tensor(TRUE_NOISE, dtype=dtype)) * torch.randn(n, dtype=dtype)
    return xx, yy


def read_pos(model):
    p = model._gp_params.pos.detach()
    return float(p[0]), float(p[1]), float(p[2])


def compute_design_grid():
    ker = SquaredExponential(dimension=d, init_lengthscale=LS_DESIGN, init_variance=1.0)
    xis_1d, h, m_cap = get_xis(kernel_obj=ker, eps=EPS, L=1.0,
                               use_integral=True, l2scaled=False)
    return xis_1d.to(dtype=dtype), float(h), int(m_cap)


def run(x, y, mode, design_xis_1d, design_h, design_mtot):
    torch.manual_seed(1)
    kernel = SquaredExponential(dimension=d, init_lengthscale=INIT_LS, init_variance=INIT_SF2)
    model = EFGPND(x, y, kernel=kernel, sigmasq=INIT_NOISE, eps=EPS, estimate_params=False)

    fg = None
    m_cap = None
    if mode == "capped":
        m_cap = design_mtot
    elif mode == "frozen_headroom":
        fg = (design_xis_1d, design_h)

    optimizer = Adam(model._gp_params.parameters(), lr=LR)
    hist = {"ls": [], "sf2": [], "noise": [], "mtot": []}
    ls0, sf20, n0 = read_pos(model)
    hist["ls"].append(ls0); hist["sf2"].append(sf20); hist["noise"].append(n0)

    t0 = time.time()
    for it in range(MAX_ITERS):
        optimizer.zero_grad()
        model.compute_gradients(
            trace_samples=TRACE_SAMPLES, cg_tol=CG_TOL,
            noise_floor=NOISE_FLOOR, apply_gradients=True,
            frozen_grid=fg, max_mtot_1d=m_cap,
        )
        optimizer.step()
        lsv, sf2v, nv = read_pos(model)
        hist["ls"].append(lsv); hist["sf2"].append(sf2v); hist["noise"].append(nv)
        hist["mtot"].append(model.last_gradient_stats.get("mtot"))
    dt = time.time() - t0
    return hist, dt


design_xis, design_h, design_mtot = compute_design_grid()
print(f"Design grid: ls_design={LS_DESIGN}, eps={EPS:g}  ->  mtot_1d={design_mtot}, h={design_h:.4g}")
print(f"Total design features: mtot^d = {design_mtot**d}")

results = {}
for true_ls in true_ls_list:
    print(f"\n######## TRUE_LS = {true_ls} ########")
    x, y = make_data(true_ls)
    results[true_ls] = {}
    for mode in MODES:
        print(f"\n--- mode={mode} ---")
        h, dt = run(x, y, mode, design_xis, design_h, design_mtot)
        print(f"  final: ls={h['ls'][-1]:.4f}  sf2={h['sf2'][-1]:.4f}  "
              f"noise={h['noise'][-1]:.4f}  time={dt:.1f}s  mtot_end={h['mtot'][-1]}")
        results[true_ls][mode] = (h, dt)


print("\n==================== FINAL HYPERS ====================")
print(f"{'true_ls':>10} {'mode':>18} {'ls_final':>10} {'sf2':>10} {'noise':>10} {'mtot':>8} {'time':>8}")
for true_ls in true_ls_list:
    for mode in MODES:
        h, dt = results[true_ls][mode]
        print(f"{true_ls:>10.4f} {mode:>18} {h['ls'][-1]:>10.4f} {h['sf2'][-1]:>10.4f} "
              f"{h['noise'][-1]:>10.4f} {str(h['mtot'][-1]):>8} {dt:>8.1f}")

# --- plot ---------------------------------------------------------------
fig, axes = plt.subplots(len(true_ls_list), 4, figsize=(18, 4*len(true_ls_list)))
keys = ["ls", "sf2", "noise", "mtot"]
mode_styles = {"fixed_eps": "-", "capped": "--", "frozen_headroom": ":"}
mode_colors = {"fixed_eps": "tab:blue", "capped": "tab:orange", "frozen_headroom": "tab:green"}
for row, true_ls in enumerate(true_ls_list):
    truths = [true_ls, TRUE_SF2, TRUE_NOISE, None]
    for col, (key, truth) in enumerate(zip(keys, truths)):
        ax = axes[row, col]
        for mode in MODES:
            h, _ = results[true_ls][mode]
            series = h[key]
            if key == "mtot":
                series = [m for m in series if m is not None]
            ax.plot(series, color=mode_colors[mode], linestyle=mode_styles[mode],
                    lw=1.5, label=mode)
        if truth is not None:
            ax.axhline(truth, color="red", lw=0.9, label="truth")
        ax.set_xlabel("iter")
        ax.set_title(f"true_ls={true_ls}  :  {key}")
        if key in ("sf2", "noise", "ls"):
            ax.set_yscale("log")
    axes[row, 0].legend(fontsize=7)
fig.suptitle(f"Degradation vs truth<ls_design  (ls_design={LS_DESIGN}, eps={EPS:g}, n={n}, d={d})")
plt.tight_layout()
plt.savefig("degradation_sweep.png", dpi=130)
print("\nSaved plot to degradation_sweep.png")
