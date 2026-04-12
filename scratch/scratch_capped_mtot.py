"""
Test capped-mtot training: adaptive quadrature grid each step, but capped at
the M_cap that get_xis would produce at (eps_design, ls_design=0.03).

Compares three modes per eps:
  (a) fixed_eps         — no cap (original behavior)
  (b) capped            — adaptive with M_cap ceiling derived from ls_design=0.03
  (c) frozen_headroom   — static grid built once at (eps, ls_design=0.03)
"""
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.optim import Adam
from copy import deepcopy

from kernels.squared_exponential import SquaredExponential
from efgpnd import EFGPND
from utils.kernels import get_xis
from vanilla_gp_sampling import sample_gp_rff

dtype = torch.float64

# --- data ---------------------------------------------------------------
n, d = 20_000, 2
TRUE_LS, TRUE_SF2, TRUE_NOISE = 0.05, 1.0, 0.01

torch.manual_seed(1)
x = torch.rand(n, d, dtype=dtype)
f = sample_gp_rff(x, length_scale=TRUE_LS, variance=TRUE_SF2, num_features=5000, seed=0)
torch.manual_seed(1)
y = f + torch.sqrt(torch.tensor(TRUE_NOISE, dtype=dtype)) * torch.randn(n, dtype=dtype)
print(f"Data: n={n}, d={d}, y std={y.std():.3f}")

INIT_LS, INIT_SF2, INIT_NOISE = 0.3, 1.0, 0.3
MAX_ITERS = 50
LR = 0.5
TRACE_SAMPLES = 1
NOISE_FLOOR = 1e-5
CG_TOL = 1e-5
LS_DESIGN = 0.03

eps_list = [1e-2, 1e-3, 1e-4]
MODES = ["fixed_eps", "capped", "frozen_headroom"]


def read_pos(model):
    p = model._gp_params.pos.detach()
    return float(p[0]), float(p[1]), float(p[2])


def compute_M_cap(eps):
    L_val = 1.0
    ker_design = SquaredExponential(dimension=d, init_lengthscale=LS_DESIGN, init_variance=1.0)
    _, _, m_cap = get_xis(kernel_obj=ker_design, eps=eps, L=L_val,
                          use_integral=True, l2scaled=False)
    return int(m_cap)


def build_frozen(eps):
    L_val = 1.0
    ker_design = SquaredExponential(dimension=d, init_lengthscale=LS_DESIGN, init_variance=1.0)
    xis_1d, h, mtot = get_xis(kernel_obj=ker_design, eps=eps, L=L_val,
                              use_integral=True, l2scaled=False)
    return xis_1d.to(dtype=dtype), float(h), int(mtot)


def run(eps, mode):
    torch.manual_seed(1)
    kernel = SquaredExponential(dimension=d, init_lengthscale=INIT_LS, init_variance=INIT_SF2)
    model = EFGPND(x, y, kernel=kernel, sigmasq=INIT_NOISE, eps=eps, estimate_params=False)

    fg = None
    m_cap = None
    if mode == "capped":
        m_cap = compute_M_cap(eps)
        print(f"  capped: M_cap={m_cap}")
    elif mode == "frozen_headroom":
        xis_1d_f, h_f, mtot_f = build_frozen(eps)
        fg = (xis_1d_f, h_f)
        print(f"  frozen: mtot={mtot_f}, h={h_f:.4g}")

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


results = {}
for eps in eps_list:
    results[eps] = {}
    for mode in MODES:
        print(f"\n=== eps={eps:g}  mode={mode} ===")
        h, dt = run(eps, mode)
        print(f"  final: ls={h['ls'][-1]:.4f}  sf2={h['sf2'][-1]:.4f}  "
              f"noise={h['noise'][-1]:.4f}  time={dt:.1f}s  mtot_end={h['mtot'][-1]}")
        results[eps][mode] = (h, dt)


print("\n==================== FINAL HYPERS ====================")
print(f"{'eps':>10} {'mode':>18} {'ls':>10} {'sf2':>10} {'noise':>10} {'mtot':>8} {'time':>8}")
print(f"{'TRUE':>10} {'':>18} {TRUE_LS:>10.4f} {TRUE_SF2:>10.4f} {TRUE_NOISE:>10.4f}")
for eps in eps_list:
    for mode in MODES:
        h, dt = results[eps][mode]
        print(f"{eps:>10.0e} {mode:>18} {h['ls'][-1]:>10.4f} {h['sf2'][-1]:>10.4f} "
              f"{h['noise'][-1]:>10.4f} {str(h['mtot'][-1]):>8} {dt:>8.1f}")


# --- plot ---------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(18, 4.3))
keys = ["ls", "sf2", "noise", "mtot"]
truths = [TRUE_LS, TRUE_SF2, TRUE_NOISE, None]
mode_styles = {"fixed_eps": "--", "capped": "-", "frozen_headroom": ":"}
cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(eps_list)))
for ax, key, truth in zip(axes, keys, truths):
    for ci, eps in enumerate(eps_list):
        for mode in MODES:
            h, _ = results[eps][mode]
            series = h[key]
            if key == "mtot":
                series = [m for m in series if m is not None]
            ax.plot(series, color=cmap[ci], linestyle=mode_styles[mode], lw=1.3,
                    label=f"eps={eps:.0e} {mode}" if key == "sf2" else None)
    if truth is not None:
        ax.axhline(truth, color="red", lw=0.9, label="truth" if key == "sf2" else None)
    ax.set_xlabel("iter"); ax.set_title(key)
    if key in ("sf2", "noise"):
        ax.set_yscale("log")
axes[1].legend(fontsize=6, loc="best", ncol=2)
fig.suptitle(f"Capped mtot (ls_design={LS_DESIGN}) vs fixed_eps vs frozen  (n={n}, d={d})")
plt.tight_layout()
plt.savefig("capped_mtot_comparison.png", dpi=130)
print("\nSaved plot to capped_mtot_comparison.png")
