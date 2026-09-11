"""
Paper Figure 1 -- EFGP framework overview (compact, double-column).

Self-contained generator (consolidates the old scratch/fig1_compact.py +
scratch/fig1_compact_v2.py two-file split into one file). Layout:
  A  data from diverse scientific domains        (full-width thumbnail row)
  B  fast GP inference via equispaced Fourier features
        kernel -> Bochner spectral density -> equispaced grid -> Toeplitz op -> posterior
  C  methods for core GP tasks                   (2x2 task icons, bottom-left)
  D  faster and more accurate                    (cost-vs-accuracy Pareto, bottom-right)

Panel D reads the scaling benchmark results from  scaling/scaling_data.json
(regenerate those with  paper_figures/scaling/run_benchmark.py  -- see
paper_figures/scaling/README.md).  The x-axis is wall-clock for hyperparameter
learning (50 iters) PLUS posterior prediction; the y-axis is latent-recovery nRMSE
measured at held-out TEST points (not training points).

Output ->  <repo>/fig1_compact_v2.png   (referenced by the paper's main.tex, Fig 1).
Run:  ~/myenv/bin/python paper_figures/fig1_overview.py    (from anywhere)

Geometry conventions (keep these when editing -- they are what makes the panel
boxes read as a single system):
  * All placement is in figure fractions on a fixed FIGSIZE canvas; the save
    then trims the uniform outer margin (cropping moves the canvas edges, it
    does not rescale the axes, so squares stay square).
  * sq(h) converts a height (figure fraction) to the width of a true square, so
    image thumbnails and the Toeplitz matrix are never stretched.
  * Panel B's five cartoons share one top edge (B_TOP) and one bottom edge
    (B_BOT); only their widths differ. Sub-labels sit on one line (B_LAB), the
    connecting arrows on one line (B_MID).  No prose annotations in B -- the
    figure states the pipeline, the text does the interpreting.
  * Panel A's thumbnails share one baseline, as do panel C's two icon rows.
  * C's grey card and D's axes FRAME share CD_TOP/CD_BOT, so the two bottom
    panels have identical visible footprints -- D's tick and axis labels hang
    outside the frame rather than shrinking it.

To edit: A/B are schematic cartoons (knobs: ELL, the B_* row constants + the
BOXES width table in panel_B); C is the task-icon grid; D is the real benchmark
plot (colours/labels in _STYLE / the annotate calls). EFGP is the only saturated
("hero") colour; baselines are muted.
"""
import os, json, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent          # paper_figures/
ROOT = HERE.parent                              # repo root
os.chdir(ROOT)                                  # so 'figures/...' + output paths resolve at repo root
SCALING_DATA = HERE / "scaling" / "scaling_data.json"
OUT = ROOT / "fig1_compact_v2"              # <- main.tex \includegraphics path (Fig 1)

# ---------------------------------------------------------------- style / config
import pnas_style as ST
ST.apply(plt)
# 5.0 in tall before: at \linewidth that printed 4.5 in.  The bands compress by
# 0.85 here while the type drops from 10.5/12.5 to pnas_style's 8.5/11 (0.81/0.88),
# so text has slightly MORE room inside each band than it did.
FIGSIZE = (ST.FIG_W, 4.25)
ASP = FIGSIZE[1] / FIGSIZE[0]                    # fig-fraction width of a height-1 square
MARG_L, MARG_R = 0.035, 0.965                    # shared left/right content edges
ELL = 0.16                                       # kernel lengthscale for the panel-B cartoon
COL = dict(four="#3a6b4f", space="#3a5e7a", red="#9c4a39", grey="#8a939c", band="#c3d4e2")
SPINE = ST.SPINE
FRAME_LW = 0.7
FS_TITLE, FS_LETTER, FS_SUB = ST.FS_TITLE, ST.FS_LETTER, ST.FS_LABEL
X = np.linspace(0, 1, 500)


def sq(h):
    """Width (figure fraction) of a square whose height is `h` figure fractions."""
    return h * ASP


# ---------------------------------------------------------------- helpers
def fig_text(fig, x, y, s, fs=11, b=False, c="black", ha="center", va="center", it=False, z=6):
    fig.text(x, y, s, fontsize=fs, fontweight="bold" if b else "normal", color=c,
             ha=ha, va=va, style="italic" if it else "normal", zorder=z)

def panel_head(fig, y, letter, title):
    """Panel letter + title on a shared baseline, so all four headers line up."""
    fig_text(fig, MARG_L, y, letter, fs=FS_LETTER, b=True, ha="left")
    fig_text(fig, MARG_L + 0.027, y, title, fs=FS_TITLE, b=True, ha="left")

def arrow(fig, x1, y1, x2, y2, c="#777", lw=1.5, mut=12):
    fig.add_artist(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=mut,
                   lw=lw, color=c, connectionstyle="arc3,rad=0", zorder=4, transform=fig.transFigure))

def style(ax, c=SPINE, lw=FRAME_LW):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color(c); sp.set_linewidth(lw)

def kern(t):  return np.exp(-t ** 2 / (2 * ELL ** 2))
def khat(x):  return np.exp(-2 * np.pi ** 2 * ELL ** 2 * x ** 2)

def smooth_fn(seed, nmodes=8, decay=0.28):
    rng = np.random.default_rng(seed)
    k = np.arange(1, nmodes + 1); a = np.exp(-(decay * k) ** 2)
    ph = rng.random(nmodes) * 2 * np.pi; s = rng.choice([-1, 1], nmodes)
    f = sum(si * ai * np.cos(2 * np.pi * ki * X + pi) for ki, ai, pi, si in zip(k, a, ph, s))
    return (f - f.mean()) / f.std()

F_TRUE = smooth_fn(2)
_rng = np.random.default_rng(8)
DATA_X = np.sort(np.random.default_rng(0).random(13))
def band_of(dx, base, scale): return base + scale * np.min(np.abs(X[:, None] - dx[None, :]), axis=1)

def sqim(path):
    im = plt.imread(path); hh, ww = im.shape[:2]; s = min(hh, ww)
    return im[(hh - s) // 2:(hh - s) // 2 + s, (ww - s) // 2:(ww - s) // 2 + s]


# ============================================================ A: inputs
A_TITLE = 0.975
A_TOP, A_H = 0.952, 0.120                        # thumbnails: shared top edge + square height
A_LAB = A_TOP - A_H - 0.018

def panel_A(fig):
    panel_head(fig, A_TITLE, "A", "Data from diverse scientific domains")
    w = sq(A_H)
    thumbs = [("Sea-surface temperature", 0.215, "figures/sst_thumb.png"),
              ("Spatial transcriptomics", 0.500, "figures/transcriptomics_thumb.png"),
              ("Neural population dynamics", 0.785, "figures/neural_thumb.png")]
    for name, cx, path in thumbs:
        a = fig.add_axes([cx - w / 2, A_TOP - A_H, w, A_H]); a.set_zorder(3)
        a.imshow(sqim(path), aspect="auto"); style(a)
        fig_text(fig, cx, A_LAB, name, fs=7.6, c="#444")


# ============================================================ B: the spectral engine
B_TITLE = 0.775
B_LAB = 0.741                                    # one line for all five sub-labels
B_TOP, B_BOT = 0.726, 0.556                      # shared top/bottom edges of the five cartoons
B_MID = 0.5 * (B_TOP + B_BOT)                    # arrow line
B_H = B_TOP - B_BOT
# widths of the five cartoons; the Toeplitz tile is a true square, the rest are
# wider than tall.  Gaps (for the arrows) are whatever is left over, split evenly.
BOXES = [0.135, 0.135, 0.155, sq(B_H), 0.230]

def _b_slots():
    span = MARG_R - MARG_L
    gap = (span - sum(BOXES)) / (len(BOXES) - 1)
    xs, x = [], MARG_L
    for w in BOXES:
        xs.append((x, w)); x += w + gap
    return xs, gap

def panel_B(fig):
    panel_head(fig, B_TITLE, "B", "Fast GP inference via equispaced Fourier features")
    (slots, gap) = _b_slots()
    def rect(i):  return [slots[i][0], B_BOT, slots[i][1], B_H]
    def cx(i):    return slots[i][0] + slots[i][1] / 2
    def link(i):  # arrow from box i to box i+1, inset a little from both edges
        x1 = slots[i][0] + slots[i][1] + 0.16 * gap
        x2 = slots[i + 1][0] - 0.16 * gap
        arrow(fig, x1, B_MID, x2, B_MID, c="#aaa", lw=1.2, mut=10)

    tau = np.linspace(-1.1, 1.1, 400); xi = np.linspace(-4, 4, 400)

    # --- covariance kernel
    ax = fig.add_axes(rect(0))
    ax.plot(tau, kern(tau), color=COL["four"], lw=1.7)
    ax.axhline(0, color="#eee", lw=0.5); ax.set_ylim(-0.15, 1.12); style(ax)
    fig_text(fig, cx(0), B_LAB, "covariance kernel", fs=FS_SUB, c="#333")
    link(0)

    # --- spectral density
    ax = fig.add_axes(rect(1))
    ax.plot(xi, khat(xi), color=COL["space"], lw=1.7)
    ax.axhline(0, color="#eee", lw=0.5); ax.set_ylim(-0.13, 1.12); style(ax)
    fig_text(fig, cx(1), B_LAB, "spectral density", fs=FS_SUB, c="#333")
    link(1)

    # --- equispaced frequencies (the highlighted step)
    xmax = 2.55; nodes = np.arange(-2.4, 2.41, 0.42)
    ax = fig.add_axes(rect(2))
    ax.plot(xi, khat(xi), color=COL["space"], lw=1.1, alpha=0.5)
    ax.fill_between(xi[np.abs(xi) > xmax], 0, khat(xi[np.abs(xi) > xmax]), color=COL["red"], alpha=0.3)
    mk, sl, bl = ax.stem(nodes, khat(nodes), basefmt=" ")
    plt.setp(sl, color=COL["space"], lw=0.9); plt.setp(mk, color=COL["space"], ms=2.8)
    ax.axvline(xmax, color=COL["red"], lw=0.7, ls=":"); ax.axvline(-xmax, color=COL["red"], lw=0.7, ls=":")
    ax.set_ylim(-0.05, 1.12); ax.set_xlim(-3.4, 3.4); style(ax, c=COL["four"], lw=1.0)
    fig_text(fig, cx(2), B_LAB, "equispaced frequencies", fs=FS_SUB, b=True, c=COL["four"])
    link(2)

    # --- Toeplitz operator (square tile, same height as its neighbours)
    mm = 5
    axm = fig.add_axes(rect(3)); axm.set_zorder(3)
    Dg = np.subtract.outer(np.arange(mm), np.arange(mm))
    axm.imshow(Dg, cmap="coolwarm", vmin=-(mm - 1), vmax=(mm - 1), alpha=0.55, aspect="auto",
               extent=[-0.5, mm - 0.5, mm - 0.5, -0.5])
    for i in range(mm):
        for j in range(mm):
            axm.text(j, i, rf"$t_{{{j - i}}}$", ha="center", va="center", fontsize=6.6, color="#222")
    for g in np.arange(-0.5, mm, 1):
        axm.axhline(g, color="white", lw=1.0); axm.axvline(g, color="white", lw=1.0)
    axm.set_xlim(-0.5, mm - 0.5); axm.set_ylim(mm - 0.5, -0.5)
    style(axm, c=COL["four"], lw=1.0)
    for off, aa in ((0, 0.55), (1, 0.38)):
        axm.add_patch(FancyArrowPatch((off - 0.28, -0.28), (mm - 1, mm - 1 - off),
                      arrowstyle="-|>", mutation_scale=8, lw=0.9, color="#555", alpha=aa, zorder=5))
    fig_text(fig, cx(3), B_LAB, "Toeplitz operator", fs=FS_SUB, c="#333")
    link(3)

    # --- posterior
    ax = fig.add_axes(rect(4))
    b = band_of(DATA_X, 0.10, 2.2)
    ax.fill_between(X, F_TRUE - b, F_TRUE + b, color=COL["band"], alpha=0.65)
    ax.plot(X, F_TRUE, color=COL["space"], lw=1.8)
    ax.scatter(DATA_X, np.interp(DATA_X, X, F_TRUE) + _rng.standard_normal(len(DATA_X)) * 0.10,
               s=12, c="#222", zorder=3, edgecolor="w", lw=0.4)
    style(ax)
    fig_text(fig, cx(4), B_LAB, "posterior mean & uncertainty", fs=FS_SUB, c="#333")


# ============================================================ C: task icons (2x2)
CD_TITLE = 0.500                                 # shared header baseline for C and D
CD_TOP, CD_BOT = 0.474, 0.112                    # shared top/bottom edges of C's card + D's frame
CD_W = 0.380                                     # shared width, so C and D are the same size box
C_BOX = (MARG_L, CD_BOT, CD_W, CD_TOP - CD_BOT)  # x, y, w, h of the grey card
C_ICON_H = 0.098                                 # icon square height
C_ROWS = (0.348, 0.186)                          # bottom edge of the two icon rows
C_COLS = (0.128, 0.322)                          # icon centres (equal padding inside the card)
C_LAB_DY = 0.016                                 # icon bottom -> label top

def panel_C(fig):
    panel_head(fig, CD_TITLE, "C", "Methods for core GP tasks")
    fig.add_artist(FancyBboxPatch(C_BOX[:2], C_BOX[2], C_BOX[3],
                                  boxstyle="round,pad=0,rounding_size=0.012",
                                  fc="#f7f7f5", ec="#dddddd", lw=0.8, zorder=0,
                                  transform=fig.transFigure))
    tasks = [("Regression &\nhyperparameter learning", C_COLS[0], C_ROWS[0]),
             ("Counts &\nclassification",              C_COLS[1], C_ROWS[0]),
             ("Uncertainty\nquantification",           C_COLS[0], C_ROWS[1]),
             ("Latent\ndynamics",                      C_COLS[1], C_ROWS[1])]
    h_c = C_ICON_H; w_c = sq(h_c)
    for name, cx, yb in tasks:
        a = fig.add_axes([cx - w_c / 2, yb, w_c, h_c]); a.set_zorder(3)
        if name.startswith("Regression"):
            a.imshow(np.add.outer(np.sin(np.linspace(0, 4, 20)), np.cos(np.linspace(0, 5, 20))),
                     cmap="RdBu_r", aspect="auto")
        elif name.startswith("Counts"):
            rng = np.random.default_rng(3)
            a.scatter(rng.random(20), rng.random(20), s=13, c=rng.integers(0, 4, 20),
                      cmap="Purples", marker="h"); a.set_xlim(0, 1); a.set_ylim(0, 1)
        elif name.startswith("Uncertainty"):
            xx = np.linspace(0, 1, 80); ff = np.sin(2 * np.pi * xx)
            a.fill_between(xx, ff - 0.5, ff + 0.5, color=COL["band"], alpha=0.6)
            a.plot(xx, ff, color=COL["space"], lw=1.1); a.set_ylim(-1.9, 1.9)
        else:
            a.imshow(sqim("figures/flowfield_thumb.png"), aspect="auto")
        style(a)
        fig_text(fig, cx, yb - C_LAB_DY, name, fs=6.9, c="#444", va="top")


# ============================================================ D: cost-vs-accuracy Pareto
# EFGP is the only saturated ("hero") colour; baselines muted (SKI = violet, not green).
_STYLE = {
    "efgp":     ("EFGP",            COL["space"], "o"),
    "ski":      ("SKI",             "#7a5c99",    "D"),
    "sgpr1024": ("SGPR ($m$=1024)", COL["red"],   "^"),
    "sgpr49":   ("SGPR ($m$=49)",   COL["grey"],  "s"),
}
_DROP = {"oom": "OOM", "timeout": "time-out", "error": "OOM"}
D_AXES = [MARG_R - CD_W, CD_BOT, CD_W, CD_TOP - CD_BOT]   # frame exactly coincident with C's card

def _tlab(n):
    return f"{n // 1000}k" if n < 1_000_000 else f"{n // 1_000_000}M"

def panel_D(fig):
    fig_text(fig, 0.478, CD_TITLE, "D", fs=FS_LETTER, b=True, ha="left")
    fig_text(fig, 0.505, CD_TITLE, "Faster and more accurate", fs=FS_TITLE, b=True, ha="left")
    ax = fig.add_axes(D_AXES); ax.set_zorder(3)

    blob = json.load(open(SCALING_DATA)); by = {}
    for r in blob["results"]:
        by.setdefault(r["method"], []).append(r)
    for m in by:
        by[m].sort(key=lambda r: r["T"])

    # per-point n-label offsets. EFGP cluster is tight: 10k & 100k share ~the same x, so
    # stack them vertically; 500k & 1M sit to the right at separated heights.
    EFF = {10000: (0, 7.5, "center"), 100000: (0, -9.5, "center"), 250000: (0, 8, "center"),
           500000: (6, 4, "left"), 1000000: (6, -6, "left")}
    NOFF = {"sgpr49": (0, -8.5, "center"), "sgpr1024": (0, -8.5, "center"), "ski": (0, -8.5, "center")}
    _EFF_DEFAULT, _NOFF_DEFAULT = (6, 0, "left"), (0, -8.5, "center")

    for m in ("efgp", "ski", "sgpr1024", "sgpr49"):
        recs = by.get(m, []); lab, col, mk = _STYLE[m]
        ok = [r for r in recs if r.get("status") == "ok"]; hero = (m == "efgp")
        if ok:
            ts = [r["time"] for r in ok]; er = [r["nrmse"] for r in ok]
            ax.plot(ts, er, "-", color=col, lw=2.1 if hero else 1.3,
                    alpha=0.95 if hero else 0.8, zorder=6 if hero else 4, label=lab)
            ax.scatter(ts, er, s=32, marker=mk, color=col, edgecolors="white",
                       linewidths=0.6, zorder=7 if hero else 5)
            for r in ok:
                dx, dy, ha = (EFF.get(r["T"], _EFF_DEFAULT) if hero
                              else NOFF.get(m, _NOFF_DEFAULT))
                ax.annotate(_tlab(r["T"]), (r["time"], r["nrmse"]), textcoords="offset points",
                            xytext=(dx, dy), ha=ha, va="center", fontsize=6.0, color="#666", zorder=8)
            drop = next((r for r in recs if r.get("status") in _DROP), None)
            if drop is not None:
                oom_dy = 3 if m == "sgpr1024" else 7   # red SGPR-1024 tag sits a touch lower
                ax.annotate(f"{_DROP[drop['status']]} ($n\\geq${_tlab(drop['T'])})",
                            (ts[-1], er[-1]), textcoords="offset points", xytext=(0, oom_dy),
                            fontsize=6.6, color=col, fontweight="bold", va="bottom", ha="center")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.25, 4e4); ax.set_ylim(1.1e-3, 2.2)   # headroom for the NE legend
    ax.set_xticks([1, 10, 100, 1000, 10000]); ax.set_yticks([0.01, 0.1, 1])
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("learning + prediction wall-clock (s)", fontsize=ST.FS_LABEL, labelpad=2.0)
    ax.set_ylabel("recovery error  (nRMSE)", fontsize=ST.FS_LABEL, labelpad=2.0)
    ax.tick_params(labelsize=7.2, pad=2.0)
    ax.grid(alpha=0.22, which="both", lw=0.4)
    # 'better' direction cross: two charcoal arrows pointing at the EFGP corner.
    cx0, cy0 = 2.5, 0.15
    gr = "#333333"
    ax.annotate("", xy=(0.5, cy0), xytext=(cx0, cy0), arrowprops=dict(arrowstyle="-|>", color=gr, lw=1.3))
    ax.annotate("", xy=(cx0, 0.022), xytext=(cx0, cy0), arrowprops=dict(arrowstyle="-|>", color=gr, lw=1.3))
    ax.text(1.0, 0.19, "faster", fontsize=7.6, color=gr, style="italic", ha="center", va="bottom")
    ax.text(3.2, 0.057, "more accurate", fontsize=7.6, color=gr, style="italic",
            rotation=90, ha="left", va="center")
    # NE corner: kept compact so its left edge clears the SGPR(m=49) run, whose last
    # point sits at ~80 s -- the legend box must start to the right of that.
    leg = ax.legend(loc="upper right", bbox_to_anchor=(0.998, 0.996), ncol=2, fontsize=5.9,
                    frameon=True, edgecolor="#bbbbbb", facecolor="white", framealpha=0.95,
                    handlelength=0.85, handletextpad=0.3, columnspacing=0.65,
                    labelspacing=0.25, borderpad=0.4)
    leg.get_frame().set_linewidth(0.6)
    for sp in ax.spines.values(): sp.set_color("#888"); sp.set_linewidth(FRAME_LW)


# ============================================================ assemble
def build():
    fig = plt.figure(figsize=FIGSIZE); fig.patch.set_facecolor("white")
    panel_A(fig); panel_B(fig); panel_C(fig); panel_D(fig)
    return fig


if __name__ == "__main__":
    fig = build()
    ST.save(fig, str(OUT))
    plt.close(fig)
    print(f"wrote {OUT}")
