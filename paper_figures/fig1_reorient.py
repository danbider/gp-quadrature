"""
Paper Figure 1 -- REORIENTED drafts for the "move to page 1/2, strip down" edit.

Two variants, both built to pnas_style (FIG_W = 7.2 in, so the printed type matches
the rest of the figure set) but re-flowed WIDE-AND-SHORT so the figure can sit at the
top of page 1 or 2 without pushing the text far down the column:

  v1  (A + D)      the message-only teaser:
                     A  three domain thumbnails, reoriented from a horizontal strip
                        into a VERTICAL stack on the left  ("GPs are everywhere")
                     D  the cost-vs-accuracy Pareto, enlarged on the right
                        ("EFGP is an order-of-magnitude improvement over SOTA")

  v2  (A + B + D)  same A|D top row, with panel B's five-box spectral pipeline as a
                   full-width strip beneath it (B needs the full page width for its
                   five cartoons).

Panel C ("methods for core GP tasks") is dropped in both, per the feedback.

Panel D reads the real benchmark from  scaling/scaling_data.json  (unchanged from
fig1_overview.py).  Everything else is schematic.

Outputs (repo root):  fig1_reorient_v1_AD.{png,pdf}   fig1_reorient_v2_ABD.{png,pdf}
Run:  ~/myenv/bin/python paper_figures/fig1_reorient.py
"""
import os, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch

HERE = Path(__file__).resolve().parent          # paper_figures/
ROOT = HERE.parent                              # repo root
os.chdir(ROOT)                                  # so 'figures/...' resolve at repo root
SCALING_DATA = HERE / "scaling" / "scaling_data.json"

import pnas_style as ST
ST.apply(plt)

# ---------------------------------------------------------------- palette / config
COL = dict(four="#3a6b4f", space="#3a5e7a", red="#9c4a39", grey="#8a939c", band="#c3d4e2")
SPINE = ST.SPINE
FRAME_LW = 0.7
FS_TITLE, FS_LETTER, FS_SUB = ST.FS_TITLE, ST.FS_LETTER, ST.FS_LABEL
ELL = 0.16                                       # kernel lengthscale for the panel-B cartoon
X = np.linspace(0, 1, 500)

# ASP is the fig-fraction width of a height-1 square; it depends on the canvas, so
# it is (re)set at the top of each build().  sq() reads it.
ASP = 1.0
def sq(h):
    return h * ASP

# ---------------------------------------------------------------- small helpers
def fig_text(fig, x, y, s, fs=11, b=False, c="black", ha="center", va="center", it=False, z=6):
    fig.text(x, y, s, fontsize=fs, fontweight="bold" if b else "normal", color=c,
             ha=ha, va=va, style="italic" if it else "normal", zorder=z)

def panel_head(fig, x, y, letter, title, gap=0.027):
    fig_text(fig, x, y, letter, fs=FS_LETTER, b=True, ha="left")
    fig_text(fig, x + gap, y, title, fs=FS_TITLE, b=True, ha="left")

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

THUMBS = [("Sea-surface temperature", "figures/sst_thumb.png"),
          ("Spatial transcriptomics", "figures/transcriptomics_thumb.png"),
          ("Neural population dynamics", "figures/neural_thumb.png")]


# ============================================================ A: domain thumbnails
def panel_A_rows(fig, x0, title_y, centers, h_thumb, lab_x, fs=7.7, names=None):
    """Three domain thumbnails stacked at column x0, each with its label to the right,
    vertically centred on the thumbnail.  Rows evenly distributed -> no dead space."""
    panel_head(fig, x0, title_y, "A", "Data from diverse scientific domains")
    w = sq(h_thumb)
    items = list(zip([n for n, _ in THUMBS] if names is None else names,
                     [p for _, p in THUMBS]))
    for (name, path), cy in zip(items, centers):
        a = fig.add_axes([x0, cy - h_thumb / 2, w, h_thumb]); a.set_zorder(3)
        a.imshow(sqim(path), aspect="auto"); style(a)
        fig_text(fig, lab_x, cy, name, fs=fs, c="#333", ha="left")


def panel_A_gallery(fig, title_y, cxs, top, h_thumb, label_dy, fs=7.6):
    """Three domain thumbnails in a horizontal gallery, labels centred below each."""
    panel_head(fig, 0.035, title_y, "A", "Data from diverse scientific domains")
    w = sq(h_thumb)
    for (name, path), cx in zip(THUMBS, cxs):
        a = fig.add_axes([cx - w / 2, top - h_thumb, w, h_thumb]); a.set_zorder(3)
        a.imshow(sqim(path), aspect="auto"); style(a)
        fig_text(fig, cx, top - h_thumb - label_dy, name, fs=fs, c="#333", va="top")


# ============================================================ B: five-box spectral pipeline
def panel_B_strip(fig, ml, mr, title_y, lab_y, b_top, b_bot):
    panel_head(fig, ml, title_y, "B", "Fast GP inference via equispaced Fourier features")
    b_h = b_top - b_bot
    b_mid = 0.5 * (b_top + b_bot)
    boxes = [0.135, 0.135, 0.155, sq(b_h), 0.230]
    span = mr - ml
    gap = (span - sum(boxes)) / (len(boxes) - 1)
    slots, x = [], ml
    for w in boxes:
        slots.append((x, w)); x += w + gap

    def rect(i):  return [slots[i][0], b_bot, slots[i][1], b_h]
    def cx(i):    return slots[i][0] + slots[i][1] / 2
    def link(i):
        x1 = slots[i][0] + slots[i][1] + 0.16 * gap
        x2 = slots[i + 1][0] - 0.16 * gap
        arrow(fig, x1, b_mid, x2, b_mid, c="#aaa", lw=1.2, mut=10)

    tau = np.linspace(-1.1, 1.1, 400); xi = np.linspace(-4, 4, 400)

    ax = fig.add_axes(rect(0))
    ax.plot(tau, kern(tau), color=COL["four"], lw=1.7)
    ax.axhline(0, color="#eee", lw=0.5); ax.set_ylim(-0.15, 1.12); style(ax)
    fig_text(fig, cx(0), lab_y, "covariance kernel", fs=FS_SUB, c="#333"); link(0)

    ax = fig.add_axes(rect(1))
    ax.plot(xi, khat(xi), color=COL["space"], lw=1.7)
    ax.axhline(0, color="#eee", lw=0.5); ax.set_ylim(-0.13, 1.12); style(ax)
    fig_text(fig, cx(1), lab_y, "spectral density", fs=FS_SUB, c="#333"); link(1)

    xmax = 2.55; nodes = np.arange(-2.4, 2.41, 0.42)
    ax = fig.add_axes(rect(2))
    ax.plot(xi, khat(xi), color=COL["space"], lw=1.1, alpha=0.5)
    ax.fill_between(xi[np.abs(xi) > xmax], 0, khat(xi[np.abs(xi) > xmax]), color=COL["red"], alpha=0.3)
    mk, sl, bl = ax.stem(nodes, khat(nodes), basefmt=" ")
    plt.setp(sl, color=COL["space"], lw=0.9); plt.setp(mk, color=COL["space"], ms=2.8)
    ax.axvline(xmax, color=COL["red"], lw=0.7, ls=":"); ax.axvline(-xmax, color=COL["red"], lw=0.7, ls=":")
    ax.set_ylim(-0.05, 1.12); ax.set_xlim(-3.4, 3.4); style(ax, c=COL["four"], lw=1.0)
    fig_text(fig, cx(2), lab_y, "equispaced frequencies", fs=FS_SUB, b=True, c=COL["four"]); link(2)

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
    fig_text(fig, cx(3), lab_y, "Toeplitz operator", fs=FS_SUB, c="#333"); link(3)

    ax = fig.add_axes(rect(4))
    b = band_of(DATA_X, 0.10, 2.2)
    ax.fill_between(X, F_TRUE - b, F_TRUE + b, color=COL["band"], alpha=0.65)
    ax.plot(X, F_TRUE, color=COL["space"], lw=1.8)
    ax.scatter(DATA_X, np.interp(DATA_X, X, F_TRUE) + _rng.standard_normal(len(DATA_X)) * 0.10,
               s=12, c="#222", zorder=3, edgecolor="w", lw=0.4)
    style(ax)
    fig_text(fig, cx(4), lab_y, "posterior mean & uncertainty", fs=FS_SUB, c="#333")


# ============================================================ D: cost-vs-accuracy Pareto
_STYLE = {
    "efgp":     ("EFGP",            COL["space"], "o"),
    "ski":      ("SKI",             "#7a5c99",    "D"),
    "sgpr1024": ("SGPR ($m$=1024)", COL["red"],   "^"),
    "sgpr49":   ("SGPR ($m$=49)",   COL["grey"],  "s"),
}
_DROP = {"oom": "OOM", "timeout": "time-out", "error": "OOM"}

def _tlab(n):
    return f"{n // 1000}k" if n < 1_000_000 else f"{n // 1_000_000}M"

def panel_D_plot(fig, letter, letter_x, title_x, title_y, axes_rect):
    fig_text(fig, letter_x, title_y, letter, fs=FS_LETTER, b=True, ha="left")
    fig_text(fig, title_x, title_y, "Faster and more accurate", fs=FS_TITLE, b=True, ha="left")
    ax = fig.add_axes(axes_rect); ax.set_zorder(3)

    blob = json.load(open(SCALING_DATA)); by = {}
    for r in blob["results"]:
        by.setdefault(r["method"], []).append(r)
    for m in by:
        by[m].sort(key=lambda r: r["T"])

    EFF = {10000: (0, 7.5, "center"), 100000: (0, -9.5, "center"), 250000: (0, 8, "center"),
           500000: (6, 4, "left"), 1000000: (6, -6, "left")}
    NOFF = {"sgpr49": (0, -8.5, "center"), "sgpr1024": (0, -8.5, "center"), "ski": (0, -8.5, "center")}
    _EFF_DEFAULT, _NOFF_DEFAULT = (6, 0, "left"), (0, -8.5, "center")
    _err = lambda r: r.get("rmse", r.get("nrmse"))   # 'rmse' (current) or legacy 'nrmse'

    for m in ("efgp", "ski", "sgpr1024", "sgpr49"):
        recs = by.get(m, []); lab, col, mk = _STYLE[m]
        ok = [r for r in recs if r.get("status") == "ok"]; hero = (m == "efgp")
        if ok:
            ts = [r["time"] for r in ok]; er = [_err(r) for r in ok]
            ax.plot(ts, er, "-", color=col, lw=2.1 if hero else 1.3,
                    alpha=0.95 if hero else 0.8, zorder=6 if hero else 4, label=lab)
            ax.scatter(ts, er, s=32, marker=mk, color=col, edgecolors="white",
                       linewidths=0.6, zorder=7 if hero else 5)
            for r in ok:
                dx, dy, ha = (EFF.get(r["T"], _EFF_DEFAULT) if hero else NOFF.get(m, _NOFF_DEFAULT))
                ax.annotate(_tlab(r["T"]), (r["time"], _err(r)), textcoords="offset points",
                            xytext=(dx, dy), ha=ha, va="center", fontsize=6.0, color="#666", zorder=8)
            drop = next((r for r in recs if r.get("status") in _DROP), None)
            if drop is not None:
                oom_dy = 3 if m == "sgpr1024" else 7
                ax.annotate(f"{_DROP[drop['status']]} ($n\\geq${_tlab(drop['T'])})",
                            (ts[-1], er[-1]), textcoords="offset points", xytext=(0, oom_dy),
                            fontsize=6.6, color=col, fontweight="bold", va="bottom", ha="center")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.25, 4e4); ax.set_ylim(1.1e-3, 2.2)
    ax.set_xticks([1, 10, 100, 1000, 10000]); ax.set_yticks([0.01, 0.1, 1])
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("learning + prediction wall-clock (s)", fontsize=ST.FS_LABEL, labelpad=2.0)
    ax.set_ylabel("recovery error  (RMSE)", fontsize=ST.FS_LABEL, labelpad=2.0)
    ax.tick_params(labelsize=7.2, pad=2.0)
    ax.grid(alpha=0.22, which="both", lw=0.4)
    cx0, cy0 = 2.5, 0.15
    gr = "#333333"
    ax.annotate("", xy=(0.5, cy0), xytext=(cx0, cy0), arrowprops=dict(arrowstyle="-|>", color=gr, lw=1.3))
    ax.annotate("", xy=(cx0, 0.022), xytext=(cx0, cy0), arrowprops=dict(arrowstyle="-|>", color=gr, lw=1.3))
    ax.text(1.0, 0.19, "faster", fontsize=7.6, color=gr, style="italic", ha="center", va="bottom")
    ax.text(3.2, 0.057, "more accurate", fontsize=7.6, color=gr, style="italic",
            rotation=90, ha="left", va="center")
    leg = ax.legend(loc="upper right", bbox_to_anchor=(0.998, 0.996), ncol=2, fontsize=5.9,
                    frameon=True, edgecolor="#bbbbbb", facecolor="white", framealpha=0.95,
                    handlelength=0.85, handletextpad=0.3, columnspacing=0.65,
                    labelspacing=0.25, borderpad=0.4)
    leg.get_frame().set_linewidth(0.6)
    for sp in ax.spines.values(): sp.set_color("#888"); sp.set_linewidth(FRAME_LW)


# ============================================================ assemble the two variants
def build_v1():
    """A (domain thumbnails, left) | D (enlarged Pareto, right) -- one compact landscape row."""
    global ASP
    figsize = (ST.FIG_W, 3.5)
    ASP = figsize[1] / figsize[0]
    fig = plt.figure(figsize=figsize); fig.patch.set_facecolor("white")
    # A: three domain thumbnails, left, evenly distributed with labels to the right
    h_thumb = 0.25
    panel_A_rows(fig, x0=0.045, title_y=0.93, centers=(0.71, 0.43, 0.15),
                 h_thumb=h_thumb, lab_x=0.045 + sq(h_thumb) + 0.022)
    # B (this figure's second panel): enlarged Pareto on the right (the hero result)
    panel_D_plot(fig, "B", letter_x=0.50, title_x=0.527, title_y=0.93,
                 axes_rect=[0.585, 0.165, 0.385, 0.60])
    return fig


def build_v2():
    """Stacked in reading order: A (domain gallery) -> B (spectral pipeline) -> D (Pareto).
    B needs the full page width for its five cartoons, so A and D are full-width rows too."""
    global ASP
    figsize = (ST.FIG_W, 4.7)
    ASP = figsize[1] / figsize[0]
    fig = plt.figure(figsize=figsize); fig.patch.set_facecolor("white")
    ml, mr = 0.035, 0.965
    # A: horizontal domain gallery, full width
    panel_A_gallery(fig, title_y=0.965, cxs=(0.19, 0.50, 0.81),
                    top=0.915, h_thumb=0.145, label_dy=0.02)
    # B: full-width spectral pipeline
    panel_B_strip(fig, ml=ml, mr=mr, title_y=0.685, lab_y=0.648,
                  b_top=0.628, b_bot=0.478)
    # C (this figure's third panel): the Pareto, centred beneath at a good aspect
    panel_D_plot(fig, "C", letter_x=0.205, title_x=0.232, title_y=0.43,
                 axes_rect=[0.27, 0.065, 0.45, 0.335])
    return fig


from matplotlib.patches import FancyBboxPatch


def header2(fig, x, y, letter, lines, dy=0.052):
    """Panel letter + a one- or two-line title, first line on baseline y."""
    fig_text(fig, x, y, letter, fs=FS_LETTER, b=True, ha="left")
    tx = x + 0.027
    if len(lines) == 1:
        fig_text(fig, tx, y, lines[0], fs=FS_TITLE, b=True, ha="left")
    else:
        fig_text(fig, tx, y, lines[0], fs=FS_TITLE, b=True, ha="left")
        fig_text(fig, tx, y - dy, lines[1], fs=FS_TITLE, b=True, ha="left")


def _trunc_glyph(fig, rect):
    """Equispaced quadrature nodes under the spectral density, with the truncation tails."""
    ax = fig.add_axes(rect); ax.set_zorder(4); ax.patch.set_alpha(0)
    xi = np.linspace(-4, 4, 220)
    ax.plot(xi, khat(xi), color=COL["space"], lw=0.9, alpha=0.4)
    xmax = 2.4
    ax.fill_between(xi[np.abs(xi) > xmax], 0, khat(xi[np.abs(xi) > xmax]),
                    color=COL["red"], alpha=0.30)
    nodes = np.arange(-2.1, 2.11, 0.6)
    mk, sl, bl = ax.stem(nodes, khat(nodes), basefmt=" ")
    plt.setp(sl, color=COL["space"], lw=0.8); plt.setp(mk, color=COL["space"], ms=2.0)
    ax.axvline(xmax, color=COL["red"], lw=0.6, ls=":"); ax.axvline(-xmax, color=COL["red"], lw=0.6, ls=":")
    ax.set_ylim(-0.05, 1.15); ax.set_xlim(-3.4, 3.4); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)


def _toep_glyph(fig, rect):
    """Diagonal-constant (Toeplitz) tile with the constant-diagonal arrow."""
    ax = fig.add_axes(rect); ax.set_zorder(4)
    mm = 4
    Dg = np.subtract.outer(np.arange(mm), np.arange(mm))
    ax.imshow(Dg, cmap="coolwarm", vmin=-(mm - 1), vmax=(mm - 1), alpha=0.55, aspect="auto",
              extent=[-0.5, mm - 0.5, mm - 0.5, -0.5])
    for g in np.arange(-0.5, mm, 1):
        ax.axhline(g, color="white", lw=0.8); ax.axvline(g, color="white", lw=0.8)
    ax.add_patch(FancyArrowPatch((-0.2, -0.2), (mm - 1.2, mm - 1.2), arrowstyle="-|>",
                 mutation_scale=6, lw=0.8, color="#555", alpha=0.55, zorder=5))
    ax.set_xlim(-0.5, mm - 0.5); ax.set_ylim(mm - 0.5, -0.5); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)


def _spec_glyph(fig, rect):
    """The Bochner spectral density (a smooth bump)."""
    ax = fig.add_axes(rect); ax.set_zorder(4); ax.patch.set_alpha(0)
    xi = np.linspace(-4, 4, 220)
    ax.plot(xi, khat(xi), color=COL["space"], lw=1.5)
    ax.set_ylim(-0.08, 1.15); ax.set_xlim(-3.4, 3.4); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)


def _img_glyph(path):
    """Glyph fn that drops a square thumbnail into a box."""
    def g(fig, rect):
        a = fig.add_axes(rect); a.set_zorder(4)
        a.imshow(sqim(path), aspect="auto"); a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values(): sp.set_visible(False)
    return g


def _box(fig, x, y, w, h, glyph_fn, label, hero=False, fs=6.4, bold=True):
    """A square, thin-bordered box holding a glyph, with a label to its right."""
    fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.006",
                                  fc=ST.SCHEM_FC, ec=(COL["four"] if hero else ST.SCHEM_MID),
                                  lw=1.1, zorder=2, transform=fig.transFigure))
    pad = 0.009
    glyph_fn(fig, [x + pad, y + pad, w - 2 * pad, h - 2 * pad])
    fig_text(fig, x + w + 0.013, y + h / 2, label, fs=fs, b=bold,
             c=(COL["four"] if hero else "#333"), ha="left")


def _boxc(fig, cx, cy, h, glyph_fn, label, hero=False, fs=6.5, bold=True):
    """A square, thin-bordered box centred at (cx, cy) with its label centred below."""
    w = sq(h); x = cx - w / 2; y = cy - h / 2
    fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.006",
                                  fc=ST.SCHEM_FC, ec=(COL["four"] if hero else ST.SCHEM_MID),
                                  lw=1.1, zorder=2, transform=fig.transFigure))
    pad = 0.010
    glyph_fn(fig, [x + pad, y + pad, w - 2 * pad, h - 2 * pad])
    fig_text(fig, cx, y - 0.016, label, fs=fs, b=bold,
             c=(COL["four"] if hero else "#333"), va="top", ha="center")


def _draw_A(fig, cxA, rows, BH):
    """Panel A: three domain thumbnail boxes, centred in the column."""
    header2(fig, 0.030, 0.95, "A", ["Data from diverse", "scientific domains"])
    a_names = ["Sea-surface\ntemperature", "Spatial\ntranscriptomics",
               "Neural population\ndynamics"]
    for name, (_, path), cy in zip(a_names, THUMBS, rows):
        _boxc(fig, cxA, cy, BH, _img_glyph(path), name, fs=6.5, bold=False)


def _divider(fig, x, y0=0.05, y1=0.86):
    from matplotlib.lines import Line2D
    fig.add_artist(Line2D([x, x], [y0, y1], color="#d9d9d9", lw=0.8,
                          transform=fig.transFigure, zorder=1))


def build_flow_v1():
    """Three-step B, with vertical dividers separating the panels so A and B do not read
    as a 1:1 grid.  A (3 domains) | B (spectral density, equispaced approx, Toeplitz) | C."""
    global ASP
    figsize = (ST.FIG_W, 4.1)
    ASP = figsize[1] / figsize[0]
    fig = plt.figure(figsize=figsize); fig.patch.set_facecolor("white")

    BH = 0.190; rows = (0.764, 0.486, 0.208)
    _draw_A(fig, 0.120, rows, BH)
    _divider(fig, 0.228); _divider(fig, 0.442)

    header2(fig, 0.245, 0.95, "B", ["Efficient computation", "in the EFGP basis"])
    steps = [(_spec_glyph, "spectral\ndensity", False),
             (_trunc_glyph, "equispaced\napprox.", True),
             (_toep_glyph, "Toeplitz\noperator", False)]
    for (gfn, lab, hero), cy in zip(steps, rows):
        _boxc(fig, 0.335, cy, BH, gfn, lab, hero=hero, fs=6.5)

    panel_D_plot(fig, "C", letter_x=0.490, title_x=0.517, title_y=0.95,
                 axes_rect=[0.520, 0.150, 0.445, 0.63])
    return fig


def build_flow_v2():
    """Two-step B (equispaced approx, Toeplitz), boxes centred vertically and offset from A's
    three rows, so there is clearly no 1:1 correspondence between A and B."""
    global ASP
    figsize = (ST.FIG_W, 4.1)
    ASP = figsize[1] / figsize[0]
    fig = plt.figure(figsize=figsize); fig.patch.set_facecolor("white")

    BH = 0.190; rows = (0.764, 0.486, 0.208)
    _draw_A(fig, 0.120, rows, BH)

    header2(fig, 0.245, 0.95, "B", ["Efficient computation", "in the EFGP basis"])
    BH2 = 0.220; rows_B = (0.630, 0.270)               # two bigger boxes, centred & staggered
    steps = [(_trunc_glyph, "equispaced\napprox.", True),
             (_toep_glyph, "Toeplitz\noperator", False)]
    for (gfn, lab, hero), cy in zip(steps, rows_B):
        _boxc(fig, 0.335, cy, BH2, gfn, lab, hero=hero, fs=6.6)

    panel_D_plot(fig, "C", letter_x=0.490, title_x=0.517, title_y=0.95,
                 axes_rect=[0.520, 0.150, 0.445, 0.63])
    return fig


if __name__ == "__main__":
    f1 = build_v1();      ST.save(f1, str(ROOT / "fig1_reorient_v1_AD"));    plt.close(f1)
    f2 = build_v2();      ST.save(f2, str(ROOT / "fig1_reorient_v2_ABD"));   plt.close(f2)
    f3 = build_flow_v1(); ST.save(f3, str(ROOT / "fig1_reorient_flow_v1")); plt.close(f3)
    f4 = build_flow_v2(); ST.save(f4, str(ROOT / "fig1_reorient_flow_v2")); plt.close(f4)
    print("done")
