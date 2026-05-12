"""Save hyperparameter_comparison.ipynb big-n results and make rollout figures.

Values are transcribed from the notebook cell outputs (n=250000, d=2).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

OUT_DIR = Path(__file__).parent
f_var = 0.9758  # Var(f_true) from cell 26
n_big = 250_000
d = 2

results = {
    "f_var": f_var,
    "n": n_big,
    "d": d,
    "methods": {
        "SGPR M=50":   {"fit_time": 38.49,   "pred_time": 0.29,  "mse": 0.452409},
        "SGPR M=250":  {"fit_time": 228.36,  "pred_time": 2.07,  "mse": 0.100026},
        "SGPR M=1000": {"fit_time": 2552.88, "pred_time": 21.91, "mse": 0.006244},
        "SKI":         {"fit_time": 5931.52, "pred_time": 0.09,  "mse": 0.000176},
        "EFGP eps=1e-3": {"fit_time": 7.10,  "pred_time": 0.02,  "mse": 0.000089},
    },
}
for m, r in results["methods"].items():
    r["total_time"] = r["fit_time"] + r["pred_time"]
    r["rel_mse"] = r["mse"] / f_var

with open(OUT_DIR / "hyperparam_comparison_results.json", "w") as fh:
    json.dump(results, fh, indent=2)

styles = {
    "SGPR M=50":   {"color": "red",    "marker": "s"},
    "SGPR M=250":  {"color": "orange", "marker": "s"},
    "SGPR M=1000": {"color": "purple", "marker": "s"},
    "SKI":         {"color": "gray",   "marker": "^"},
    "EFGP eps=1e-3": {"color": "dodgerblue", "marker": "o"},
}

rollout_order = ["SGPR M=50", "SGPR M=250", "SGPR M=1000", "SKI", "EFGP eps=1e-3"]

XLIM = (3, 2e4)
YLIM = (5e-5, 1.5)


def base_axes():
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.set_xlabel("Total wall time: fit + predict (seconds)", fontsize=11)
    ax.set_ylabel("Relative MSE  (MSE / Var(f))", fontsize=11)
    ax.set_title(f"Accuracy vs Cost (n={n_big:,}, d={d})", fontsize=13)
    ax.grid(True, which="major", alpha=0.3, linestyle="-")
    ax.grid(True, which="minor", alpha=0.15, linestyle=":")
    return fig, ax


def plot_point(ax, name):
    r = results["methods"][name]
    s = styles[name]
    ax.scatter(
        r["total_time"], r["rel_mse"], s=160, color=s["color"], marker=s["marker"],
        edgecolors="black", linewidths=0.8, zorder=10,
    )
    ax.annotate(
        name, (r["total_time"], r["rel_mse"]),
        textcoords="offset points", xytext=(10, 6),
        fontsize=10, fontweight="bold", color=s["color"],
    )


for k in range(len(rollout_order) + 1):
    fig, ax = base_axes()
    for name in rollout_order[:k]:
        plot_point(ax, name)
    fig.tight_layout()
    fname = OUT_DIR / f"hyperparam_rollout_{k}.png"
    fig.savefig(fname, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fname.name}")

print("done")
