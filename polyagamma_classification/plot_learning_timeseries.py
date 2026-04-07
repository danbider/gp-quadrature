from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot hyperparameter learning against elapsed time.")
    parser.add_argument("--histories", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def main() -> int:
    args = parse_args()
    payloads = [load_payload(path) for path in args.histories]
    colors = ["#1982c4", "#ff595e", "#ff924c", "#8ac926", "#6a4c93"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    truth = payloads[0]["truth"]
    fields = [
        ("lengthscale", "Lengthscale", truth["lengthscale"]),
        ("variance", "Variance", truth["variance"]),
        ("total_count", "Total count", truth["total_count"]),
    ]

    for color, payload in zip(colors, payloads):
        history = payload["history"]
        t = [row.get("elapsed_sec", 0.0) for row in history]
        label = str(payload.get("label", payload.get("method", "run")))
        for ax, (field, title, truth_value) in zip(axes, fields):
            y = [row[field] for row in history]
            ax.plot(t, y, linewidth=2.0, color=color, label=label)
            ax.scatter(t[-1], y[-1], color=color, s=30)
            ax.axhline(truth_value, color="black", linestyle=":", linewidth=1.2)
            ax.set_title(title)
            ax.set_xlabel("Elapsed time (s)")
            ax.grid(True, linestyle=":")

    axes[0].set_ylabel("Value")
    axes[0].legend(loc="best", fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
