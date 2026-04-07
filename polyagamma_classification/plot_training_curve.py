from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot lengthscale vs variance training paths.")
    parser.add_argument("--pg-history", type=Path, required=True)
    parser.add_argument("--gpcounts-history", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-points", type=int, default=5000)
    parser.add_argument("--label-every", type=int, default=10)
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text())
    return data


def plot_cycles(
    pg_payload: dict[str, object],
    gp_payloads: list[dict[str, object]],
    output: Path,
    *,
    n_points: int,
    label_every: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    pg_history = pg_payload["history"]
    truth = pg_payload["truth"]
    ax.plot(
        [row["lengthscale"] for row in pg_history],
        [row["variance"] for row in pg_history],
        "-o",
        label=str(pg_payload.get("label", "PG")),
        markevery=5,
        linewidth=2.5,
        color="#1982c4",
    )
    _annotate_history(ax, pg_history, color="#1982c4", label_every=label_every)
    gp_colors = ["#ff595e", "#ff924c", "#8ac926", "#6a4c93"]
    for idx, payload in enumerate(gp_payloads):
        history = payload["history"]
        color = gp_colors[idx % len(gp_colors)]
        ax.plot(
            [row["lengthscale"] for row in history],
            [row["variance"] for row in history],
            "-s",
            label=str(payload.get("label", f"GPcounts {idx}")),
            markevery=max(1, len(history) // 8),
            linewidth=2.0,
            color=color,
            alpha=0.95,
        )
        _annotate_history(ax, history, color=color, label_every=label_every)
        ax.scatter(
            history[-1]["lengthscale"],
            history[-1]["variance"],
            marker="D",
            s=80,
            color=color,
        )
    ax.scatter(
        truth["lengthscale"],
        truth["variance"],
        marker="*",
        s=180,
        color="black",
        label="Truth",
    )
    ax.scatter(pg_history[0]["lengthscale"], pg_history[0]["variance"], marker="*", s=120, label="PG init")
    ax.scatter(pg_history[-1]["lengthscale"], pg_history[-1]["variance"], marker="X", s=120, label="PG final")
    ax.set_xlabel("Lengthscale")
    ax.set_ylabel("Variance")
    ax.set_title(f"Training hyperparameter paths (n={n_points})")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, linestyle=":")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _annotate_history(ax, history: list[dict[str, float]], *, color: str, label_every: int) -> None:
    if not history:
        return
    step = max(1, int(label_every))
    indices = sorted(set([0, len(history) - 1] + list(range(0, len(history), step))))
    for idx in indices:
        row = history[idx]
        label = f"{int(row['step'])}|{float(row.get('elapsed_sec', 0.0)):.1f}s"
        ax.annotate(
            label,
            (row["lengthscale"], row["variance"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
            color=color,
            alpha=0.9,
        )


def main() -> int:
    args = parse_args()
    pg_payload = load_payload(args.pg_history)
    gp_payloads = [load_payload(path) for path in args.gpcounts_history]
    plot_cycles(pg_payload, gp_payloads, args.output, n_points=args.n_points, label_every=args.label_every)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
