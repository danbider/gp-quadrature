"""
Sweep lengthscale on OISST and 3D ball to see whether the Jacobi-beats-Kron
finding holds across a range of kernel smoothness levels, and whether the
kron+jac additive combo is robust across lengthscales.
"""

from __future__ import annotations
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scratch.bench_precond_support import (
    run_case, uniform_ball, gp_like_y, load_oisst_2d, DTYPE
)
import torch

PRECONDS = ("jacobi", "kronecker", "kron+jac")


def main():
    torch.set_default_dtype(DTYPE)
    results = []

    # --- 3D ball sweep ---
    n = 30_000
    x = uniform_ball(n, R=2.0, seed=0)
    for ls in [0.2, 0.4, 0.8]:
        y = gp_like_y(x, lengthscale=ls, seed=0)
        r = run_case(f"3d_ball_ls{ls}", x, y,
                     lengthscale=ls, variance=1.0, sigmasq=0.01,
                     eps=1e-3, cg_tol=1e-4, preconds=PRECONDS)
        results.append(r); print(json.dumps(r, indent=2))

    # --- OISST sweep ---
    n = 50_000
    x, y_raw = load_oisst_2d(n, seed=0)
    y = y_raw - y_raw.mean()
    for ls in [2.0, 5.0, 10.0, 20.0]:
        r = run_case(f"oisst_ls{ls}", x, y,
                     lengthscale=ls, variance=float(y.var().item()),
                     sigmasq=0.01, eps=1e-3, cg_tol=1e-4,
                     preconds=PRECONDS)
        results.append(r); print(json.dumps(r, indent=2))

    outp = Path(__file__).with_suffix(".json")
    outp.write_text(json.dumps(results, indent=2))
    print("wrote", outp)


if __name__ == "__main__":
    main()
