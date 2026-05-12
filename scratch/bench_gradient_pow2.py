"""
End-to-end gradient-step time: force_pow2=True vs False.

We already flipped the default in efgpnd.ToeplitzND to False; for the "before"
numbers we monkey-patch the init to hard-force True.

d=2 headline, sweep n and lengthscale (which controls mtot).

Run: ~/myenv/bin/python scratch/bench_gradient_pow2.py
"""
import gc
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402

from efgpnd import EFGPND, ToeplitzND  # noqa: E402
from kernels.squared_exponential import SquaredExponential  # noqa: E402


D = 2
NS = [10_000, 100_000, 1_000_000]
LENGTHSCALES = [0.1, 0.05, 0.03]  # sweep mtot
INIT_VAR = 1.0
INIT_NOISE = 0.05
EPSILON = 1e-4
CG_TOL = 1e-5
NOISE_FLOOR = 1e-5
N_TIMED = 3
DTYPE = torch.float64


def make_data(n, seed=0):
    torch.manual_seed(seed)
    x = torch.rand(n, D, dtype=DTYPE)
    y = 0.5 * torch.randn(n, dtype=DTYPE)
    return x, y


def time_grad(x, y, ls, force_pow2_override):
    """Time one gradient step with the given force_pow2 setting."""
    # Monkey-patch ToeplitzND's init default by wrapping.
    orig_init = ToeplitzND.__init__

    def patched_init(self, v, *, force_pow2=force_pow2_override, precompute_fft=True):
        orig_init(self, v, force_pow2=force_pow2_override, precompute_fft=precompute_fft)

    ToeplitzND.__init__ = patched_init
    try:
        kernel = SquaredExponential(dimension=D, init_lengthscale=ls, init_variance=INIT_VAR)
        model = EFGPND(x, y, kernel=kernel, sigmasq=INIT_NOISE, eps=EPSILON, estimate_params=False)

        def _clear():
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()

        # Warmup
        _clear()
        model.compute_gradients(trace_samples=1, cg_tol=CG_TOL, noise_floor=NOISE_FLOOR)

        t0 = time.time()
        for _ in range(N_TIMED):
            _clear()
            model.compute_gradients(trace_samples=1, cg_tol=CG_TOL, noise_floor=NOISE_FLOOR)
        elapsed = (time.time() - t0) / N_TIMED

        mtot = None
        try:
            mtot = int(model.last_gradient_stats.get("mtot") or 0)
        except Exception:
            pass

        del model, kernel
        gc.collect()
        return elapsed, mtot
    finally:
        ToeplitzND.__init__ = orig_init


def main():
    out = {"d": D, "eps": EPSILON, "cg_tol": CG_TOL, "runs": {}}
    out_path = Path(__file__).with_suffix(".json")

    for n in NS:
        x, y = make_data(n)
        print(f"\n=== n={n:,} ===", flush=True)
        out["runs"][str(n)] = {}
        for ls in LENGTHSCALES:
            row = {}
            for label, fp in (("pow2", True), ("smooth", False)):
                try:
                    t, mtot = time_grad(x, y, ls, fp)
                    row[label] = dict(sec=t, mtot=mtot)
                    print(f"  ls={ls:<5} {label:<6} {t:7.3f}s mtot={mtot}", flush=True)
                except Exception as e:
                    row[label] = dict(error=f"{type(e).__name__}: {str(e)[:120]}")
                    print(f"  ls={ls:<5} {label:<6} FAIL {e}", flush=True)
            if "pow2" in row and "smooth" in row and "sec" in row["pow2"] and "sec" in row["smooth"]:
                row["speedup"] = row["pow2"]["sec"] / row["smooth"]["sec"]
                print(f"  ls={ls:<5} speedup={row['speedup']:.2f}x", flush=True)
            out["runs"][str(n)][str(ls)] = row
            with open(out_path, "w") as fh:
                json.dump(out, fh, indent=2)
        del x, y
        gc.collect()

    print("\nsaved", out_path)


if __name__ == "__main__":
    main()
