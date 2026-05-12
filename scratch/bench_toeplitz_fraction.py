"""
Measure: what fraction of a gradient step is spent inside ToeplitzND.__call__?

Wraps ToeplitzND.__call__ with a timer+counter, runs compute_gradients,
reports: total time, toeplitz time, #applies, CG iters, and other dominant
stages via stats_out.

Run: ~/myenv/bin/python scratch/bench_toeplitz_fraction.py
"""
import gc
import sys
import time
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402

from efgpnd import EFGPND, ToeplitzND  # noqa: E402
from kernels.squared_exponential import SquaredExponential  # noqa: E402


D = 2
EPSILON = 1e-4
CG_TOL = 1e-5
NOISE_FLOOR = 1e-5
DTYPE = torch.float64


def make_data(n, seed=0):
    torch.manual_seed(seed)
    x = torch.rand(n, D, dtype=DTYPE)
    y = 0.5 * torch.randn(n, dtype=DTYPE)
    return x, y


class ToeplitzTimer:
    def __init__(self):
        self.t_total = 0.0
        self.calls = 0
        self.elements_applied = 0  # batch rows summed

    def attach(self):
        orig = ToeplitzND.__call__
        timer = self

        def wrapped(self_, x):
            t0 = time.perf_counter()
            out = orig(self_, x)
            timer.t_total += time.perf_counter() - t0
            timer.calls += 1
            # count "rows" applied for batched inputs
            if x.ndim > len(self_.ns):
                rows = x.shape[0]
            else:
                rows = 1
            timer.elements_applied += rows
            return out

        self._orig = orig
        ToeplitzND.__call__ = wrapped

    def detach(self):
        ToeplitzND.__call__ = self._orig


def run(n, ls, force_pow2, trace_samples=5):
    # monkey-patch the ToeplitzND default for this run
    orig_init = ToeplitzND.__init__

    def patched_init(self_, v, *, force_pow2=force_pow2, precompute_fft=True):
        orig_init(self_, v, force_pow2=force_pow2, precompute_fft=precompute_fft)

    ToeplitzND.__init__ = patched_init
    try:
        x, y = make_data(n)
        kernel = SquaredExponential(dimension=D, init_lengthscale=ls, init_variance=1.0)
        model = EFGPND(x, y, kernel=kernel, sigmasq=0.05, eps=EPSILON, estimate_params=False)

        # warmup
        model.compute_gradients(trace_samples=trace_samples, cg_tol=CG_TOL, noise_floor=NOISE_FLOOR)

        # timed
        timer = ToeplitzTimer()
        timer.attach()
        try:
            t0 = time.perf_counter()
            model.compute_gradients(trace_samples=trace_samples, cg_tol=CG_TOL, noise_floor=NOISE_FLOOR)
            t_total = time.perf_counter() - t0
        finally:
            timer.detach()

        stats = dict(model.last_gradient_stats)
        mtot = int(stats.get("mtot") or 0)
        return dict(
            n=n, ls=ls, force_pow2=force_pow2, mtot=mtot,
            t_total=t_total,
            t_toeplitz=timer.t_total,
            toeplitz_calls=timer.calls,
            toeplitz_rows=timer.elements_applied,
            frac_toeplitz=timer.t_total / t_total,
            mean_cg_iters=int(stats.get("mean_cg_iters") or 0),
            trace_cg_iters=int(stats.get("trace_cg_iters") or 0),
        )
    finally:
        ToeplitzND.__init__ = orig_init
        gc.collect()


def main():
    cases = [
        # small mtot regime (already measured)
        (100_000, 0.05),   # mtot=39
        # larger mtot — where flop-bound FFT savings should translate
        (100_000, 0.02),   # mtot ~90
        (100_000, 0.01),   # mtot ~180
        (1_000_000, 0.02), # mtot ~90, larger n (more NUFFT cost)
        (1_000_000, 0.01), # mtot ~180
    ]
    rows = []
    for n, ls in cases:
        for fp in (True, False):
            r = run(n, ls, fp, trace_samples=5)
            rows.append(r)
            print(
                f"n={r['n']:>7}  ls={r['ls']:<5}  mtot={r['mtot']:<4} fp={r['force_pow2']!s:<5}"
                f"  total={r['t_total']:6.3f}s  toep={r['t_toeplitz']:6.3f}s ({r['frac_toeplitz']*100:5.1f}%)"
                f"  calls={r['toeplitz_calls']:<5} rows={r['toeplitz_rows']:<6}"
                f"  mean_cg={r['mean_cg_iters']:<3} trace_cg={r['trace_cg_iters']:<3}",
                flush=True,
            )
    out_path = Path(__file__).with_suffix(".json")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print("\nsaved", out_path)


if __name__ == "__main__":
    main()
