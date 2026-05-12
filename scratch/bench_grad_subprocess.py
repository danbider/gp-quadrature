"""
Run gradient benchmark in a SUBPROCESS per (n, ls, fp) config to avoid
cross-run state contamination (FFT plan cache, memory fragmentation, etc).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

CASES = [
    (100_000, 0.05),
    (100_000, 0.02),
    (100_000, 0.01),
    (1_000_000, 0.02),
]

WORKER = r"""
import sys, time, json
sys.path.insert(0, r"{root}")
import torch
from efgpnd import EFGPND, ToeplitzND
from kernels.squared_exponential import SquaredExponential

n, ls, fp = {n}, {ls}, {fp}
DTYPE = torch.float64

# monkey-patch
orig = ToeplitzND.__init__
def patched(self, v, *, force_pow2=fp, precompute_fft=True):
    orig(self, v, force_pow2=fp, precompute_fft=precompute_fft)
ToeplitzND.__init__ = patched

# wrap __call__ for timing
t_total_toep = [0.0]; n_calls = [0]
orig_call = ToeplitzND.__call__
def wrapped(self, x):
    t0 = time.perf_counter()
    out = orig_call(self, x)
    t_total_toep[0] += time.perf_counter() - t0
    n_calls[0] += 1
    return out
ToeplitzND.__call__ = wrapped

torch.manual_seed(0)
x = torch.rand(n, 2, dtype=DTYPE)
y = 0.5 * torch.randn(n, dtype=DTYPE)
kernel = SquaredExponential(dimension=2, init_lengthscale=ls, init_variance=1.0)
model = EFGPND(x, y, kernel=kernel, sigmasq=0.05, eps=1e-4, estimate_params=False)

# warmup
model.compute_gradients(trace_samples=5, cg_tol=1e-5, noise_floor=1e-5)
t_total_toep[0] = 0.0; n_calls[0] = 0

# timed
t0 = time.perf_counter()
for _ in range(3):
    t_total_toep[0] = 0.0; n_calls[0] = 0
    model.compute_gradients(trace_samples=5, cg_tol=1e-5, noise_floor=1e-5)
t_total = (time.perf_counter() - t0) / 3

stats = dict(model.last_gradient_stats)
mtot = int(stats.get('mtot') or 0)
result = dict(n=n, ls=ls, fp=fp, mtot=mtot,
              t_total=t_total, t_toep=t_total_toep[0],
              calls=n_calls[0],
              mean_cg=int(stats.get('mean_cg_iters') or 0),
              trace_cg=int(stats.get('trace_cg_iters') or 0))
print('RESULT:' + json.dumps(result))
"""

rows = []
for n, ls in CASES:
    for fp in (True, False):
        script = WORKER.format(root=str(_ROOT), n=n, ls=ls, fp=fp)
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=300,
        )
        wall = time.perf_counter() - t0
        if proc.returncode != 0:
            print(f"FAIL n={n} ls={ls} fp={fp}: {proc.stderr[-400:]}")
            continue
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT:"):
                r = json.loads(line[len("RESULT:"):])
                rows.append(r)
                print(
                    f"n={r['n']:>7} ls={r['ls']:<5} mtot={r['mtot']:<4} fp={r['fp']!s:<5}"
                    f" total={r['t_total']:6.3f}s toep={r['t_toep']:6.3f}s"
                    f" calls={r['calls']:<3} cg(m,t)=({r['mean_cg']},{r['trace_cg']})"
                    f" wall={wall:.1f}s",
                    flush=True,
                )

out_path = Path(__file__).with_suffix(".json")
with open(out_path, "w") as f:
    json.dump(rows, f, indent=2)
print("saved", out_path)
