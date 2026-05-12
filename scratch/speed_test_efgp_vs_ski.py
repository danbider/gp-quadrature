"""Speed test: EFGP vs SKI per-gradient-step time (d=2, SE kernel).

For each n in {1e4, 1e5, 1e6}:
  - EFGP: sweep over a range of lengthscales, measure avg time/step.
  - SKI: measure avg time/step (constant in lengthscale). OOM is caught.
"""
import gc
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from efgpnd import EFGPND
from kernels.squared_exponential import SquaredExponential
from vanilla_gp_sampling import sample_gp_rff
from utils.ski import fit_ski_gp

OUT = Path(__file__).parent / "speed_test_efgp_vs_ski_results.json"

D = 2
NS = [10_000, 100_000, 1_000_000]
LENGTHSCALES = [0.3, 0.1, 0.05, 0.03, 0.02]
INIT_VAR = 1.0
INIT_NOISE = 0.05
LR = 0.1
EPSILON = 1e-4
CG_TOL = 1e-5
NOISE_FLOOR = 1e-5
N_TIMED_STEPS = 5
DTYPE = torch.float64


def make_data(n, seed=0):
    torch.manual_seed(seed)
    x = torch.rand(n, D, dtype=DTYPE)
    # Per-step cost is invariant to y's exact distribution; use cheap random y
    # so n=1e6 doesn't OOM while building a giant RFF feature matrix.
    y = 0.5 * torch.randn(n, dtype=DTYPE)
    return x, y


def time_efgp_step(x, y, ls):
    kernel = SquaredExponential(dimension=D, init_lengthscale=ls, init_variance=INIT_VAR)
    model = EFGPND(x, y, kernel=kernel, sigmasq=INIT_NOISE, eps=EPSILON, estimate_params=False)

    # We time compute_gradients (forward + trace + mean solve + backward).
    # No optimizer.step(): the kernel must stay pinned at `ls` so every
    # timed call measures the cost *at that lengthscale*.
    def _clear_grads():
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()

    # Warmup (first call includes FFT plan / allocation overhead).
    _clear_grads()
    model.compute_gradients(trace_samples=1, cg_tol=CG_TOL, noise_floor=NOISE_FLOOR)

    t0 = time.time()
    for _ in range(N_TIMED_STEPS):
        _clear_grads()
        model.compute_gradients(trace_samples=1, cg_tol=CG_TOL, noise_floor=NOISE_FLOOR)
    elapsed = (time.time() - t0) / N_TIMED_STEPS

    # Sanity-check: lengthscale must not have drifted.
    ls_final = float(model.kernel.get_hyper("lengthscale"))
    assert abs(ls_final - ls) < 1e-12, f"lengthscale drifted: {ls} -> {ls_final}"

    m_total = None
    try:
        m_total = int(model.last_gradient_stats.get("mtot") or 0)
    except Exception:
        pass

    del model, kernel
    gc.collect()
    return elapsed, m_total


def time_ski_step(x, y):
    # max_iters = 1 warmup + N_TIMED_STEPS timed; we discard iter 0.
    try:
        res = fit_ski_gp(
            x, y, kernel="SE",
            max_iters=N_TIMED_STEPS + 1,
            lr=LR,
            init_lengthscale=0.1,
            init_outputscale=INIT_VAR,
            init_noise=INIT_NOISE,
            num_trace_samples=1,
            cg_tolerance=1.0,
            dtype=torch.float32,
            verbose=False,
        )
    except (RuntimeError, MemoryError) as e:
        return None, f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"

    h = res["history"]
    fwd = h["forward_sec"][1:]
    bwd = h["backward_sec"][1:]
    per_step = [f + b for f, b in zip(fwd, bwd)]
    return float(np.mean(per_step)), None


def main():
    results = {"d": D, "epsilon": EPSILON, "lengthscales": LENGTHSCALES, "runs": {}}
    # Merge with existing results so partial reruns don't clobber prior work.
    if OUT.exists():
        try:
            prior = json.load(open(OUT))
            if "runs" in prior:
                results["runs"].update(prior["runs"])
                print(f"Loaded prior results for n={list(prior['runs'].keys())}")
        except Exception as e:
            print(f"Could not load prior results: {e}")

    ns_to_run = [n for n in NS if str(n) not in results["runs"]]
    if not ns_to_run:
        print("All n already computed; forcing full rerun.")
        ns_to_run = list(NS)
        results["runs"] = {}

    for n in ns_to_run:
        print(f"\n{'='*60}\nn = {n:,}\n{'='*60}", flush=True)
        x, y = make_data(n)

        efgp_entries = {}
        for ls in LENGTHSCALES:
            try:
                dt, mtot = time_efgp_step(x, y, ls)
                print(f"  EFGP ls={ls:<6}  {dt:7.3f} s/step   mtot={mtot}", flush=True)
                efgp_entries[str(ls)] = {"sec_per_step": dt, "mtot": mtot}
            except Exception as e:
                print(f"  EFGP ls={ls}  FAILED: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                efgp_entries[str(ls)] = {"sec_per_step": None,
                                         "error": f"{type(e).__name__}: {str(e)[:200]}"}

        # Save after EFGP in case SKI OOM-kills the whole process.
        results["runs"][str(n)] = {
            "efgp": efgp_entries,
            "ski_sec_per_step": None,
            "ski_error": "pending",
        }
        with open(OUT, "w") as fh:
            json.dump(results, fh, indent=2)

        ski_time, ski_err = time_ski_step(x, y)
        if ski_time is None:
            print(f"  SKI           FAILED: {ski_err}", flush=True)
        else:
            print(f"  SKI           {ski_time:7.3f} s/step", flush=True)

        results["runs"][str(n)] = {
            "efgp": efgp_entries,
            "ski_sec_per_step": ski_time,
            "ski_error": ski_err,
        }

        del x, y
        gc.collect()

        with open(OUT, "w") as fh:
            json.dump(results, fh, indent=2)

    print("\n" + "=" * 60 + "\nSummary\n" + "=" * 60)
    print(f"{'n':>10}  {'d':>2}  {'SKI':>10}  {'EFGP range':>20}")
    for n in NS:
        r = results["runs"][str(n)]
        ski = r["ski_sec_per_step"]
        ski_s = "OOM" if ski is None else f"{ski:.2f}"
        efgp_vals = [v["sec_per_step"] for v in r["efgp"].values() if v.get("sec_per_step") is not None]
        if efgp_vals:
            efgp_s = f"{min(efgp_vals):.2f} - {max(efgp_vals):.2f}"
        else:
            efgp_s = "—"
        print(f"{n:>10,}  {D:>2}  {ski_s:>10}  {efgp_s:>20}")

    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
