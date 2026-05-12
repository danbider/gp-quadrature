"""EFGP-only per-gradient-step timing at n = 1e7, 1e8 (d=2, SE kernel).

Reuses `time_efgp_step` and `make_data` from speed_test_efgp_vs_ski.py.
Merges results into the same JSON file so the final table has all five n.
Saves after every lengthscale so long runs can be resumed.
"""
import gc
import json
import sys
import time
import traceback
from pathlib import Path

# Allow imports from both the project root (efgpnd, kernels, utils) and from
# the sibling speed_test_efgp_vs_ski script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (_REPO_ROOT, _REPO_ROOT / "scratch"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch  # noqa: E402

import speed_test_efgp_vs_ski as base  # noqa: E402

NS = [10_000_000, 100_000_000]
N_TIMED_STEPS = 3  # fewer samples for large n so total runtime stays reasonable
OUT = base.OUT


def main():
    # Override timed step count on the imported module.
    base.N_TIMED_STEPS = N_TIMED_STEPS

    results = {"d": base.D, "epsilon": base.EPSILON,
               "lengthscales": base.LENGTHSCALES, "runs": {}}
    if OUT.exists():
        try:
            prior = json.load(open(OUT))
            results["runs"].update(prior.get("runs", {}))
            print(f"Loaded prior results for n={sorted(results['runs'].keys())}")
        except Exception as e:
            print(f"Could not load prior results: {e}")

    for n in NS:
        key = str(n)
        if key in results["runs"] and all(
            v.get("sec_per_step") is not None
            for v in results["runs"][key]["efgp"].values()
        ):
            print(f"n={n:,}: already complete, skipping")
            continue

        print(f"\n{'='*60}\nn = {n:,}  (EFGP only)\n{'='*60}", flush=True)
        t_data0 = time.time()
        x, y = base.make_data(n)
        print(f"  data built in {time.time()-t_data0:.1f}s  "
              f"(x={tuple(x.shape)} {x.dtype}, y={tuple(y.shape)} {y.dtype})", flush=True)

        efgp_entries = results["runs"].get(key, {}).get("efgp", {})
        for ls in base.LENGTHSCALES:
            if efgp_entries.get(str(ls), {}).get("sec_per_step") is not None:
                print(f"  EFGP ls={ls:<6}  (cached)", flush=True)
                continue
            try:
                t0 = time.time()
                dt, mtot = base.time_efgp_step(x, y, ls)
                print(f"  EFGP ls={ls:<6}  {dt:7.3f} s/step   mtot={mtot}   "
                      f"(wall {time.time()-t0:.1f}s)", flush=True)
                efgp_entries[str(ls)] = {"sec_per_step": dt, "mtot": mtot}
            except Exception as e:
                print(f"  EFGP ls={ls}  FAILED: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                efgp_entries[str(ls)] = {"sec_per_step": None,
                                         "error": f"{type(e).__name__}: {str(e)[:200]}"}

            results["runs"][key] = {
                "efgp": efgp_entries,
                "ski_sec_per_step": None,
                "ski_error": "skipped (EFGP-only run)",
            }
            with open(OUT, "w") as fh:
                json.dump(results, fh, indent=2)

        del x, y
        gc.collect()

    print("\n" + "=" * 60 + "\nSummary (all n)\n" + "=" * 60)
    print(f"{'n':>12}  {'d':>2}  {'SKI':>14}  {'EFGP range':>20}")
    for n_str, r in sorted(results["runs"].items(), key=lambda kv: int(kv[0])):
        ski = r.get("ski_sec_per_step")
        ski_s = (r.get("ski_error") or "").split(";")[0] if ski is None else f"{ski:.2f}"
        ski_s = ski_s[:14] if ski is None else ski_s
        efgp_vals = [v["sec_per_step"] for v in r["efgp"].values()
                     if v.get("sec_per_step") is not None]
        efgp_s = f"{min(efgp_vals):.2f} - {max(efgp_vals):.2f}" if efgp_vals else "—"
        print(f"{int(n_str):>12,}  {base.D:>2}  {ski_s:>14}  {efgp_s:>20}")


if __name__ == "__main__":
    main()
