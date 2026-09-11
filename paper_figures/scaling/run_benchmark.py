"""
Driver for the PNAS scaling benchmark: EFGP vs SGPR(M=49) vs SGPR(M=1024) vs SKI
on Gaussian SE-kernel synthetic data at T = 10k, 100k, 250k, 500k, 1M.

Each (method, T) fit runs in a subprocess (worker.py). This driver monitors the
child's RSS and wall-clock and KILLS it if it exceeds the memory cap (-> "oom") or
the per-method time cap (-> "timeout"). Once a method drops out at some T, all larger
T for that method are marked with the same status without running (they can only be
worse). Results are written to the output JSON after EVERY fit, so progress is never
lost, and a running results table is printed after each fit.

Fairness / contention controls (the paper's claim rests on honest BASELINE timings):
  * Fits run strictly one at a time (never concurrent) in isolated subprocesses.
  * The worker pins identical thread counts for every method (see worker.THREADS).
  * A pre-flight quiescence gate waits for load/RAM to be sane before each fit.
  * Per-fit telemetry (max load, min available RAM, swap delta, peak RSS) is recorded,
    with contention_suspected / swap_suspected flags so a contaminated fit can be re-run.
  * A short cooldown between fits avoids thermal carry-over from a long baseline fit.

Recovery is measured at HELD-OUT TEST points and the reported time includes both
hyperparameter learning (50 iters) and posterior prediction (see worker.py).
"""
import os, re, sys, json, time, shutil, subprocess, signal, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent        # paper_figures/scaling/
ROOT = HERE.parents[1]                         # repo root
PY = str(Path.home() / "myenv" / "bin" / "python")
WORKER = str(HERE / "worker.py")
OUT = HERE / "scaling_data.json"               # canonical panel-D data (read by ../fig1_overview.py)
ARCHIVE = HERE / "scaling_data_trainEval_archive.json"  # backup of the old train-eval data

METHODS = ["efgp", "sgpr49", "sgpr1024", "ski"]
SIZES = [10_000, 100_000, 250_000, 500_000, 1_000_000]
# optional env overrides (handy for smoke tests / partial reruns)
if os.environ.get("BENCH_METHODS"):
    METHODS = os.environ["BENCH_METHODS"].split(",")
if os.environ.get("BENCH_SIZES"):
    SIZES = [int(s) for s in os.environ["BENCH_SIZES"].split(",")]
RSS_LIMIT_GB = float(os.environ.get("BENCH_RSS_LIMIT_GB", 14.0))  # of 17 GB physical; kill above -> "oom"
RESUME = os.environ.get("BENCH_RESUME") == "1"  # keep already-completed (method,T) in scaling_data.json
# per-method time caps: allow the slow baselines to finish (SKI@250k ~71min,
# SGPR-1024@500k ~3.76h), while EFGP/SGPR-49 stay tightly capped.
TIMEOUTS_S = {"efgp": 1800, "sgpr49": 1800, "sgpr1024": 18000, "ski": 7200}
POLL_S = float(os.environ.get("BENCH_POLL_S", 1.0))
# Optional swap-growth guard, OFF by default. NOTE: swap growth is measured system-wide, so a big
# fit ramping up makes macOS swap out *background* daemons, which trips this even though the fit
# itself would complete (it falsely OOM'd sgpr1024/ski @250k, which run fine under the RSS cap). The
# real guard is the RSS cap + fast poll below; enable this only if you actually want a swap ceiling.
SWAP_KILL_MB = float(os.environ.get("BENCH_SWAP_KILL_MB", 1e12))
# Last-resort absolute floor on available RAM (in case swap is disabled/full).
MIN_AVAIL_KILL_GB = float(os.environ.get("BENCH_MIN_AVAIL_KILL_GB", 0.3))
COOLDOWN_S = float(os.environ.get("BENCH_COOLDOWN_S", 15.0))   # thermal cooldown before each fit
LOAD_GATE = float(os.environ.get("BENCH_LOAD_GATE", 1.5))      # gate if load1 > LOAD_GATE * cores
MIN_FREE_GB = float(os.environ.get("BENCH_MIN_FREE_GB", 3.0))  # gate if available RAM below this
GATE_MAX_WAIT_S = float(os.environ.get("BENCH_GATE_MAX_WAIT_S", 120.0))


def _physical_cores() -> int:
    try:
        return int(subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"]).strip())
    except Exception:
        return os.cpu_count() or 1


CORES = _physical_cores()

CONFIG = {
    "study": "synthetic SE-kernel GP regression scaling (Fig 1D)",
    "methods": METHODS, "sizes": SIZES,
    "rss_limit_gb": RSS_LIMIT_GB, "timeouts_s": TIMEOUTS_S,
    "swap_kill_mb": SWAP_KILL_MB, "min_avail_kill_gb": MIN_AVAIL_KILL_GB, "poll_s": POLL_S,
    "contention_note": "all fits run serially, session-driven (Claude Code ~0.4GB, uniform); "
                       "primary crash guard = swap growth since fit-start > swap_kill_mb -> oom "
                       "(macOS reclaims cache before swapping, so this is the honest pre-OOM signal); "
                       "avail_ram floor + rss cap are last-resort guards",
    "true_ls": 0.05, "true_var": 1.0, "true_noise": 0.01, "d": 2,
    "init_ls": 0.3, "init_var": 0.5, "init_noise": 0.3,
    "max_iters": 50, "metric": "latent RMSE = ||fhat-f||/sqrt(N) at held-out test points",
    "eval": "test", "n_test": 50_000, "test_seed": 999, "rff_seed": 12345,
    "time_includes_prediction": True,
    "cores": CORES,
    "efgp": {"eps": 1e-3, "nufft_eps": 1e-4, "cg_tol": 1e-4, "noise_floor": 1e-5,
             "refresh_grid_every": 5, "preconditioner": "kronecker (mean+trace)",
             "trace_samples": 1, "optimizer": "Adam", "lr": 0.3, "dtype": "float64"},
    "ski": {"grid": "100x100 (target_grid_points=10000)",
            "grid_rule": "fixed; resolves lengthscale (spacing 0.01 vs ls=0.05); "
                         "= gpytorch choose_grid_size default at n=1e4",
            "cg_tolerance_train": 1.0, "eval_cg_tolerance": "0.01 (gpytorch default)",
            "num_trace_samples": 1, "max_preconditioner_size": 0,
            "preconditioner_note": "disabled: ~7x slower with identical RMSE at N=1e4 (pure overhead "
                                   "given Toeplitz structure + noise floor + loose CG tol)",
            "interpolation": "cubic (order 4)", "optimizer": "Adam", "lr": 0.3, "dtype": "float32"},
    "sgpr": {"num_inducing": [49, 1024], "learn_inducing_locations": True,
             "optimizer": "Adam", "lr": 0.3, "dtype": "float64"},
}


# ------------------------------- system telemetry -------------------------------
def rss_gb(pid):
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)],
                                      stderr=subprocess.DEVNULL)
        return int(out.strip()) / 1e6  # ps RSS is in KB -> GB
    except Exception:
        return 0.0


def load1():
    try:
        return os.getloadavg()[0]
    except Exception:
        return 0.0


def avail_ram_gb():
    """Available RAM in GB (psutil if present, else macOS vm_stat)."""
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:
        pass
    try:
        out = subprocess.check_output(["vm_stat"]).decode()
        page = 4096
        m = re.search(r"page size of (\d+) bytes", out)
        if m:
            page = int(m.group(1))
        free = re.search(r"Pages free:\s+(\d+)", out)
        inactive = re.search(r"Pages inactive:\s+(\d+)", out)
        spec = re.search(r"Pages speculative:\s+(\d+)", out)
        pages = sum(int(x.group(1)) for x in (free, inactive, spec) if x)
        return pages * page / 1e9
    except Exception:
        return None


def swap_used_mb():
    try:
        out = subprocess.check_output(["sysctl", "-n", "vm.swapusage"]).decode()
        m = re.search(r"used\s*=\s*([\d.]+)M", out)
        return float(m.group(1)) if m else 0.0
    except Exception:
        return 0.0


def stray_worker_pids(exclude_pid=None):
    """Other worker.py processes not spawned by us (leftover from a crashed run)."""
    try:
        out = subprocess.check_output(["pgrep", "-f", "scaling/worker.py"],
                                      stderr=subprocess.DEVNULL).decode()
        pids = [int(p) for p in out.split()]
        return [p for p in pids if p != exclude_pid and p != os.getpid()]
    except Exception:
        return []


def preflight(method, T):
    """Wait (bounded) for the machine to be quiescent; return a telemetry/warn dict."""
    strays = stray_worker_pids()
    if strays:
        print(f"[warn] stray worker.py processes {strays} — kill them for a clean run.", flush=True)
    t0 = time.time()
    warns = []
    while True:
        l1, avail = load1(), avail_ram_gb()
        hot = l1 > LOAD_GATE * CORES
        low = (avail is not None and avail < MIN_FREE_GB)
        if not (hot or low) or (time.time() - t0) > GATE_MAX_WAIT_S:
            if hot:
                warns.append(f"load1={l1:.1f} (> {LOAD_GATE * CORES:.0f})")
            if low:
                warns.append(f"avail_ram={avail:.1f}GB (< {MIN_FREE_GB})")
            return {"preflight_load1": l1, "preflight_avail_gb": avail,
                    "preflight_warns": warns, "stray_workers": strays}
        print(f"[gate] waiting for quiescence (load1={l1:.1f}, avail={avail}): {method} T={T:,}",
              flush=True)
        time.sleep(3.0)


def run_one(method, T):
    tf = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tf.close()
    pre = preflight(method, T)
    swap0 = swap_used_mb()
    timeout_s = TIMEOUTS_S.get(method, 1800)

    proc = subprocess.Popen([PY, WORKER, method, str(T), tf.name])
    t0 = time.time()
    status = None
    oom_reason = None
    peak = 0.0
    max_load = 0.0
    min_avail = float("inf")
    max_swap_delta = 0.0
    low_avail_hits = 0
    while True:
        ret = proc.poll()
        if ret is not None:
            break
        peak = max(peak, rss_gb(proc.pid))
        max_load = max(max_load, load1())
        a = avail_ram_gb()
        if a is not None:
            min_avail = min(min_avail, a)
        sd = swap_used_mb() - swap0
        max_swap_delta = max(max_swap_delta, sd)
        # primary guard: swap growth since fit start (the honest pre-OOM signal on macOS)
        if sd > SWAP_KILL_MB:
            status = "oom"; oom_reason = "swap_ceiling"
            proc.send_signal(signal.SIGKILL); break
        # last-resort absolute floor on available RAM (debounced against transient dips)
        if a is not None and a < MIN_AVAIL_KILL_GB:
            low_avail_hits += 1
            if low_avail_hits >= 3:
                status = "oom"; oom_reason = "avail_floor"
                proc.send_signal(signal.SIGKILL); break
        else:
            low_avail_hits = 0
        if peak > RSS_LIMIT_GB:
            status = "oom"; oom_reason = "rss_cap"
            proc.send_signal(signal.SIGKILL); break
        if time.time() - t0 > timeout_s:
            status = "timeout"; proc.send_signal(signal.SIGKILL); break
        time.sleep(POLL_S)
    try:
        proc.wait(timeout=30)
    except Exception:
        proc.kill()
    wall = time.time() - t0
    swap_delta = swap_used_mb() - swap0

    telemetry = {
        "peak_rss_gb": peak, "max_load1": max_load,
        "min_avail_gb": (None if min_avail == float("inf") else min_avail),
        "swap_delta_mb": swap_delta, "max_swap_delta_mb": max_swap_delta,
        "swap_suspected": max(swap_delta, max_swap_delta) > 200.0,
        "contention_suspected": max_load > LOAD_GATE * CORES,
        **pre,
    }

    if status in ("oom", "timeout"):
        return {"method": method, "T": T, "status": status, "eval": "test",
                "oom_reason": oom_reason, "wall_at_kill_s": wall, **telemetry}
    # process exited on its own -> read result file
    try:
        rec = json.loads(Path(tf.name).read_text())
    except Exception as e:
        rec = {"method": method, "T": T, "status": "error",
               "error": f"no result file: {e}", "exit_code": proc.returncode}
    rec.update(telemetry)
    Path(tf.name).unlink(missing_ok=True)
    return rec


# ------------------------------- running table -------------------------------
def _tlab(n):
    return f"{n // 1000}k" if n < 1_000_000 else f"{n // 1_000_000}M"


def print_table(results):
    by = {}
    for r in results:
        by[(r["method"], r["T"])] = r
    w = 16
    header = "method".ljust(10) + "".join(_tlab(s).rjust(w) for s in SIZES)
    print("\n" + header)
    print("-" * len(header))
    for m in METHODS:
        row = m.ljust(10)
        for s in SIZES:
            r = by.get((m, s))
            if r is None:
                cell = "·"
            elif r.get("status") == "ok":
                flag = "!" if (r.get("contention_suspected") or r.get("swap_suspected")) else ""
                cell = f"{r['time']:.2g}s/{r['rmse']:.3g}{flag}"
            else:
                cell = r.get("status", "?")
            row += cell.rjust(w)
        print(row)
    print("", flush=True)


def main():
    # back up the old train-eval data once (do not clobber an existing archive)
    if OUT.exists() and not ARCHIVE.exists():
        shutil.copy(OUT, ARCHIVE)
        print(f"[backup] {OUT.name} -> {ARCHIVE.name}", flush=True)

    print(f"[env] cores={CORES}  RSS_cap={RSS_LIMIT_GB}GB  timeouts={TIMEOUTS_S}  "
          f"cooldown={COOLDOWN_S}s  resume={RESUME}", flush=True)

    results = []
    dropped = {}  # method -> status at which it dropped out
    done = {}     # (method, T) -> saved record, when resuming
    if RESUME and OUT.exists():
        try:
            prev = json.loads(OUT.read_text()).get("results", [])
        except Exception:
            prev = []
        for r in prev:
            if r.get("eval") == "test" and "method" in r and "T" in r:
                done[(r["method"], r["T"])] = r
                if r.get("status") in ("oom", "timeout"):
                    dropped.setdefault(r["method"], r["status"])
        results = list(done.values())   # carry ALL completed fits forward, even ones not iterated below
        print(f"[resume] loaded {len(done)} completed fits from {OUT.name}", flush=True)

    for method in METHODS:
        for T in SIZES:
            if (method, T) in done:
                print(f"[keep] {method} T={T:,} -> {done[(method, T)].get('status')} (already done)",
                      flush=True)
                continue
            if method in dropped:
                rec = {"method": method, "T": T, "status": dropped[method], "eval": "test",
                       "note": "skipped (dropped out at smaller T)"}
                print(f"[skip] {method} T={T:,} -> {dropped[method]}", flush=True)
            else:
                if COOLDOWN_S > 0:
                    time.sleep(COOLDOWN_S)
                print(f"[run ] {method} T={T:,} ...", flush=True)
                rec = run_one(method, T)
                s = rec.get("status")
                if s == "ok":
                    flag = " [CONTENTION]" if (rec.get("contention_suspected")
                                               or rec.get("swap_suspected")) else ""
                    print(f"[ ok ] {method} T={T:,}  t={rec['time']:.2f}s "
                          f"(learn={rec.get('t_learn', float('nan')):.2f}+"
                          f"pred={rec.get('t_predict', float('nan')):.2f})  "
                          f"rmse={rec['rmse']:.5f}  reps={rec.get('n_reps', 1)}  "
                          f"peakRSS={rec.get('peak_rss_gb', 0):.1f}GB{flag}", flush=True)
                else:
                    print(f"[{s:^7}] {method} T={T:,}  {rec.get('error', '')}", flush=True)
                    if s in ("oom", "timeout"):
                        dropped[method] = s
            results.append(rec)
            OUT.write_text(json.dumps({"config": CONFIG, "results": results}, indent=2))
            print_table(results)
    print(f"\nDONE. Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
