"""
Single-fit worker for the PNAS scaling benchmark (regression, Gaussian SE-kernel GP).

Runs ONE (method, T) fit in its own process so an OOM / timeout is isolated and
catchable by the driver (which monitors RSS + wall-clock and kills this process).

Usage:  python worker.py <method> <T> <out_json>
  method in {efgp, sgpr49, sgpr1024, ski}

What each fit reports (status "ok"):
  {method, T, eval:"test", n_test, threads,
   time (= median t_total over reps), t_learn, t_predict, t_total,
   sec_all, sec_median, sec_spread, n_reps,
   rmse (latent-recovery RMSE at held-out TEST points), ls, var, noise,
   peak_rss_gb, status:"ok"}
On any exception writes status:"error" with the message. (OOM / timeout are
detected by the driver via kill, not here.)

Fairness notes
--------------
* Thread counts are pinned identically for every method (see THREADS below) BEFORE
  torch/numpy/finufft import, so EFGP (finufft) and the gpytorch baselines get the
  same CPU budget. finufft honours OMP_NUM_THREADS.
* Recovery is measured on a FIXED held-out test set (uniform on [0,1]^2, TEST_SEED),
  the same points for every (method, T); true f comes from the fixed RFF realization.
* Timing includes hyperparameter learning (50 iters) AND posterior prediction at the
  test points. Cheap fits (< CHEAP_SEC) are repeated K_REPS times -> median time.
"""
import os, sys, json, time, math, traceback, subprocess, statistics


# -------------------- pin the compute budget BEFORE numpy/torch/finufft --------------------
def _physical_cores() -> int:
    try:
        return int(subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"]).strip())
    except Exception:
        return os.cpu_count() or 1


THREADS = int(os.environ.get("BENCH_THREADS", _physical_cores()))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = str(THREADS)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
torch.set_num_threads(THREADS)

ROOT = Path(__file__).resolve().parents[2]   # repo root (paper_figures/scaling/worker.py)
sys.path.insert(0, str(ROOT))

# ---- ground truth (matches hyperparameter_comparison.ipynb synthetic setup) ----
TRUE_LS, TRUE_VAR, TRUE_NOISE, D = 0.05, 1.0, 0.01, 2
INIT_LS, INIT_VAR, INIT_NOISE = 0.3, 0.5, 0.3
MAX_ITERS = int(os.environ.get("BENCH_MAX_ITERS", 50))
J = 1
RFF_F = 4000          # random Fourier features for the (method-neutral) ground-truth draw
NEVAL = int(os.environ.get("BENCH_NEVAL", 50_000))  # held-out TEST points for latent-recovery nRMSE
RFF_SEED = 12345      # fixes the single latent realization across all T
TEST_SEED = 999       # fixes the held-out test inputs across all (method, T)

# repetition / cheap-fit policy (fairness: stabilise the fast fits)
CHEAP_SEC = float(os.environ.get("BENCH_CHEAP_SEC", 60.0))
K_REPS = int(os.environ.get("BENCH_K_REPS", 3))


# ------------------------------- ground-truth field -------------------------------
_RFF = None


def _rff_params():
    """Fixed RFF realization (frequencies/phases/weights). Same for all T and for test."""
    gf = torch.Generator().manual_seed(RFF_SEED)
    omega = torch.randn(RFF_F, D, generator=gf) / TRUE_LS
    phi = torch.rand(RFF_F, generator=gf) * (2 * math.pi)
    w = torch.randn(RFF_F, generator=gf)
    return omega, phi, w


def rff_field(x: torch.Tensor) -> torch.Tensor:
    """True latent f(x) for the fixed RFF realization -- defined everywhere on the domain.

    Both the training targets and the held-out test truth are evaluated through this
    single function, so 'recovery at test points' is exactly recovery of this f.
    (Swap this out for a non-Fourier generator to run the deferred generator ablation.)
    """
    global _RFF
    if _RFF is None:
        _RFF = _rff_params()
    omega, phi, w = _RFF
    scale = math.sqrt(2.0 * TRUE_VAR / RFF_F)
    f = torch.empty(x.shape[0])
    B = 25_000
    for i in range(0, x.shape[0], B):
        xi = x[i:i + B]
        f[i:i + B] = scale * (torch.cos(xi @ omega.T + phi) @ w)
    return f


def gen_data(T, seed=0):
    """Deterministic SE-GP draw. x is nested across T (first 10k of the 1M draw == the 10k draw)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(T, D, generator=g)
    f = rff_field(x)
    gn = torch.Generator().manual_seed(seed + 1)
    y = f + math.sqrt(TRUE_NOISE) * torch.randn(T, generator=gn)
    return x, y, f


def gen_test_points(n_test=NEVAL):
    """Fixed held-out test inputs (uniform on [0,1]^2, disjoint from training) + true f there."""
    g = torch.Generator().manual_seed(TEST_SEED)
    xe = torch.rand(n_test, D, generator=g)
    fe = rff_field(xe)
    return xe, fe


def rmse(fhat, f):
    """Standard RMSE = sqrt(mean((fhat - f)^2)) = ||fhat - f|| / sqrt(N). No demeaning."""
    a = np.asarray(fhat, dtype=np.float64).reshape(-1)
    b = np.asarray(f, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(a - b) / math.sqrt(b.size))


def peak_rss_gb():
    import resource
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kB
    return ru / 1e9 if sys.platform == "darwin" else ru / 1e6


# ------------------------------- fits -------------------------------
# Each fit returns a dict with t_learn / t_predict / t_total (learning + prediction at
# the TEST points) and the latent-recovery nrmse against fe (true f at test points).
def fit_efgp(x, y, f, xe, fe):
    from efgpnd import EFGPND
    from kernels.squared_exponential import SquaredExponential
    from torch.optim import Adam
    k = SquaredExponential(dimension=D, init_lengthscale=INIT_LS, init_variance=INIT_VAR)
    model = EFGPND(x, y, kernel=k, sigmasq=INIT_NOISE, eps=1e-3, estimate_params=False,
                   opts={"mean_cg_preconditioner_type": "kronecker",
                         "trace_cg_preconditioner": True,
                         "mean_cg_preconditioner": True})
    opt = Adam(model.parameters(), lr=0.3)
    t0 = time.time()
    for _ in range(MAX_ITERS):
        opt.zero_grad()
        model.compute_gradients(trace_samples=J, cg_tol=1e-4, noise_floor=1e-5,
                                refresh_grid_every=5)
        opt.step()
    t_learn = time.time() - t0
    t1 = time.time()
    mean, _ = model.predict(xe, return_variance=False)
    t_predict = time.time() - t1
    ls = model.kernel.get_hyper('lengthscale')
    var = model.kernel.get_hyper('variance')
    noise = float(model._gp_params.sig2)
    return {"t_learn": t_learn, "t_predict": t_predict, "t_total": t_learn + t_predict,
            "rmse": rmse(mean.detach().numpy(), fe.numpy()), "ls": ls, "var": var, "noise": noise}


def _predict_gpytorch(model, likelihood, xe):
    import gpytorch
    model.eval(); likelihood.eval()
    dt = next(model.parameters()).dtype
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), \
         gpytorch.settings.max_cholesky_size(0), gpytorch.settings.skip_posterior_variances(True):
        m = model(xe.to(dt)).mean
    return m.double()


def fit_sgpr_m(x, y, f, xe, fe, m):
    from utils.sgpr import fit_sgpr
    t0 = time.time()
    res = fit_sgpr(x, y, kernel='SE', num_inducing=m, max_iters=MAX_ITERS, lr=0.3,
                   init_lengthscale=INIT_LS, init_outputscale=INIT_VAR, init_noise=INIT_NOISE,
                   dtype=torch.float64, verbose=False)
    t_learn = time.time() - t0
    t1 = time.time()
    mean = _predict_gpytorch(res['model'], res['likelihood'], xe)
    t_predict = time.time() - t1
    h = res['history']
    return {"t_learn": t_learn, "t_predict": t_predict, "t_total": t_learn + t_predict,
            "rmse": rmse(mean.numpy(), fe.numpy()),
            "ls": h['lengthscale'][-1], "var": h['outputscale'][-1], "noise": h['noise'][-1]}


def fit_ski(x, y, f, xe, fe):
    from utils.ski import fit_ski_gp
    t0 = time.time()
    # max_preconditioner_size=0 disables the pivoted-Cholesky CG preconditioner: with the Toeplitz
    # structure + noise floor and loose cg_tolerance, CG converges in ~1 step, so the preconditioner
    # is pure overhead -- measured ~7x slower at N=10k with identical RMSE (and worse marginal lik).
    res = fit_ski_gp(x, y, kernel='SE', max_iters=MAX_ITERS, lr=0.3,
                     init_lengthscale=INIT_LS, init_outputscale=INIT_VAR, init_noise=INIT_NOISE,
                     dtype=torch.float32, num_trace_samples=J, cg_tolerance=1.0,
                     target_grid_points=10_000, max_preconditioner_size=0, verbose=False)
    t_learn = time.time() - t0
    t1 = time.time()
    mean = _predict_gpytorch(res['model'], res['likelihood'], xe)
    t_predict = time.time() - t1
    h = res['history']
    return {"t_learn": t_learn, "t_predict": t_predict, "t_total": t_learn + t_predict,
            "rmse": rmse(mean.numpy(), fe.numpy()), "grid_size": res.get("grid_size"),
            "ls": h['lengthscale'][-1], "var": h['outputscale'][-1], "noise": h['noise'][-1]}


FITS = {
    "efgp":     lambda x, y, f, xe, fe: fit_efgp(x, y, f, xe, fe),
    "sgpr49":   lambda x, y, f, xe, fe: fit_sgpr_m(x, y, f, xe, fe, 49),
    "sgpr1024": lambda x, y, f, xe, fe: fit_sgpr_m(x, y, f, xe, fe, 1024),
    "ski":      lambda x, y, f, xe, fe: fit_ski(x, y, f, xe, fe),
}


def main():
    method, T, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    rec = {"method": method, "T": T, "eval": "test", "n_test": NEVAL, "threads": THREADS}
    try:
        x, y, f = gen_data(T)
        xe, fe = gen_test_points()
        # first run
        r0 = FITS[method](x, y, f, xe, fe)
        times = [r0["t_total"]]
        n_reps = 1
        # repeat only cheap fits (can't repeat multi-hour fits)
        if r0["t_total"] < CHEAP_SEC:
            for _ in range(K_REPS - 1):
                rr = FITS[method](x, y, f, xe, fe)
                times.append(rr["t_total"])
                n_reps += 1
        med = float(statistics.median(times))
        rec.update(r0)
        rec["time"] = med          # canonical field the plotter reads (= median t_total)
        rec["t_total"] = med
        rec["sec_all"] = times
        rec["sec_median"] = med
        rec["sec_spread"] = float(max(times) - min(times)) if n_reps > 1 else 0.0
        rec["n_reps"] = n_reps
        rec["peak_rss_gb"] = peak_rss_gb()
        rec["status"] = "ok"
    except Exception as e:  # noqa
        rec.update(status="error", error=f"{type(e).__name__}: {e}",
                   traceback=traceback.format_exc(), peak_rss_gb=peak_rss_gb())
    Path(out).write_text(json.dumps(rec, indent=2))
    print(json.dumps({k: rec[k] for k in rec if k != "traceback"}), flush=True)


if __name__ == "__main__":
    main()
