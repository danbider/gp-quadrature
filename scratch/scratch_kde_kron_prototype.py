"""
Prototype: KDE-tapered Kronecker preconditioner.

Idea: the 1D Toeplitz blocks T_k used by Kron are built from the empirical axis
characteristic function v_k(δ). At small N these are noisy. Replace v_k with
KDE(Gaussian)-tapered version: multiply by exp(-2π² h_k² δ²), where h_k is the
Silverman bandwidth for axis k. Algebraically this is the CF of the Gaussian-KDE
estimate of p_k.

Implementation without touching efgpnd.py: monkey-patch
`efgpnd.create_kronecker_precond` with a wrapper that pre-tapers v_kernel along
each axis (taper is separable, so axis-k slice picks up only the axis-k factor).

Validates:
  (1) at small N (below the empirical N > m² threshold) KDE-Kron beats raw Kron
  (2) at large N KDE-Kron converges to raw Kron (bandwidth → 0 as N → ∞)
  (3) neither gets worse than Jacobi anywhere

Run: ~/myenv/bin/python -u scratch/scratch_kde_kron_prototype.py
"""
from __future__ import annotations
import sys, os, time, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import efgpnd as efgp_mod
from efgpnd import EFGPND
from kernels.squared_exponential import SquaredExponential
from vanilla_gp_sampling import sample_gp_rff
from utils.kernels import get_xis

torch.set_default_dtype(torch.float64)
DT = torch.float64


# ---------- KDE-tapered Kron: monkey-patchable wrapper -----------------------

_ORIG_CREATE_KRON = efgp_mod.create_kronecker_precond


def make_taper_wrapper(x_data: torch.Tensor, freq_step: float, bw_multiplier: float = 1.0):
    """Return a drop-in replacement for create_kronecker_precond that pre-tapers v_kernel."""
    N, d = x_data.shape
    sigmas = x_data.double().std(dim=0).tolist()
    # Silverman's rule: h = 1.06 * sigma * N^{-1/5}
    bws = [bw_multiplier * 1.06 * s * (N ** (-0.2)) for s in sigmas]

    def patched(ws, v_kernel, sigmasq_scalar, *args, **kw):
        d_ = kw.get("d", args[0] if len(args) >= 1 else None)
        # v_kernel is (L,)*d_. Apply separable Gaussian taper along each axis.
        # Taper at δ_k = (j - ctr) * freq_step is exp(-2π² h_k² δ_k²).
        L = v_kernel.shape[0]
        ctr = (L - 1) // 2
        idx = torch.arange(L, device=v_kernel.device, dtype=DT)
        deltas = (idx - ctr) * freq_step

        v_tapered = v_kernel
        for k in range(d_):
            tap = torch.exp(-2.0 * (math.pi ** 2) * (bws[k] ** 2) * deltas ** 2)
            shape = [1] * d_
            shape[k] = L
            v_tapered = v_tapered * tap.view(shape).to(v_kernel.dtype)

        return _ORIG_CREATE_KRON(ws, v_tapered, sigmasq_scalar, *args, **kw)

    return patched, bws


def install_kde_patch(x_data, freq_step, bw_multiplier=1.0):
    patched, bws = make_taper_wrapper(x_data, freq_step, bw_multiplier)
    efgp_mod.create_kronecker_precond = patched
    return bws


def uninstall_kde_patch():
    efgp_mod.create_kronecker_precond = _ORIG_CREATE_KRON


# ---------- Helper: infer freq_step for a (kernel, eps, L) setup -------------

def infer_freq_step(kernel, eps, L):
    xis_1d, h, mtot = get_xis(kernel_obj=kernel, eps=eps, L=L, use_integral=True,
                              l2scaled=False)
    return float((xis_1d[1] - xis_1d[0]).item()), int(mtot)


# ---------- Timing harness ---------------------------------------------------

def make_data(n, d=2, true_ls=0.05, true_var=1.0, true_noise=0.01, seed=1):
    torch.manual_seed(seed)
    x = torch.rand(n, d, dtype=DT)
    f = sample_gp_rff(x, length_scale=true_ls, variance=true_var,
                      num_features=2000, seed=0)
    torch.manual_seed(seed)
    y = f + torch.sqrt(torch.tensor(true_noise)) * torch.randn(f.numel(), dtype=DT)
    return x, y


def time_grad(x, y, precond, *, ls, var, sig2, K=3, warmup=1,
              eps=1e-3, cg_tol=1e-4, J=1, noise_floor=1e-5, cg_max=3000):
    d = x.shape[1]
    kernel = SquaredExponential(dimension=d, init_lengthscale=ls, init_variance=var)
    model = EFGPND(x, y, kernel=kernel, sigmasq=sig2, eps=eps,
                   estimate_params=False,
                   opts={"mean_cg_preconditioner_type": precond,
                         "max_cg_iterations": cg_max})
    for _ in range(warmup):
        model.compute_gradients(trace_samples=J, cg_tol=cg_tol, noise_floor=noise_floor)
    ts, iters = [], []
    for _ in range(K):
        t0 = time.perf_counter()
        model.compute_gradients(trace_samples=J, cg_tol=cg_tol, noise_floor=noise_floor)
        ts.append(time.perf_counter() - t0)
        s = model.last_gradient_stats
        iters.append(s.get("trace_cg_iters"))
    return dict(times=ts, iters=iters,
                M=model.last_gradient_stats.get("feature_count"))


def run_one(x, y, variant, *, ls, var, sig2, eps=1e-3, bw_multiplier=1.0):
    """variant ∈ {'kron', 'kron_kde', 'jacobi'}"""
    if variant == "kron_kde":
        kernel_for_grid = SquaredExponential(dimension=x.shape[1],
                                             init_lengthscale=ls, init_variance=var)
        L = float((x.max(0).values - x.min(0).values).max().item())
        if L <= 1e-9: L = 1.0
        freq_step, _ = infer_freq_step(kernel_for_grid, eps, L)
        bws = install_kde_patch(x, freq_step, bw_multiplier=bw_multiplier)
        try:
            r = time_grad(x, y, "kronecker", ls=ls, var=var, sig2=sig2, eps=eps)
            r['bws'] = bws
        finally:
            uninstall_kde_patch()
    elif variant == "kron":
        uninstall_kde_patch()
        r = time_grad(x, y, "kronecker", ls=ls, var=var, sig2=sig2, eps=eps)
    elif variant == "jacobi":
        uninstall_kde_patch()
        r = time_grad(x, y, "jacobi", ls=ls, var=var, sig2=sig2, eps=eps)
    else:
        raise ValueError(variant)
    return r


def mean(xs):
    xs = [v for v in xs if v is not None]
    return sum(xs) / len(xs) if xs else float('nan')


# ---------- Sweeps -----------------------------------------------------------

if __name__ == "__main__":
    LS, VAR, SIG2 = 0.01, 0.5, 1e-4
    N_grid = [5_000, 25_000, 100_000, 500_000]
    BW_MULTS = [0.01, 0.05, 0.2, 1.0]  # multiplier on Silverman

    print(f"KDE-Kron prototype: sweep bw_mult × N. ℓ={LS}, σ_f²={VAR}, σ²={SIG2}, d=2",
          flush=True)
    print(f"eps=1e-3, cg_tol=1e-4. K=3 + 1 warmup.\n", flush=True)
    print(f"{'N':>7s} {'variant':<12s}  {'bw':>8s}  "
          f"{'bw/ℓ':>5s}  {'wall(s)':>7s}  {'iters':>6s}", flush=True)
    print("-" * 60, flush=True)
    for n in N_grid:
        x, y = make_data(n, d=2)
        # raw kron baseline
        r = run_one(x, y, "kron", ls=LS, var=VAR, sig2=SIG2)
        t = sum(r['times']) / len(r['times']); it = mean(r['iters'])
        print(f"{n:>7d} {'kron':<12s}  {'-':>8s}  {'-':>5s}  {t:>7.3f}  {it:>6.1f}",
              flush=True)
        # kron_kde at various bandwidths
        for mult in BW_MULTS:
            r = run_one(x, y, "kron_kde", ls=LS, var=VAR, sig2=SIG2,
                        bw_multiplier=mult)
            t = sum(r['times']) / len(r['times']); it = mean(r['iters'])
            bw = r['bws'][0]
            print(f"{n:>7d} {'kron_kde':<12s}  {bw:>8.5f}  {bw/LS:>5.2f}  "
                  f"{t:>7.3f}  {it:>6.1f}", flush=True)
        # jacobi baseline
        r = run_one(x, y, "jacobi", ls=LS, var=VAR, sig2=SIG2)
        t = sum(r['times']) / len(r['times']); it = mean(r['iters'])
        print(f"{n:>7d} {'jacobi':<12s}  {'-':>8s}  {'-':>5s}  {t:>7.3f}  {it:>6.1f}",
              flush=True)
        print(flush=True)
