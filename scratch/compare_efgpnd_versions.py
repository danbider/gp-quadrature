"""Compare current (working-tree) efgpnd.py against origin/dev efgpnd.py.

Reproduces the EFGP block from `hyperparameter_comparison.ipynb` exactly,
runs Adam on both versions with identical seeds and identical inputs, and
prints per-iter trajectories side-by-side.

Run: ~/myenv/bin/python scratch/compare_efgpnd_versions.py
"""
import importlib.util
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kernels.squared_exponential import SquaredExponential
from vanilla_gp_sampling import sample_gp_rff


def load_efgpnd(path: Path, alias: str):
    """Import an efgpnd.py file under a given module alias."""
    spec = importlib.util.spec_from_file_location(alias, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def make_data(n=100, d=2, true_ls=0.05, true_var=1.0, true_noise=0.01):
    dtype = torch.float64
    torch.manual_seed(42)
    x = torch.rand(n, d, dtype=dtype)
    y = sample_gp_rff(x, length_scale=true_ls, variance=true_var,
                     num_features=5000, seed=42)
    torch.manual_seed(1)
    y = y + torch.sqrt(torch.tensor(true_noise, dtype=dtype)) * torch.randn(n, dtype=dtype)
    return x, y


def run_efgp(EFGPND_cls, x, y, *, init_ls, init_var, init_noise, eps, cg_tol,
             lr, max_iters, J, noise_floor, seed=0, label="",
             reseed_each_iter=False, reseed_iter_base=10_000):
    d = x.shape[1]
    torch.manual_seed(seed)
    kernel = SquaredExponential(dimension=d, init_lengthscale=init_ls, init_variance=init_var)
    model = EFGPND_cls(x, y, kernel=kernel, sigmasq=init_noise, eps=eps, estimate_params=False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    log = {'iter': [0],
           'lengthscale': [model.kernel.get_hyper('lengthscale')],
           'variance': [model.kernel.get_hyper('variance')],
           'sigmasq': [model._gp_params.sig2.item()]}
    t0 = time.time()
    for it in range(max_iters):
        if reseed_each_iter:
            torch.manual_seed(reseed_iter_base + it)
        opt.zero_grad()
        model.compute_gradients(trace_samples=J, cg_tol=cg_tol, noise_floor=noise_floor)
        opt.step()
        log['iter'].append(it + 1)
        log['lengthscale'].append(model.kernel.get_hyper('lengthscale'))
        log['variance'].append(model.kernel.get_hyper('variance'))
        log['sigmasq'].append(model._gp_params.sig2.item())
    log['time'] = time.time() - t0
    log['label'] = label
    return log


def main():
    cur_path = REPO / 'efgpnd.py'
    old_path = Path('/tmp/efgpnd_compare/efgpnd_old.py')
    if not old_path.exists():
        raise SystemExit(f"Old version not found at {old_path}; run:\n  git show origin/dev:efgpnd.py > {old_path}")

    # Load CURRENT first under alias `efgpnd_cur`. We deliberately do NOT
    # import as `efgpnd` because the OLD file likely also self-imports under
    # that name in some places (it doesn't, but be safe).
    print("Loading current efgpnd.py ...")
    mod_cur = load_efgpnd(cur_path, "efgpnd_cur")
    print("Loading origin/dev efgpnd.py ...")
    mod_old = load_efgpnd(old_path, "efgpnd_old")

    print(f"\nCurrent EFGPND : {mod_cur.EFGPND}")
    print(f"Old     EFGPND : {mod_old.EFGPND}")

    x, y = make_data()
    print(f"\nData: n={x.shape[0]}, d={x.shape[1]}, y.std={y.std():.4f}")

    cfg = dict(
        init_ls=0.3, init_var=0.5, init_noise=0.3,
        eps=1e-4, cg_tol=1e-5, lr=0.1, max_iters=50,
        J=1, noise_floor=1e-5,
    )

    # ------- Pass 1: notebook config (J=1, no reseed) -------
    print("\n=========== Pass 1: notebook config (J=1, no reseed) ===========")
    print("\n--- Running OLD (origin/dev) ---")
    log_old = run_efgp(mod_old.EFGPND, x, y, **cfg, seed=0, label="OLD")
    print("\n--- Running CURRENT (working tree) ---")
    log_cur = run_efgp(mod_cur.EFGPND, x, y, **cfg, seed=0, label="CUR")

    # Side-by-side trajectory
    print("\n" + "=" * 110)
    print(f"{'iter':>5} | {'OLD ℓ':>10} {'CUR ℓ':>10} | {'OLD σ_f²':>10} {'CUR σ_f²':>10} | {'OLD σ_n²':>12} {'CUR σ_n²':>12} | {'Δσ_n²':>10}")
    print("=" * 110)
    every = max(1, len(log_old['iter']) // 25)
    for i in range(0, len(log_old['iter']), every):
        row_iter = log_old['iter'][i]
        ol = log_old['lengthscale'][i]; cl = log_cur['lengthscale'][i]
        ov = log_old['variance'][i]; cv = log_cur['variance'][i]
        on = log_old['sigmasq'][i]; cn = log_cur['sigmasq'][i]
        print(f"{row_iter:5d} | {ol:10.5f} {cl:10.5f} | {ov:10.5f} {cv:10.5f} | {on:12.6e} {cn:12.6e} | {(cn-on):+10.3e}")
    # Always show the last row
    if (len(log_old['iter']) - 1) % every != 0:
        i = len(log_old['iter']) - 1
        row_iter = log_old['iter'][i]
        ol = log_old['lengthscale'][i]; cl = log_cur['lengthscale'][i]
        ov = log_old['variance'][i]; cv = log_cur['variance'][i]
        on = log_old['sigmasq'][i]; cn = log_cur['sigmasq'][i]
        print(f"{row_iter:5d} | {ol:10.5f} {cl:10.5f} | {ov:10.5f} {cv:10.5f} | {on:12.6e} {cn:12.6e} | {(cn-on):+10.3e}")

    print("\n--- Final (Pass 1) ---")
    for tag, log in [('OLD', log_old), ('CUR', log_cur)]:
        print(f"  {tag}: ℓ={log['lengthscale'][-1]:.5f}  σ_f²={log['variance'][-1]:.5f}  σ_n²={log['sigmasq'][-1]:.6e}  ({log['time']:.1f}s)")
    print(f"  Δℓ    = {log_cur['lengthscale'][-1] - log_old['lengthscale'][-1]:+.3e}")
    print(f"  Δσ_f² = {log_cur['variance'][-1] - log_old['variance'][-1]:+.3e}")
    print(f"  Δσ_n² = {log_cur['sigmasq'][-1] - log_old['sigmasq'][-1]:+.3e}")

    # ------- Pass 2: same probes (reseed_each_iter=True, J=1) -------
    # Forces both versions to draw probes from the same RNG state at each step,
    # so any remaining divergence is *algorithmic*, not stochastic.
    print("\n=========== Pass 2: reseed RNG before every iter (J=1) ===========")
    print("\n--- Running OLD (origin/dev) ---")
    log_old2 = run_efgp(mod_old.EFGPND, x, y, **cfg, seed=0, label="OLD",
                       reseed_each_iter=True)
    print("\n--- Running CURRENT (working tree) ---")
    log_cur2 = run_efgp(mod_cur.EFGPND, x, y, **cfg, seed=0, label="CUR",
                       reseed_each_iter=True)
    print("\n  iter | OLDℓ        CURℓ        | OLDσ_f²    CURσ_f²    | OLDσ_n²       CURσ_n²       | Δσ_n²")
    every = max(1, len(log_old2['iter']) // 12)
    last_i = len(log_old2['iter']) - 1
    for i in list(range(0, last_i + 1, every)) + [last_i]:
        print(f"  {log_old2['iter'][i]:4d} | {log_old2['lengthscale'][i]:10.6f}  {log_cur2['lengthscale'][i]:10.6f} | "
              f"{log_old2['variance'][i]:9.5f}  {log_cur2['variance'][i]:9.5f} | "
              f"{log_old2['sigmasq'][i]:12.6e}  {log_cur2['sigmasq'][i]:12.6e} | "
              f"{(log_cur2['sigmasq'][i] - log_old2['sigmasq'][i]):+10.3e}")
    print("\n--- Final (Pass 2) ---")
    for tag, log in [('OLD', log_old2), ('CUR', log_cur2)]:
        print(f"  {tag}: ℓ={log['lengthscale'][-1]:.5f}  σ_f²={log['variance'][-1]:.5f}  σ_n²={log['sigmasq'][-1]:.6e}")
    print(f"  Δℓ    = {log_cur2['lengthscale'][-1] - log_old2['lengthscale'][-1]:+.3e}")
    print(f"  Δσ_f² = {log_cur2['variance'][-1] - log_old2['variance'][-1]:+.3e}")
    print(f"  Δσ_n² = {log_cur2['sigmasq'][-1] - log_old2['sigmasq'][-1]:+.3e}")

    # ------- Pass 3: J=16, no reseed (reduce trace noise) -------
    print("\n=========== Pass 3: J=16, no reseed (denoised trace) ===========")
    cfg3 = {**cfg, 'J': 16, 'max_iters': 30}
    print("\n--- Running OLD (origin/dev) ---")
    log_old3 = run_efgp(mod_old.EFGPND, x, y, **cfg3, seed=0, label="OLD")
    print("\n--- Running CURRENT (working tree) ---")
    log_cur3 = run_efgp(mod_cur.EFGPND, x, y, **cfg3, seed=0, label="CUR")
    print("\n--- Final (Pass 3) ---")
    for tag, log in [('OLD', log_old3), ('CUR', log_cur3)]:
        print(f"  {tag}: ℓ={log['lengthscale'][-1]:.5f}  σ_f²={log['variance'][-1]:.5f}  σ_n²={log['sigmasq'][-1]:.6e}  ({log['time']:.1f}s)")
    print(f"  Δℓ    = {log_cur3['lengthscale'][-1] - log_old3['lengthscale'][-1]:+.3e}")
    print(f"  Δσ_f² = {log_cur3['variance'][-1] - log_old3['variance'][-1]:+.3e}")
    print(f"  Δσ_n² = {log_cur3['sigmasq'][-1] - log_old3['sigmasq'][-1]:+.3e}")


if __name__ == '__main__':
    main()
