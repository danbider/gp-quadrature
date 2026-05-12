"""Compare v0 (pre-Woodbury, 5-NUFFT, N-space Hutchinson) gradient against
v2 (current, 1-NUFFT, M-space BD' Hutchinson) gradient under the same
hyperparameter Adam loop as `hyperparameter_comparison.ipynb`.

v0 source: efgpnd.py at commit a88f3a9~1 (parent of `efgpnd gradient:
single-NUFFT path via M-space Woodbury`).
v2 source: working-tree efgpnd.py.

The motivation: the user has noticed that v2 occasionally lands at
different hypers than v0. v0 used N-space Rademacher probes and an extra
forward NUFFT for alpha; v2 collapses everything to M-space scalars +
BD' trace Hutchinson, claiming lower variance. This script pits them
head-to-head on the same data.
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


def load(path, alias):
    spec = importlib.util.spec_from_file_location(alias, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def make_data(n=100, d=2, true_ls=0.05, true_var=1.0, true_noise=0.01):
    dt = torch.float64
    torch.manual_seed(42)
    x = torch.rand(n, d, dtype=dt)
    y = sample_gp_rff(x, length_scale=true_ls, variance=true_var, num_features=10000, seed=42)
    torch.manual_seed(1)
    y = y + torch.sqrt(torch.tensor(true_noise, dtype=dt)) * torch.randn(n, dtype=dt)
    return x, y


def run_one(EFGPND_cls, x, y, *, init_ls, init_var, init_noise, eps, cg_tol,
            lr, max_iters, J, noise_floor, seed):
    d = x.shape[1]
    torch.manual_seed(seed)
    kernel = SquaredExponential(dimension=d, init_lengthscale=init_ls, init_variance=init_var)
    model = EFGPND_cls(x, y, kernel=kernel, sigmasq=init_noise, eps=eps, estimate_params=False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    log = {'iter': [0],
           'lengthscale': [model.kernel.get_hyper('lengthscale')],
           'variance':    [model.kernel.get_hyper('variance')],
           'sigmasq':     [model._gp_params.sig2.item()]}
    for it in range(max_iters):
        opt.zero_grad()
        model.compute_gradients(trace_samples=J, cg_tol=cg_tol, noise_floor=noise_floor)
        opt.step()
        log['iter'].append(it + 1)
        log['lengthscale'].append(model.kernel.get_hyper('lengthscale'))
        log['variance'].append(model.kernel.get_hyper('variance'))
        log['sigmasq'].append(model._gp_params.sig2.item())
    return log


def mean_std(vals):
    t = torch.tensor(vals, dtype=torch.float64)
    return float(t.mean()), float(t.std()), float(t.std() / max(1, len(vals)) ** 0.5)


def regime(label, *, true_ls, true_noise, init_ls, init_noise, n=100, d=2,
           true_var=1.0, init_var=0.5, lr=0.1, max_iters=50, J=1, nseeds=8,
           eps=1e-4, cg_tol=1e-5, noise_floor=1e-5, show_traj_seed=None):
    print("\n" + "#" * 80)
    print(f"# {label}")
    print(f"# true: ℓ={true_ls}  σ_f²={true_var}  σ_n²={true_noise}")
    print(f"# init: ℓ={init_ls}  σ_f²={init_var}  σ_n²={init_noise}")
    print(f"# n={n}  d={d}  lr={lr}  iters={max_iters}  J={J}  seeds={nseeds}")
    print("#" * 80)
    x, y = make_data(n=n, d=d, true_ls=true_ls, true_var=true_var, true_noise=true_noise)
    print(f"  data y.std={y.std():.4f}")

    v0 = sys.modules['efgpnd_v0']
    v2 = sys.modules['efgpnd_v2']
    cfg = dict(init_ls=init_ls, init_var=init_var, init_noise=init_noise,
               eps=eps, cg_tol=cg_tol, lr=lr, max_iters=max_iters,
               J=J, noise_floor=noise_floor)

    rows = []
    t0 = time.time()
    for s in range(nseeds):
        log0 = run_one(v0.EFGPND, x, y, **cfg, seed=s)
        log2 = run_one(v2.EFGPND, x, y, **cfg, seed=s)
        rows.append((s, log0, log2))
        l0, v0_, n0 = log0['lengthscale'][-1], log0['variance'][-1], log0['sigmasq'][-1]
        l2, v2_, n2 = log2['lengthscale'][-1], log2['variance'][-1], log2['sigmasq'][-1]
        print(f"  seed={s:2d}  v0 ℓ={l0:.5f} σ_f²={v0_:.4f} σ_n²={n0:.4e}  | "
              f"v2 ℓ={l2:.5f} σ_f²={v2_:.4f} σ_n²={n2:.4e}")
    print(f"  ...{time.time()-t0:.1f}s total")

    l_o = [r[1]['lengthscale'][-1] for r in rows]
    v_o = [r[1]['variance'][-1]    for r in rows]
    n_o = [r[1]['sigmasq'][-1]     for r in rows]
    l_c = [r[2]['lengthscale'][-1] for r in rows]
    v_c = [r[2]['variance'][-1]    for r in rows]
    n_c = [r[2]['sigmasq'][-1]     for r in rows]

    print("  ---- distribution summary (z = Δmean / pooled-SE):")
    for name, oldvals, curvals, truth in [
        ('ℓ',    l_o, l_c, true_ls),
        ('σ_f²', v_o, v_c, true_var),
        ('σ_n²', n_o, n_c, true_noise),
    ]:
        mo, so, seo = mean_std(oldvals); mc, sc, sec = mean_std(curvals)
        z = (mc - mo) / max((seo**2 + sec**2)**0.5, 1e-30)
        print(f"     {name:>5}: TRUE={truth:8.5g}   v0={mo:8.5g}±{seo:7.2g}   v2={mc:8.5g}±{sec:7.2g}   z={z:+.2f}σ")

    if show_traj_seed is not None:
        log0 = rows[show_traj_seed][1]
        log2 = rows[show_traj_seed][2]
        print(f"\n  ---- trajectory (seed={show_traj_seed}) ----")
        every = max(1, len(log0['iter']) // 12)
        last_i = len(log0['iter']) - 1
        for i in list(range(0, last_i + 1, every)) + [last_i]:
            print(f"    iter {log0['iter'][i]:4d} | "
                  f"v0 ℓ={log0['lengthscale'][i]:.5f}  σ_f²={log0['variance'][i]:.4f}  σ_n²={log0['sigmasq'][i]:.3e} | "
                  f"v2 ℓ={log2['lengthscale'][i]:.5f}  σ_f²={log2['variance'][i]:.4f}  σ_n²={log2['sigmasq'][i]:.3e}")


def main():
    v2 = load(REPO / 'efgpnd.py', 'efgpnd_v2')
    v0 = load(Path('/tmp/efgpnd_compare/efgpnd_v0.py'), 'efgpnd_v0')

    # Same as the notebook: n=100, d=2, true ℓ=0.05, true σ_n²=0.01
    regime("Notebook config (n=100, d=2, J=1)",
           true_ls=0.05, true_noise=0.01, init_ls=0.3, init_noise=0.3,
           n=100, d=2, lr=0.1, max_iters=50, J=1, nseeds=6,
           show_traj_seed=0)


if __name__ == '__main__':
    main()
