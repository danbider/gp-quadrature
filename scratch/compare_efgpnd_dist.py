"""Distribution-of-final-hypers test.

Run the same Adam loop as the notebook on the same data with `nseeds`
different torch RNG seeds (which determine the Hutchinson probes) for both
OLD and CURRENT efgpnd. Compare the *distributions* of final hypers.

If both versions are algorithmically equivalent, mean(final_hyper_OLD) ≈
mean(final_hyper_CUR) within the per-version standard error.
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
    y = sample_gp_rff(x, length_scale=true_ls, variance=true_var, num_features=5000, seed=42)
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
    for it in range(max_iters):
        opt.zero_grad()
        model.compute_gradients(trace_samples=J, cg_tol=cg_tol, noise_floor=noise_floor)
        opt.step()
    return (model.kernel.get_hyper('lengthscale'),
            model.kernel.get_hyper('variance'),
            model._gp_params.sig2.item())


def main():
    cur = load(REPO / 'efgpnd.py', 'efgpnd_cur')
    old = load(Path('/tmp/efgpnd_compare/efgpnd_old.py'), 'efgpnd_old')
    x, y = make_data()
    cfg = dict(init_ls=0.3, init_var=0.5, init_noise=0.3,
               eps=1e-4, cg_tol=1e-5, lr=0.1, max_iters=50,
               J=1, noise_floor=1e-5)

    nseeds = 8
    seeds = list(range(nseeds))
    print(f"Running {nseeds} seeds × 2 versions × {cfg['max_iters']} iters with J=1 ...")

    rows = []
    t0 = time.time()
    for s in seeds:
        l_old, v_old, n_old = run_one(old.EFGPND, x, y, **cfg, seed=s)
        l_cur, v_cur, n_cur = run_one(cur.EFGPND, x, y, **cfg, seed=s)
        rows.append((s, l_old, v_old, n_old, l_cur, v_cur, n_cur))
        print(f"  seed={s:2d}  OLD ℓ={l_old:.5f} σ_f²={v_old:.4f} σ_n²={n_old:.4e}  | "
              f"CUR ℓ={l_cur:.5f} σ_f²={v_cur:.4f} σ_n²={n_cur:.4e}")
    print(f"  ...{time.time()-t0:.1f}s total")

    def mean_std(vals):
        t = torch.tensor(vals, dtype=torch.float64)
        return float(t.mean()), float(t.std()), float(t.std() / max(1, len(vals)) ** 0.5)

    l_olds = [r[1] for r in rows]; v_olds = [r[2] for r in rows]; n_olds = [r[3] for r in rows]
    l_curs = [r[4] for r in rows]; v_curs = [r[5] for r in rows]; n_curs = [r[6] for r in rows]

    print("\n" + "=" * 80)
    print(f"Distribution of final hypers across {nseeds} seeds (J=1):")
    print("=" * 80)
    for name, oldvals, curvals in [
        ('ℓ',      l_olds, l_curs),
        ('σ_f²',   v_olds, v_curs),
        ('σ_n²',   n_olds, n_curs),
    ]:
        mo, so, seo = mean_std(oldvals)
        mc, sc, sec = mean_std(curvals)
        diff_mean = mc - mo
        # 2-sided z-test under "both have same population mean"
        pooled_se = (seo**2 + sec**2) ** 0.5
        z = diff_mean / max(pooled_se, 1e-30)
        print(f"  {name:>5}:  OLD mean={mo:.5g} ± {seo:.3g}  CUR mean={mc:.5g} ± {sec:.3g}  Δ={diff_mean:+.3g}  z={z:+.2f}σ")


if __name__ == '__main__':
    main()
